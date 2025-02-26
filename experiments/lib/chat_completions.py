import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import hashlib
import json
import obstore
from openai import AsyncOpenAI
from openai._types import Body, Headers, Query
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.completion_usage import CompletionUsage
import os
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    Protocol,
    TypedDict,
    Unpack,
)


from .stream import consume_chat_completion_stream
from .utils import timeout

MAX_INT = 2**31 - 1


class ClientParams(TypedDict):
    base_url: str
    organization: str | None
    project: str | None


class CreateParams(CompletionCreateParamsBase, total=False):
    extra_headers: Headers
    extra_query: Query
    extra_body: Body


class ChatCompletionRequest(TypedDict):
    client_params: ClientParams
    create_params: CreateParams


data_dir = os.path.abspath(os.getenv("DATA_DIR", "./data"))
chat_completions_dir = os.path.join(data_dir, "chat-completions")
chat_completion_logs_dir = os.path.join(data_dir, "chat-completion-logs")

os.makedirs(chat_completions_dir, exist_ok=True)
os.makedirs(chat_completion_logs_dir, exist_ok=True)
stores = [
    obstore.store.from_url(
        url=url,
        client_options={"timeout": timedelta(minutes=10)},
    )
    for url in (
        url
        for url in (
            # "memory:///",
            f"file://{data_dir}",
            os.getenv("OBJECT_STORE_URL"),
        )
        if url
    )
]


class TokenScheduler(Protocol):
    def tokens(self, params: CreateParams) -> AsyncContextManager[int]: ...

    def is_finished(
        self,
        chat_completion: ChatCompletion,
        params: CreateParams,
        max_completion_tokens: int,
    ) -> bool: ...


class UnlimitedTokenScheduler(TokenScheduler):
    def tokens(self, _: CreateParams) -> AsyncContextManager[int]:
        @asynccontextmanager
        async def unlimited_tokens() -> AsyncGenerator[int, None]:
            try:
                yield 0
            finally:
                pass

        return unlimited_tokens()

    def is_finished(
        self,
        _: ChatCompletion,
        __: CreateParams,
        ___: int,
    ) -> bool:
        return True


async def get_chat_completion(
    client: AsyncOpenAI,
    cache: bool = True,
    log_results: bool = True,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    token_scheduler: TokenScheduler | None = None,
    **create_params: Unpack[CreateParams],
) -> ChatCompletion:
    """
    Given a client and arguments to openai.chat.completions.create, this function will return a chat completion with some additional features:
    - Caching of results to object storage
    - Logging of results to a file
    - Streaming of results to the callback function
    - Support for capping concurrent requests with a semaphore
    - Advanced support for token scheduling and breaking up long completions

    Args:
        client (AsyncOpenAI): An AsyncOpenAI client
        cache (bool): Whether to cache the results of the chat completion
        log_results (bool): Whether to log the results of the chat completion
        on_chunk (Callable[[ChatCompletionChunk, ChatCompletion], None]): A callback function that will be called with each chunk of the chat completion
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent requests
        token_scheduler (TokenScheduler): A token scheduler to limit the number of tokens used in the completion

    Returns:
        ChatCompletion: A chat completion
    """
    request = ChatCompletionRequest(
        client_params=ClientParams(
            base_url=str(client.base_url),
            organization=client.organization,
            project=client.project,
        ),
        create_params=create_params,
    )
    cache_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if cache:
        results = await asyncio.gather(
            *[
                obstore.get_async(store, f"chat-completions/{cache_key}.json")
                for store in stores
            ],
            return_exceptions=True,
        )
        if result := next(
            (r for r in results if not isinstance(r, BaseException)), None
        ):
            json_data = bytes(await result.bytes_async())
            chat_completion = ChatCompletion.model_validate_json(json_data)
            if is_valid_chat_completion(chat_completion):
                for store, result in zip(stores, results):
                    if isinstance(result, FileNotFoundError):
                        await obstore.put_async(
                            store,
                            f"chat-completions/{cache_key}.json",
                            json_data,
                        )
                return chat_completion
    async with semaphore or asyncio.Semaphore():
        if log_results or on_chunk:
            log_file = os.path.join(
                chat_completion_logs_dir, f"{datetime.now().isoformat()}.log"
            )
            if log_results:
                with open(log_file, "w") as f:
                    f.write(
                        "".join(
                            f"{message['role'].capitalize()}:\n{message.get('content', '')}\n\n"
                            for message in create_params["messages"]
                        )
                        + "Assistant:\n"
                    )

            def _on_chunk(
                chunk: ChatCompletionChunk, completion: ChatCompletion
            ) -> None:
                if on_chunk:
                    on_chunk(chunk, completion)
                if log_results and chunk.choices:
                    try:
                        with timeout():
                            with open(log_file, "a") as f:
                                f.write(chunk.choices[0].delta.content or "")
                    except TimeoutError:
                        pass  # Skip writing this chunk if it times out

        else:
            _on_chunk = None  # type: ignore
        chat_completion = await _get_chat_completion(
            client,
            create_params,
            token_scheduler or UnlimitedTokenScheduler(),
            _on_chunk,
        )
    if cache:
        json_data = chat_completion.model_dump_json().encode()
        for store in stores:
            await obstore.put_async(
                store,
                f"chat-completions/{cache_key}.json",
                json_data,
            )
        with open("./data/chat-completion-requests.jsonl", "a") as f:
            f.write(json.dumps(request, sort_keys=True) + "\n")
    return chat_completion


async def _get_chat_completion(
    client: AsyncOpenAI,
    create_params: CreateParams,
    token_scheduler: TokenScheduler,
    _on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
) -> ChatCompletion:
    create_params = create_params.copy()
    async with token_scheduler.tokens(create_params) as max_completion_tokens:
        _create_params = create_params.copy()
        if max_completion_tokens:
            _create_params["max_completion_tokens"] = _create_params["max_tokens"] = (
                min(
                    _create_params.get(
                        "max_completion_tokens",
                        _create_params.get("max_tokens", MAX_INT),
                    )
                    or MAX_INT,
                    max_completion_tokens,
                )
            )
        if _on_chunk:
            stream = await client.chat.completions.create(**_create_params, stream=True)
            chat_completion = await consume_chat_completion_stream(
                stream,
                on_chunk=_on_chunk,
            )
        else:
            chat_completion = await client.chat.completions.create(**_create_params)
    if token_scheduler.is_finished(
        chat_completion, create_params, max_completion_tokens
    ):
        return chat_completion
    messages = create_params["messages"] = list(create_params["messages"])
    # TODO: Support multiple choices
    assistant_content = chat_completion.choices[0].message.content
    assert assistant_content is not None
    extra_body = create_params.setdefault("extra_body", {})
    assert isinstance(extra_body, dict)
    if extra_body.get("continue_final_message"):
        assert messages[-1]["role"] == "assistant"
        messages[-1] = messages[-1].copy()
        messages[-1]["content"] = assistant_content
    else:
        messages.append(
            {
                "role": "user",
                "content": assistant_content,
            }
        )
        extra_body["add_generation_prompt"] = False
        extra_body["continue_final_message"] = True
    new_completion = await _get_chat_completion(
        client,
        create_params,
        token_scheduler,
        _on_chunk,
    )
    _merge_chat_completions(chat_completion, new_completion)
    return new_completion


def _merge_chat_completions(
    original: ChatCompletion,
    into: ChatCompletion,
) -> None:
    for original_choice, into_choice in zip(original.choices, into.choices):
        _merge_choices(original_choice, into_choice)
    if original.usage and into.usage:
        _merge_usage(original.usage, into.usage)


def _merge_choices(original: Choice, into: Choice) -> None:
    if original.logprobs and into.logprobs:
        _merge_choice_logprobs(original.logprobs, into.logprobs)
    if original.message.content and into.message.content:
        into.message.content = original.message.content + into.message.content


def _merge_choice_logprobs(original: ChoiceLogprobs, into: ChoiceLogprobs) -> None:
    if original.content and into.content:
        into.content = original.content + into.content
    if original.refusal and into.refusal:
        into.refusal = original.refusal + into.refusal


def _merge_usage(original: CompletionUsage, into: CompletionUsage) -> None:
    into.prompt_tokens = original.prompt_tokens
    into.completion_tokens += original.completion_tokens
    into.total_tokens = into.prompt_tokens + into.completion_tokens


def is_valid_chat_completion(chat_completion: ChatCompletion) -> bool:
    return bool(
        chat_completion.choices
        and all(
            choice.message.content or choice.message.refusal
            for choice in chat_completion.choices
        )
    )
