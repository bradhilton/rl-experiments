import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import obstore
from openai import AsyncOpenAI
from openai._types import Body, Headers, Query
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
import os
from typing import Callable, TypedDict, Unpack


from .stream import consume_chat_completion_stream
from .utils import timeout


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


async def get_chat_completion(
    client: AsyncOpenAI,
    cache: bool = True,
    log_results: bool = True,
    on_chunk: Callable[[ChatCompletionChunk, ChatCompletion], None] | None = None,
    **create_params: Unpack[CreateParams],
) -> ChatCompletion:
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
    if log_results or on_chunk:
        stream = await client.chat.completions.create(**create_params, stream=True)
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

        def _on_chunk(chunk: ChatCompletionChunk, completion: ChatCompletion) -> None:
            if on_chunk:
                on_chunk(chunk, completion)
            if log_results and chunk.choices:
                try:
                    with timeout():
                        with open(log_file, "a") as f:
                            f.write(chunk.choices[0].delta.content or "")
                except TimeoutError:
                    pass  # Skip writing this chunk if it times out

        chat_completion = await consume_chat_completion_stream(
            stream,
            on_chunk=_on_chunk,
        )
    else:
        chat_completion = await client.chat.completions.create(**create_params)
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


def is_valid_chat_completion(chat_completion: ChatCompletion) -> bool:
    return bool(
        chat_completion.choices
        and all(
            choice.message.content or choice.message.refusal
            for choice in chat_completion.choices
        )
    )
