from aioitertools.helpers import maybe_await
import asyncio
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
import sys
from typing import Awaitable, Callable, TypeVar

from .chat_completions import get_chat_completion
from .tqdm import tqdm

Grader = Callable[[Choice], float | Awaitable[float]]


@dataclass
class Task:
    messages: list[ChatCompletionMessageParam]
    grader: Grader


@dataclass
class TaskResult:
    task: Task
    chat_completion: ChatCompletion
    reward: float


T = TypeVar("T")


async def get_task_results(
    tasks: list[Task],
    client: AsyncOpenAI,
    model: str,
    transform: Callable[[TaskResult], T | Awaitable[T]] = lambda x: x,
) -> list[T]:
    pbar = tqdm.tqdm(total=len(tasks))
    stats = TaskResultStats(pbar=pbar)

    async def get_task_result(task: Task, client: AsyncOpenAI, model: str) -> T:
        chat_completion = await get_chat_completion(
            client,
            on_chunk=lambda chunk, _: stats.update(id=chunk.id, chunk=chunk),
            messages=task.messages,
            model=model,
            max_tokens=2**17,
            logprobs=True,
            top_logprobs=5,
        )
        stats.update(id=chat_completion.id, usage=chat_completion.usage)
        reward = await maybe_await(task.grader(chat_completion.choices[0]))
        stats.update(id=chat_completion.id, reward=reward)
        return await maybe_await(
            transform(
                TaskResult(
                    task=task,
                    chat_completion=chat_completion,
                    reward=reward,
                )
            )
        )

    return await asyncio.gather(
        *(get_task_result(task, client, model) for task in tasks)
    )


@dataclass
class TaskResultStats:
    pbar: tqdm.tqdm
    completion_tokens: int = 0
    grades: int = 0
    new_completion_ids: set[str] = field(default_factory=set)
    new_completion_tokens: int = 0
    new_prompt_tokens: int = 0
    prompt_tokens: int = 0
    token_logprobs: int = 0
    total_reward: float = 0
    usages: int = 0

    def update(
        self,
        *,
        id: str,
        chunk: ChatCompletionChunk | None = None,
        usage: CompletionUsage | None = None,
        reward: float | None = None,
    ) -> None:
        if chunk:
            self.new_completion_ids.add(id)
            self.token_logprobs += sum(
                len(choice.logprobs.content or choice.logprobs.refusal or [])
                for choice in chunk.choices
                if choice.logprobs
            )
        elif usage:
            self.completion_tokens += usage.completion_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.usages += 1
            if id in self.new_completion_ids:
                self.new_completion_tokens += usage.completion_tokens
                self.new_prompt_tokens += usage.prompt_tokens
        elif reward is not None:
            self.grades += 1
            self.total_reward += reward
            self.pbar.update()
        postfix = {
            "completion_tokens": round(self.completion_tokens / max(self.usages, 1)),
            "prompt_tokens": self.prompt_tokens / max(self.usages, 1),
            "reward": self.total_reward / max(self.grades, 1),
        }
        if self.token_logprobs:
            postfix["token_logprobs"] = self.token_logprobs
        self.pbar.set_postfix(postfix)
