from aioitertools.helpers import maybe_await
import asyncio
from dataclasses import dataclass, field
import numpy as np
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
import random
from typing import Awaitable, Callable, Never, TypeVar

from .chat_completions import CreateParams, get_chat_completion
from .tqdm import tqdm


class ChatCompletionParams(CreateParams, total=False):
    messages: Never
    model: Never


# A grader function returns a floating-point reward,
# and optionally a dictionary of metrics as the second return value
Grade = float | tuple[float, dict[str, float]]
Grader = Callable[[Choice], Grade | Awaitable[Grade]]


@dataclass
class Task:
    messages: list[ChatCompletionMessageParam]
    grader: Grader


@dataclass
class TaskResult:
    task: Task
    chat_completions: list[ChatCompletion]
    rewards: dict[tuple[str, int], float]
    metrics: dict[tuple[str, int], dict[str, float]]
    advantages: dict[tuple[str, int], float]
    exceptions: list[Exception]


T = TypeVar("T")


class TaskResults(list[T]):
    stats: "TaskResultStats"


async def get_task_results(
    tasks: list[Task],
    client: AsyncOpenAI,
    model: str,
    cache: bool = True,
    clear_pbar: bool = False,
    print_pbar: bool = True,
    log_results: bool | float | int = True,
    log_token_logprobs: bool = True,
    n: int = 1,
    params: ChatCompletionParams | None = None,
    pbar_desc: str | None = None,
    prices: tuple[float, float] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    transform: Callable[[TaskResult], T | Awaitable[T]] = lambda x: x,
) -> TaskResults[T]:
    num_completions = len(tasks) * n
    pbar = tqdm.tqdm(total=num_completions, desc=pbar_desc)
    stats = TaskResultStats(pbar=pbar, prices=prices)

    async def get_task_result(
        task: Task, client: AsyncOpenAI, model: str, log_results: bool
    ) -> T:
        _params = (params or {}).copy()
        if "logprobs" not in _params:
            _params["logprobs"] = True
        chat_completions: list[ChatCompletion] = []
        rewards: dict[tuple[str, int], float] = {}
        metrics: dict[tuple[str, int], dict[str, float]] = {}
        exceptions: list[Exception] = []
        for chat_completion_future in asyncio.as_completed(
            get_chat_completion(
                client,
                cache=cache,
                log_results=log_results and i == 0,
                on_chunk=(
                    (lambda chunk, _: stats.update(id=chunk.id, chunk=chunk))
                    if log_token_logprobs
                    else None
                ),
                semaphore=semaphore,
                messages=task.messages,
                model=model,
                **_params,  # type: ignore
            )
            for i in range(n)
        ):
            try:
                chat_completion = await chat_completion_future
                chat_completions.append(chat_completion)
                stats.update(id=chat_completion.id, usage=chat_completion.usage)

                async def _grade(choice: Choice, grader: Grader) -> tuple[int, Grade]:
                    return choice.index, await maybe_await(grader(choice))

                for grade_future in asyncio.as_completed(
                    _grade(choice, task.grader) for choice in chat_completion.choices
                ):
                    try:
                        choice_index, grade = await grade_future
                        reward, _metrics = (
                            grade if isinstance(grade, tuple) else (grade, {})
                        )
                        stats.update(
                            id=chat_completion.id, reward=reward, metrics=_metrics
                        )
                        rewards[chat_completion.id, choice_index] = reward
                        metrics[chat_completion.id, choice_index] = _metrics
                    except Exception as e:
                        exceptions.append(e)
                        stats.update(id=chat_completion.id, exception=e)
                        continue
            except Exception as e:
                exceptions.append(e)
                stats.update(id=None, exception=e)
                continue
        if rewards:
            reward_mean = np.mean(list(rewards.values()))
            reward_std = np.std(list(rewards.values()))
            advantages = {
                key: float((reward - reward_mean) / (reward_std + 1e-6))
                for key, reward in rewards.items()
            }
        else:
            advantages = {key: 0.0 for key in rewards.keys()}
        return await maybe_await(
            transform(
                TaskResult(
                    task=task,
                    chat_completions=chat_completions,
                    rewards=rewards,
                    metrics=metrics,
                    advantages=advantages,
                    exceptions=exceptions,
                )
            )
        )

    if isinstance(log_results, int) and log_results >= 1:
        _log_results = [True] * log_results + [False] * max(len(tasks) - log_results, 0)
    elif isinstance(log_results, float):
        _log_results = [True] * int(log_results * len(tasks)) + [False] * (
            len(tasks) - int(log_results * len(tasks))
        )
    else:
        _log_results = [bool(log_results)] * len(tasks)
    random.shuffle(_log_results)
    results = TaskResults(
        await asyncio.gather(
            *(
                get_task_result(task, client, model, log_result)
                for task, log_result in zip(tasks, _log_results)
            )
        )
    )
    results.stats = stats
    pbar.close()
    if getattr(pbar, "container", None) and clear_pbar:
        pbar.container.close()
    if getattr(pbar, "container", None) and print_pbar:
        print(pbar.container.__repr__(pretty=True))
    return results


@dataclass
class TaskResultStats:
    pbar: tqdm.tqdm
    prices: tuple[float, float] | None
    completion_tokens: int = 0
    exceptions: int = 0
    grades: int = 0
    new_completion_ids: set[str] = field(default_factory=set)
    new_completion_tokens: int = 0
    new_prompt_tokens: int = 0
    prompt_tokens: int = 0
    token_logprobs: int = 0
    total_metrics: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0
    usages: int = 0

    def __del__(self) -> None:
        self.pbar.close()

    def update(
        self,
        *,
        id: str | None,
        chunk: ChatCompletionChunk | None = None,
        usage: CompletionUsage | None = None,
        reward: float | None = None,
        metrics: dict[str, float] | None = None,
        exception: Exception | None = None,
    ) -> None:
        if chunk:
            if id is not None:
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
            if metrics:
                for key, value in metrics.items():
                    if key not in self.total_metrics:
                        self.total_metrics[key] = 0
                    self.total_metrics[key] += value
        elif exception:
            self.exceptions += 1
        postfix = {
            "completion_tokens": round(self.completion_tokens / max(self.usages, 1)),
            "prompt_tokens": round(self.prompt_tokens / max(self.usages, 1)),
            "reward": self.total_reward / max(self.grades, 1),
        }
        for key, value in self.total_metrics.items():
            postfix[key] = value / max(self.grades, 1)
        if self.prices:
            postfix["spend"] = (
                f"${(
                self.new_prompt_tokens * self.prices[0]
                + (self.token_logprobs or self.new_completion_tokens) * self.prices[1]
            ) / 1_000_000:.2f}"
            )
        if self.token_logprobs:
            postfix["token_logprobs"] = self.token_logprobs
        if self.exceptions:
            postfix["exceptions"] = self.exceptions
        self.pbar.set_postfix(postfix)
