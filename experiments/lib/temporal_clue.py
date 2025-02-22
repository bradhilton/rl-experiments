import json
import math
from openai.types.chat.chat_completion import Choice
import re
from typing import Iterable, TypedDict

from .tasks import Task


class TemporalCluePuzzle(TypedDict):
    num_clues: int
    prompt: str
    solution: dict[str, str]


def get_temporal_clue_puzzles() -> list[TemporalCluePuzzle]:
    return json.load(open("./data/bradhilton/temporal-clue/puzzles.json"))


def get_temporal_clue_tasks(surprise_bonus: float = 0.0) -> Iterable[Task]:
    for puzzle in get_temporal_clue_puzzles():

        def grader(
            choice: Choice, puzzle: TemporalCluePuzzle = puzzle
        ) -> float | tuple[float, dict[str, float]]:
            content = choice.message.content
            assert isinstance(content, str)
            num_correct = 0
            for key, value in puzzle["solution"].items():
                if matches := re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content):
                    match = matches[-1]
                    if match.strip().lower() == value.lower():
                        num_correct += 1
            reward = num_correct / len(puzzle["solution"])
            if surprise_bonus != 0.0 and choice.logprobs:
                if logprobs := [
                    token_logprob.logprob
                    for token_logprob in choice.logprobs.content
                    or choice.logprobs.refusal
                    or []
                    if token_logprob.logprob is not None
                    and not math.isnan(token_logprob.logprob)
                ]:
                    surprise = -sum(logprobs) / len(logprobs)
                    return reward + surprise_bonus * surprise, dict(
                        acc=reward, surprise=surprise
                    )
            return reward

        yield Task(
            messages=[
                {
                    "role": "user",
                    "content": puzzle["prompt"],
                }
            ],
            grader=grader,
        )
