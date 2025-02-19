from lib.tasks import Task
from openai.types.chat.chat_completion import Choice
import polars as pl
import random
import re
from typing import Any, Iterable


def get_zebra_grid_tasks() -> Iterable[Task]:
    zebra_grid_puzzles = pl.read_parquet(
        "data/allenai/ZebraLogicBench-private/grid_mode/test-00000-of-00001.parquet"
    ).to_dicts()
    random.seed(42)
    random.shuffle(zebra_grid_puzzles)

    for puzzle in zebra_grid_puzzles:
        prompt = f"""{puzzle["puzzle"]}
Fill in the grid with the correct values:

| {' | '.join(puzzle["solution"]["header"])} |
| {' | '.join(["-" * len(header) for header in puzzle["solution"]["header"]])} |"""

        for _ in puzzle["solution"]["rows"]:
            prompt += f"| {' | '.join([" " * len(header) for header in puzzle["solution"]["header"]])} |\n"

        pattern = re.compile(
            r"\| " + r"\|".join(r"(.*?)" for _ in puzzle["solution"]["header"]) + r" \|"
        )

        def grader(
            choice: Choice,
            puzzle: dict[str, Any] = puzzle,
            pattern: re.Pattern[str] = pattern,
        ) -> float:
            content = choice.message.content
            assert content is not None and isinstance(content, str)
            num_cells = sum(len(row) for row in puzzle["solution"]["rows"])
            num_cells = sum(len(row) for row in puzzle["solution"]["rows"])
            num_correct = 0
            for match, row in zip(
                re.findall(pattern, content)[-len(puzzle["solution"]["rows"]) :],
                puzzle["solution"]["rows"],
            ):
                for cell, value in zip(match, row):
                    if cell.strip().lower() == value.lower():
                        num_correct += 1
            return num_correct / num_cells

        yield Task(messages=[{"role": "user", "content": prompt}], grader=grader)
