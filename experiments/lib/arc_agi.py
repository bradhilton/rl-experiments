import glob
import json
from openai.types.chat.chat_completion import Choice
from typing import Any, Iterable

from .tasks import Task


def get_arc_agi_tasks(partial_credit: float = 0.0) -> Iterable[Task]:
    for file in glob.glob("./data/fchollet/ARC-AGI/data/training/*.json"):
        task = json.load(open(file))

        def grader(choice: Choice, task: dict[str, Any] = task) -> float:
            content = choice.message.content
            assert isinstance(content, str)

            # Parse the model's output into a grid
            try:
                predicted_grid = [
                    [int(cell) for cell in line.split()]
                    for line in content.strip().split("\n")
                ]

                # Get the expected output grid
                expected_grid = task["test"][0]["output"]

                # Check if dimensions match
                if len(predicted_grid) != len(expected_grid) or any(
                    len(pred_row) != len(exp_row)
                    for pred_row, exp_row in zip(predicted_grid, expected_grid)
                ):
                    return 0.0

                # Check if grids match exactly
                if predicted_grid == expected_grid:
                    return 1.0
                else:
                    return (
                        partial_credit
                        * sum(
                            sum(
                                cell == exp_cell
                                for cell, exp_cell in zip(pred_row, exp_row)
                            )
                            for pred_row, exp_row in zip(predicted_grid, expected_grid)
                        )
                        / (len(predicted_grid) * len(predicted_grid[0]))
                    )

            except (ValueError, IndexError):
                return 0.0

        yield Task(
            messages=[{"role": "user", "content": get_prompt(task)}],
            grader=grader,
        )


def get_prompt(task: dict[str, Any]) -> str:
    """Format an ARC AGI task into a prompt string similar to the example."""
    # Build prompt with header
    prompt = "Find the common rule that maps an input grid to an output grid, given the examples below.\n\n"

    # Add training examples
    for i, example in enumerate(task["train"], 1):
        prompt += f"Example {i}:\n\n"
        prompt += "Input:\n"
        prompt += "\n".join(
            " ".join(str(cell) for cell in row) for row in example["input"]
        )
        prompt += "\n"
        prompt += "Output:\n"
        prompt += "\n".join(
            " ".join(str(cell) for cell in row) for row in example["output"]
        )
        prompt += "\n\n"

    # Add test input
    prompt += "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. "
    prompt += "Your final answer should just be the text output grid itself.\n\n"
    prompt += "Input:\n"
    prompt += "\n".join(
        " ".join(str(cell) for cell in row) for row in task["test"][0]["input"]
    )

    return prompt
