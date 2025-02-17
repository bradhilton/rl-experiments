import datetime
from itertools import cycle
import json
from openai.types.chat.chat_completion import Choice
import polars as pl
from pydantic import BaseModel
from typing import Callable, Iterable

from .tasks import Task


class ConnectionsGroup(BaseModel):
    level: int
    members: list[str]


class ConnectionsGame(BaseModel):
    board: dict[str, ConnectionsGroup]
    starting_board: list[list[str]]
    created_at: datetime.datetime
    id: int
    name: str


def get_connections_games() -> list[ConnectionsGame]:
    df = (
        pl.read_parquet("./data/corbt/connections-games/*.parquet")
        .sort("createdAt")
        .cast({"id": pl.Int64})
    )
    return [
        ConnectionsGame(
            board={
                group_name: ConnectionsGroup(
                    level=group_contents["level"],
                    members=group_contents["members"],
                )
                for group_name, group_contents in json.loads(d["board"]).items()
            },
            starting_board=json.loads(d["startingBoard"]),
            created_at=d["createdAt"],
            id=d["id"],
            name=d["name"],
        )
        for d in df.to_dicts()
    ]


def get_connections_tasks(
    games: list[ConnectionsGame], parse_answers_liberally: bool = False
) -> Iterable[Task]:
    prompts = [
        "Find groups of four items that share something in common. Output them in the following format: four total lines. On each line, there should be four comma-separated items. No additional text (like group titles or descriptions) should be in the output. Also, there should not be anything in your output before or after the solution.",
        "Group words that share a common thread. There are four words for each common thread. Output them in the following format: four total lines. On each line, there should be four comma-separated items. No additional text (like group titles or descriptions) should be in the output. Also, there should not be anything in your output before or after the solution.",
        "This is a puzzle. Create four groups of four. Words in each group fit under a specific category. Some categories might be defined by their use of wordplay (palindromes, homophones, adding or dropping letters and words) rather than the literal meanings of the words on the cards. Output them in the following format: four total lines. On each line, there should be four comma-separated items. No additional text (like group titles or descriptions) should be in the output. Also, there should not be anything in your output before or after the solution.",
    ]
    for game, (prompt, lowercase) in zip(
        cycle(games),
        cycle((prompt, lowercase) for prompt in prompts for lowercase in [True, False]),
    ):
        yield Task(
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\nWords:\n\n{"\n".join(word.lower() if lowercase else word for row in game.starting_board for word in row)}",
                }
            ],
            grader=get_grader(game, parse_answers_liberally),
        )


def get_grader(
    game: ConnectionsGame, parse_answers_liberally: bool
) -> Callable[[Choice], float]:
    def grader(choice: Choice) -> float:
        assistant_content = choice.message.content
        assert assistant_content is not None
        assistant_content = assistant_content.strip()
        parsed_groups = get_parsed_groups(
            lines=assistant_content.split("\n")[-len(game.board) :],
            parse_answers_liberally=parse_answers_liberally,
        )
        return sum(
            (
                1
                for group in game.board.values()
                if frozenset(group.members) in parsed_groups
            )
        ) / len(game.board)

    return grader


def get_parsed_groups(
    lines: list[str], parse_answers_liberally: bool
) -> list[frozenset[str]]:
    if not parse_answers_liberally:
        return [
            frozenset(word.strip().upper() for word in line.split(","))
            for line in lines
        ]
    groups = []
    for line in lines:
        line = line.split("//")[0]
        line = line.split(" - ")[0]
        words = [
            word.strip()
            .strip("'\"")
            .replace("1. ", "")
            .replace("2. ", "")
            .replace("3. ", "")
            .replace("4. ", "")
            .replace("<eos>", "")
            .upper()
            for word in line.split(",")
        ]
        words = [word.split("(")[0] for word in words]
        words = [
            (
                max(word.split(":"), key=lambda part: part.count(","))
                if ":" in word
                else word
            )
            for word in words
        ]
        words = [
            (
                word.split(".", 1)[1].strip()
                if "." in word and len(word.split(".", 1)) > 1
                else word
            )
            for word in words
        ]
        words = [word.split(" - ")[0] for word in words]
        groups.append(frozenset(words[:4]))
    return groups
