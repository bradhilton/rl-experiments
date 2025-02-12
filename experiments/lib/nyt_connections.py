import datetime
import json
import polars as pl
from pydantic import BaseModel


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
