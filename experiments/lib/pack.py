# https://github.com/bradhilton/q4-2024-atreides/blob/main/experiments/lib/pack.py

from collections import Counter
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Optional, TypedDict, Unpack

from .completion import Completion
from .episode import Episode
from ..tokenizer import Tokenizer
from ..utils import get_token, Timer, truncate_pad


class PackedTensors(TypedDict):
    tokens: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    input_pos: torch.Tensor
    ids: torch.Tensor
    model_ids: torch.Tensor


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int


class PackedDataset(Dataset[PackedTensors]):
    def __init__(self, **kwargs: Unpack[DiskPackedTensors]) -> None:
        self.tensors = packed_tensors_from_dir(**kwargs)

    def __len__(self) -> int:
        return self.tensors["tokens"].shape[0]

    def __getitem__(self, index: int) -> PackedTensors:
        return {key: tensor[index] for key, tensor in self.tensors.items()}  # type: ignore


def packed_tensors_from_dir(**kwargs: Unpack[DiskPackedTensors]) -> PackedTensors:
    os.makedirs(kwargs["dir"], exist_ok=True)
    return {
        key: torch.from_file(
            f"{kwargs["dir"]}/{key}.pt",
            shared=True,
            size=kwargs["num_sequences"]
            * kwargs["sequence_length"]
            * (kwargs["sequence_length"] if key == "mask" else 1),
            dtype=dtype,
        )
        .view(kwargs["num_sequences"], kwargs["sequence_length"], -1)
        .squeeze()
        for key, dtype in {
            "tokens": torch.long,
            "values": torch.float32,
            "advantages": torch.float32,
            "logprobs": torch.float32,
            "reference_logprobs": torch.float32,
            "weights": torch.float32,
            "mask": torch.bool,
            "input_pos": torch.long,
            "ids": torch.long,
            "model_ids": torch.long,
        }.items()
    }  # type: ignore


def packed_tensors_to_dir(tensors: PackedTensors, dir: str) -> DiskPackedTensors:
    os.makedirs(dir, exist_ok=True)
    disk_packed_tensors: DiskPackedTensors = {
        "dir": dir,
        "num_sequences": tensors["tokens"].shape[0],
        "sequence_length": tensors["tokens"].shape[1],
    }
    for key, tensor in packed_tensors_from_dir(**disk_packed_tensors).items():
        tensor.copy_(tensors[key])  # type: ignore
    return disk_packed_tensors


def packed_tensors(
    episodes: list[Episode],
    model: str,
    sample_probability_power: float,
    sequence_length: int,
    tokenizer: Tokenizer,
    trajectories_per_episode: Optional[int],
) -> PackedTensors:
    with Timer("Packed sequences"):
        sequences, completion_weights = packed_sequences(
            episodes,
            model,
            sample_probability_power,
            sequence_length,
            tokenizer,
            trajectories_per_episode,
        )
    max_ancestors = (
        max(episode.completion.max_depth({model}) for episode in episodes) + 1
    )
    with Timer("Prepared tensors"):
        completion_tensors = {
            completion: get_completion_tensors(
                completion, prev_completion, weight, model, tokenizer, max_ancestors
            )
            for (completion, weight), prev_completion in zip(
                completion_weights.items(), [None] + list(completion_weights.keys())
            )
        }
        tensors = {
            key: torch.stack(
                [
                    truncate_pad(
                        torch.cat(
                            [
                                completion_tensors[completion][key]
                                for completion in sequence
                            ]
                        ),
                        [sequence_length],
                        mode="constant",
                        value=pad_value,
                    )
                    for sequence in sequences
                ]
            )
            for key, pad_value in {
                "tokens": tokenizer.get_pad_token_id() or 0,
                "values": torch.nan,
                "advantages": torch.nan,
                "logprobs": torch.nan,
                "reference_logprobs": torch.nan,
                "weights": 0.0,
                "ids": 0,
                "ancestor_ids": 0,
                "input_pos": 0,
            }.items()
        }
        tensors["values"] = (tensors["values"] - torch.nanmean(tensors["values"])) / (
            torch.std(tensors["values"][~torch.isnan(tensors["values"])]) or 1
        )
        tensors["advantages"] = (
            tensors["advantages"] - torch.nanmean(tensors["advantages"])
        ) / (torch.std(tensors["advantages"][~torch.isnan(tensors["advantages"])]) or 1)
    with Timer("Created mask"):
        mask = get_mask(tensors["ids"], tensors["ancestor_ids"])
    return {
        "tokens": tensors["tokens"],
        "values": tensors["values"],
        "advantages": tensors["advantages"],
        "logprobs": tensors["logprobs"],
        "reference_logprobs": tensors["reference_logprobs"],
        "weights": tensors["weights"],
        "mask": mask,
        "input_pos": tensors["input_pos"],
        "ids": tensors["ids"],
        "model_ids": tensors["model_ids"],
    }


def packed_sequences(
    episodes: list[Episode],
    model: str,
    sample_probability_power: float,
    sequence_length: int,
    tokenizer: Tokenizer,
    trajectories_per_episode: Optional[int],
) -> tuple[list[list[Completion]], dict[Completion, float]]:
    sequences: list[Counter[Completion]] = []
    completions: Counter[Completion] = Counter()
    for episode in episodes:
        termini: list[Completion] = []
        possible_termini = episode.completion.leaves(models={model})
        if trajectories_per_episode is not None:
            possible_termini = list(possible_termini)
            possible_termini = random.choices(
                possible_termini,
                weights=[
                    leaf.sample_weight(
                        cache=True, models={model}, power=sample_probability_power
                    )
                    for leaf in possible_termini
                ],
                k=trajectories_per_episode,
            )
        for terminus in possible_termini:
            for terminus in terminus.ancestors(including_self=True):
                if (
                    terminus.advantage(cache=True, models={model}) != 0
                    and terminus.token_count(tokenizer, cache=True) <= sequence_length
                ):
                    break
            else:
                continue
            termini.append(terminus)
        for terminus in termini:
            while True:
                for completion in terminus.ancestors(including_self=True, reverse=True):
                    completions[completion] += 1
                    if (
                        sum(c.token_count(tokenizer, cache=True) for c in completions)
                        > sequence_length
                    ):
                        for c in completion.ancestors(including_self=True):
                            completions[c] -= 1
                        sequences.append(completions)
                        completions = Counter()
                        break
                else:
                    break
    sequences.append(completions)
    for completions in sequences:
        for completion, count in list(completions.items()):
            if count == 0:
                del completions[completion]
    total_occurances = sum(sequences, Counter())
    sequence_occurences = Counter(
        completion for completions in sequences for completion in completions
    )
    weights: dict[Completion, float] = {
        completion: (
            (
                (
                    total_occurances[completion]
                    if trajectories_per_episode is not None
                    else completion.sample_weight(
                        cache=True, models={model}, power=sample_probability_power
                    )
                )
                / sequence_occurences[completion]
            )
            if completion.advantage(cache=True, models={model}) != 0
            else 0
        )
        for completion in total_occurances
    }
    average_weight = sum(weights.values()) / len(weights)
    weights = {
        completion: weight / average_weight for completion, weight in weights.items()
    }
    return [list(sequence) for sequence in sequences], weights


def get_completion_tensors(
    completion: Completion,
    prev_completion: Optional[Completion],
    weight: float,
    model: str,
    tokenizer: Tokenizer,
    max_ancestors: int,
) -> dict[str, torch.Tensor]:
    tokens, mask = completion.tokens_and_mask(tokenizer, cache=True)
    values = torch.full_like(mask, fill_value=torch.nan, dtype=torch.float32)
    value = completion.value(cache=True, models={model})
    values[mask] = torch.tensor([value for _ in range(mask.sum())])
    advantages = torch.full_like(mask, fill_value=torch.nan, dtype=torch.float32)
    advantages[mask] = torch.tensor(
        [
            advantage
            for advantage in completion.token_advantages(cache=True, models={model})
        ]
    )
    logprobs = torch.full_like(mask, fill_value=torch.nan, dtype=torch.float32)
    logprobs[mask] = torch.tensor([logprob for logprob in completion.logprobs()])
    reference_logprobs = (completion.reference_logprobs or logprobs).clone()
    if not prev_completion is completion.parent:
        values[0] = advantages[0] = logprobs[0] = reference_logprobs[0] = torch.nan
    ancestor_ids = [
        id(ancestor) for ancestor in completion.ancestors(including_self=True)
    ]
    ancestor_ids += [ancestor_ids[-1]] * (max_ancestors - len(ancestor_ids))
    start_pos_id = (
        completion.parent.all_token_count(tokenizer, cache=True)
        if completion.parent
        else 0
    )
    return {
        "tokens": tokens,
        "values": values,
        "advantages": advantages,
        "logprobs": logprobs,
        "reference_logprobs": reference_logprobs,
        "weights": torch.tensor([weight for _ in range(tokens.shape[0])]),
        "ids": torch.tensor([id(completion) for _ in range(tokens.shape[0])]),
        "ancestor_ids": torch.tensor([ancestor_ids for _ in range(tokens.shape[0])]),
        "input_pos": torch.tensor(
            [i for i in range(start_pos_id, tokens.shape[0] + start_pos_id)]
        ),
    }


def get_replacement_token(
    tokens: torch.Tensor, tokenizer: Tokenizer
) -> tuple[str, int]:
    max_token = int(tokens.max().item())
    try:
        return tokenizer.get_token(max_token + 1), max_token + 1
    except:
        for i in range(max_token - 1, 0, -1):
            if i in tokens:
                continue
            try:
                return tokenizer.get_token(i), i
            except:
                continue
    raise ValueError("No replacement token found")


def get_mask(ids: torch.Tensor, ancestor_ids: torch.Tensor) -> torch.Tensor:
    """Creates an attention mask for hierarchical attention based on node IDs and their ancestor IDs.

    Args:
        ids: A tensor of shape (batch_size, sequence_length) containing node IDs
        ancestor_ids: A tensor of shape (batch_size, sequence_length, max_ancestors) containing ancestor IDs for each node
            including itself, padded with zeros

    Returns:
        A boolean tensor of shape (batch_size, sequence_length, sequence_length) where True indicates
        allowed attention connections. Each position can attend to itself and any of its ancestors
        in the hierarchy, but only for previous positions (due to causal masking).
    """
    # Compare each position against all ancestors of each other position
    # Shape: (batch, seq, seq, max_ancestors)
    mask = ids.unsqueeze(1).unsqueeze(3) == ancestor_ids.unsqueeze(2)
    # Reduce over ancestors dimension to get final mask
    # Shape: (batch, seq, seq)
    mask = mask.any(dim=3)
    # Apply causal mask
    mask &= torch.tril(torch.ones_like(mask, dtype=torch.bool, device=ids.device))
    return mask
