import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
from torch.utils.data import Dataset
from typing import TypedDict, Unpack

from .tokenize import TokenizedResult


class PackedTensors(TypedDict):
    tokens: torch.Tensor
    group_ids: torch.Tensor
    input_pos: torch.Tensor
    assistant_mask: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor


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


def packed_tensors_from_tokenized_results(
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
) -> PackedTensors:
    token_ids: list[list[int]] = [[]]
    group_ids: list[list[int]] = [[]]
    input_pos: list[list[int]] = [[]]
    assistant_mask: list[list[int]] = [[]]
    logprobs: list[list[float]] = [[]]
    advantages: list[list[float]] = [[]]

    for result in tokenized_results:
        if len(result.token_ids) > seq_len:
            print("Result is too long, skipping")
            continue
        if len(token_ids[-1]) + len(result.token_ids) > seq_len:
            token_ids.append([])
            group_ids.append([])
            input_pos.append([])
            assistant_mask.append([])
            logprobs.append([])
            advantages.append([])
            group_id = 0
        else:
            group_id = max(group_ids[-1], default=-1) + 1
        token_ids[-1].extend(result.token_ids)
        group_ids[-1].extend([group_id] * len(result.token_ids))
        input_pos[-1].extend(range(len(result.token_ids)))
        assistant_mask[-1].extend(result.assistant_mask)
        offset = len(logprobs[-1])
        logprobs[-1].extend([float("nan")] * len(result.token_ids))
        if result.token_logprobs:
            assistant_indices = [
                i for i, mask in enumerate(result.assistant_mask) if mask
            ]
            assert len(assistant_indices) == len(result.token_logprobs)
            for idx, token_logprob in zip(assistant_indices, result.token_logprobs):
                logprobs[-1][idx + offset] = token_logprob.logprob
        advantages[-1].extend([result.advantage] * len(result.token_ids))

    def pad(values: list[list], pad_value) -> list[list]:
        max_len = seq_len
        for value in values:
            value.extend([pad_value] * (max_len - len(value)))
        return values

    return {
        "tokens": torch.tensor(pad(token_ids, pad_token_id)),
        "group_ids": torch.tensor(pad(group_ids, -1)),
        "input_pos": torch.tensor(pad(input_pos, 0)),
        "assistant_mask": torch.tensor(pad(assistant_mask, 0), dtype=torch.bool),
        "logprobs": torch.tensor(pad(logprobs, float("nan"))),
        "advantages": torch.tensor(pad(advantages, 0.0)),
    }


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
            "group_ids": torch.long,
            "input_pos": torch.long,
            "assistant_mask": torch.bool,
            "logprobs": torch.float32,
            "advantages": torch.float32,
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


def plot_packed_tensors(packed_tensors: PackedTensors) -> None:
    plt.figure(figsize=(15, 15))

    for tensor, label, title, subplot_idx in (
        (packed_tensors["tokens"], "Token IDs", "Token IDs", 1),
        (packed_tensors["logprobs"], "Log Probabilities", "Token Log Probs", 2),
        (packed_tensors["group_ids"], "Group IDs", "Token Groups", 3),
        (packed_tensors["input_pos"], "Position", "Input Position", 4),
        (packed_tensors["assistant_mask"], "Assistant Mask", "Assistant Mask", 5),
        (packed_tensors["advantages"], "Advantages", "Token Advantages", 6),
    ):
        plt.subplot(3, 2, subplot_idx)
        sns.heatmap(
            tensor.numpy(), cmap="viridis", cbar_kws={"label": label}, xticklabels=False
        )
        plt.title(title)
        plt.xlabel("Sequence Position")
        plt.ylabel("Batch")

    plt.tight_layout()
    plt.show()
