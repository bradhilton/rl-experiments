from dataclasses import dataclass, field, fields
import torch
from typing import Iterable, Optional, Union

ignore_labels_cache: dict[
    tuple[torch.Size, Union[int, float], torch.dtype, torch.device], torch.Tensor
] = {}


def shift_tensor(
    labels: torch.Tensor, ignore_label: Optional[Union[int, float]] = None
) -> torch.Tensor:
    if ignore_label is None:
        ignore_label = (
            -100
            if labels.dtype in (torch.int32, torch.int64, torch.int16, torch.int8)
            else float("nan")
        )

    # Create a tensor of ignore labels every time if we are compiling, otherwise cache it
    if torch.compiler.is_compiling():
        ignore_labels = torch.full(
            (labels.shape[0], 1), ignore_label, dtype=labels.dtype, device=labels.device
        )
    else:
        key = (labels.shape[-1:], ignore_label, labels.dtype, labels.device)
        if key not in ignore_labels_cache:
            ignore_labels_cache[key] = torch.full(
                (labels.shape[0], 1),
                ignore_label,
                dtype=labels.dtype,
                device=labels.device,
            )
        ignore_labels = ignore_labels_cache[key]

    # Shift labels to compute loss
    return torch.cat((labels[..., 1:], ignore_labels), dim=1)


@dataclass
class GRPOResult:
    num_tokens: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    ce_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def named_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def per_token(self) -> "GRPOResult":
        return GRPOResult(
            **{name: tensor / self.num_tokens for name, tensor in self.named_tensors()}
        )

    def tensors(self) -> Iterable[torch.Tensor]:
        return (tensor for _, tensor in self.named_tensors())

    def to(self, target: Union[torch.device, torch.dtype]) -> "GRPOResult":
        return GRPOResult(
            **{name: tensor.to(target) for name, tensor in self.named_tensors()}
        )

    def __iadd__(self, other: "GRPOResult") -> "GRPOResult":
        for tensor, other_tensor in zip(self.tensors(), other.tensors()):
            tensor += other_tensor.to(tensor.device)
        return self

    @property
    def total_loss(self) -> torch.Tensor:
        return self.ce_loss


class GRPO(torch.nn.Module):
    def forward(
        self,
        logits: Union[torch.Tensor, list[torch.Tensor]],
        tokens: torch.Tensor,
        bos_id: int,
    ) -> GRPOResult:
        """
        Computes the GRPO loss for sequence data, supporting both regular and chunked inputs.

        Args:
            logits (Union[Tensor, List[Tensor]]):
                Either a single tensor of shape (batch_size, sequence_length, vocab_size)
                or a list of chunked tensors, each of shape
                (batch_size, sequence_length/num_chunks, vocab_size).
            tokens (Tensor):
                Shape: (batch_size, sequence_length)
                Token indices.
            bos_id (int):
                Index of the beginning of sequence token in the vocabulary.

        Returns:
            GRPOResult: The combined loss results across all chunks.
        """
        if isinstance(logits, list):
            result = GRPOResult().to(logits[0].device)
            num_chunks = len(logits)
            for chunked_args in zip(
                logits,
                tokens.chunk(num_chunks, dim=1),
            ):
                result += self._forward_chunk(*chunked_args, bos_id)
            return result

        return self._forward_chunk(logits, tokens, bos_id)

    def _forward_chunk(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        bos_id: int,
    ) -> GRPOResult:
        """
        Processes a single chunk of the GRPO loss computation.
        """
        # Flatten logits tensor to shape (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        tokens = shift_tensor(tokens, bos_id).view(
            -1
        )  # (batch_size * sequence_length,)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(tokens)
        ce_loss = -log_probs.sum()
        return GRPOResult(
            num_tokens=torch.tensor(tokens.numel(), device=tokens.device),
            ce_loss=ce_loss,
        )
