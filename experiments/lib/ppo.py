# https://github.com/bradhilton/q4-2024-atreides/blob/main/experiments/lib/rl/ppo.py

from dataclasses import dataclass, field, fields
import math
import torch
import torch.nn as nn
from typing import Iterable, Literal, Optional, Union


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


def gae(
    advantages: torch.Tensor,
    gamma: float,
    lam: float,
    num_vectorized_iterations: int,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE) advantages.

    Args:
        advantages (torch.Tensor): The advantages to compute GAE for.
        gamma (float): The discount factor.
        lam (float): The decay factor for GAE.
        num_vectorized_iterations (int): The number of iterations to perform to for vectorized GAE.

    Returns:
        torch.Tensor: The GAE advantages.
    """
    alpha = gamma * lam
    if num_vectorized_iterations == 0:
        # Standard GAE
        out = torch.zeros_like(advantages)
        running = 0.0
        for t in reversed(range(len(advantages))):
            out[t] = running = advantages[t] + alpha * running
        return out
    weights = torch.exp(
        torch.log(torch.tensor(alpha, dtype=advantages.dtype))
        * torch.arange(advantages.shape[0], dtype=advantages.dtype)
    ).flip(0)
    epsilon = 1e-7
    weights = weights[weights > epsilon]
    num_chunks = math.ceil(advantages.shape[0] / weights.shape[0])
    if num_chunks == 1:
        return weighted_advantages(advantages, weights)
    gae_advantages = torch.cat(
        [
            weighted_advantages(chunk, weights)
            for chunk in advantages.chunk(num_chunks, dim=0)
        ]
    )
    # Perform additional iterations with offset starting points to reduce boundary effects
    # between chunks. This helps smooth out the GAE advantages at chunk boundaries by
    # computing overlapping windows with different alignments.
    chunk_size = advantages.shape[0] // num_chunks
    for i in range(1, num_vectorized_iterations):
        offset = i * chunk_size // num_vectorized_iterations
        for chunk in advantages[offset:].chunk(num_chunks, dim=0)[:-1]:
            gae_advantages[
                offset : offset + chunk_size // num_vectorized_iterations
            ] = weighted_advantages(chunk, weights)[
                : chunk_size // num_vectorized_iterations
            ]
    return gae_advantages


def weighted_advantages(
    advantages: torch.Tensor, reversed_weights: torch.Tensor
) -> torch.Tensor:
    a = advantages.flip(0)
    w = reversed_weights[: a.shape[0]]
    return (torch.cumsum(a * w, dim=0) / torch.cumsum(w, dim=0)).flip(0)


tensor_field = lambda: field(default_factory=lambda: torch.tensor(0.0))


@dataclass
class PPOResult:
    policy_weight: torch.Tensor = tensor_field()
    unclipped_policy_weight: torch.Tensor = tensor_field()
    tanh_log_policy_weight: torch.Tensor = tensor_field()
    reinforce_weight: torch.Tensor = tensor_field()
    advantage_prediction_weight: torch.Tensor = tensor_field()
    advantage_weight: torch.Tensor = tensor_field()
    value_weight: torch.Tensor = tensor_field()
    convergence_weight: torch.Tensor = tensor_field()
    divergence_weight: torch.Tensor = tensor_field()
    entropy_weight: torch.Tensor = tensor_field()
    entropy_target_weight: torch.Tensor = tensor_field()
    kl_weight: torch.Tensor = tensor_field()
    self_kl_weight: torch.Tensor = tensor_field()
    peer_kl_weight: torch.Tensor = tensor_field()
    reverse_kl_weight: torch.Tensor = tensor_field()
    ce_weight: torch.Tensor = tensor_field()
    weighted_entropy_weight: torch.Tensor = tensor_field()
    weighted_kl_weight: torch.Tensor = tensor_field()
    weighted_reverse_kl_weight: torch.Tensor = tensor_field()
    weighted_ce_weight: torch.Tensor = tensor_field()
    policy_loss: torch.Tensor = tensor_field()
    unclipped_policy_loss: torch.Tensor = tensor_field()
    tanh_log_policy_loss: torch.Tensor = tensor_field()
    reinforce_loss: torch.Tensor = tensor_field()
    advantage_prediction_loss: torch.Tensor = tensor_field()
    advantage_loss: torch.Tensor = tensor_field()
    value_loss: torch.Tensor = tensor_field()
    convergence_loss: torch.Tensor = tensor_field()
    divergence_loss: torch.Tensor = tensor_field()
    entropy_bonus: torch.Tensor = tensor_field()
    entropy_target: torch.Tensor = tensor_field()
    kl_divergence: torch.Tensor = tensor_field()
    self_kl_divergence: torch.Tensor = tensor_field()
    peer_kl_divergence: torch.Tensor = tensor_field()
    reverse_kl_divergence: torch.Tensor = tensor_field()
    ce_loss: torch.Tensor = tensor_field()
    weighted_entropy_bonus: torch.Tensor = tensor_field()
    weighted_kl_divergence: torch.Tensor = tensor_field()
    weighted_reverse_kl_divergence: torch.Tensor = tensor_field()
    weighted_ce_loss: torch.Tensor = tensor_field()
    num_tokens: torch.Tensor = field(default_factory=lambda: torch.tensor(0))

    def named_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def per_token(self) -> "PPOResult":
        return PPOResult(
            **{name: tensor / self.num_tokens for name, tensor in self.named_tensors()}
        )

    def tensors(self) -> Iterable[torch.Tensor]:
        return (tensor for _, tensor in self.named_tensors())

    def to(self, target: Union[torch.device, torch.dtype]) -> "PPOResult":
        return PPOResult(
            **{name: tensor.to(target) for name, tensor in self.named_tensors()}
        )

    def __iadd__(self, other: "PPOResult") -> "PPOResult":
        for tensor, other_tensor in zip(self.tensors(), other.tensors()):
            tensor += other_tensor.to(tensor.device)
        return self

    @property
    def entropy_target_loss(self) -> torch.Tensor:
        return torch.abs(self.entropy_bonus - self.entropy_target)

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            (self.policy_weight / self.num_tokens) * self.policy_loss
            + (self.unclipped_policy_weight / self.num_tokens)
            * self.unclipped_policy_loss
            + (self.tanh_log_policy_weight / self.num_tokens)
            * self.tanh_log_policy_loss
            + (self.reinforce_weight / self.num_tokens) * self.reinforce_loss
            + (self.advantage_prediction_weight / self.num_tokens)
            * self.advantage_prediction_loss
            + (self.value_weight / self.num_tokens) * self.value_loss
            + (self.convergence_weight / self.num_tokens) * self.convergence_loss
            - (self.divergence_weight / self.num_tokens) * self.divergence_loss
            - (self.entropy_weight / self.num_tokens) * self.entropy_bonus
            + (self.entropy_target_weight / self.num_tokens) * self.entropy_target_loss
            + (self.kl_weight / self.num_tokens) * self.kl_divergence
            + (self.self_kl_weight / self.num_tokens) * self.self_kl_divergence
            + (self.peer_kl_weight / self.num_tokens) * self.peer_kl_divergence
            + (self.reverse_kl_weight / self.num_tokens) * self.reverse_kl_divergence
            + (self.ce_weight / self.num_tokens) * self.ce_loss
            - (self.weighted_entropy_weight / self.num_tokens)
            * self.weighted_entropy_bonus
            + (self.weighted_kl_weight / self.num_tokens) * self.weighted_kl_divergence
            + (self.weighted_reverse_kl_weight / self.num_tokens)
            * self.weighted_reverse_kl_divergence
            + (self.weighted_ce_weight / self.num_tokens) * self.weighted_ce_loss
        )


class PPOLoss(nn.Module):
    def __init__(
        self,
        policy_coef: float = 1.0,
        unclipped_policy_coef: float = 0.0,
        tanh_log_policy_coef: float = 0.0,
        tanh_log_policy_beta: float = 1.0,
        tanh_log_advantages_first: bool = False,
        reinforce_coef: float = 0.0,
        clip_epsilon: float = 0.2,
        exploitation_penalty: float = 0.0,
        use_reference_logprobs: bool = False,
        advantage_prediction_coef: float = 1.0,
        advantage_prediction_error_type: Literal["squared", "absolute"] = "squared",
        advantage_prediction_quantile: Optional[float] = None,
        advantage_prediction_quantile_weight: float = 1.0,
        gae_gamma: float = 1.0,
        gae_lam: float = 0.95,
        gae_num_vectorized_iterations: int = 0,
        predicted_advantage_weight: float = 0.5,
        advantage_coef: float = 0.0,
        advantage_ratio: bool = False,
        advantage_regularization: float = 1.0,
        advantage_quantile: Optional[float] = None,
        advantage_quantile_weight: float = 1.0,
        value_coef: float = 0.0,
        value_quantile: Optional[float] = None,
        model_id: int = 0,
        convergence_coef: float = 0.0,
        divergence_coef: float = 0.0,
        entropy_coef: float = 0.01,
        entropy_target: float = 0.5,
        entropy_target_coef: float = 0.0,
        kl_coef: float = 0.0,
        self_kl_coef: float = 0.0,
        peer_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        ce_coef: float = 0.0,
        weighted_entropy_coef: float = 0.0,
        weighted_kl_coef: float = 0.0,
        weighted_reverse_kl_coef: float = 0.0,
        weighted_ce_coef: float = 0.0,
        normalize_advantages: bool = True,
        normalize_advantage_predictions: bool = True,
        normalize_values: bool = True,
        normalize_value_predictions: bool = True,
    ):
        """
        Initializes the PPO Loss module.

        Args:
            policy_coef (float): Coefficient for the clipped policy loss. Defaults to 1.0.
            unclipped_policy_coef (float): Coefficient for the unclipped policy loss. Defaults to 0.0.
            tanh_log_policy_coef (float): Coefficient for the tanh log policy loss. Defaults to 0.0.
            tanh_log_policy_beta (float): Beta for the tanh log policy loss. Defaults to 1.0.
            tanh_log_advantages_first (bool): If True, the log ratio is multiplied by advantages before tanh/log. Defaults to False.
            reinforce_coef (float): Coefficient for the REINFORCE loss. Defaults to 0.0.s
            clip_epsilon (float): Clipping parameter for PPO (typically between 0.1 and 0.3).
            exploitation_penalty (float): Reduces the impact of positive advantages by
                multiplying them by (1 - exploitation_penalty). This helps prevent
                premature convergence and encourages exploration. Defaults to 0.0.
            use_reference_logprobs (bool): If True, uses reference_logprobs instead of
                logprobs for policy losses. Defaults to False.
            advantage_prediction_coef (float): Coefficient for the advantage prediction loss. Defaults to 1.0.
            advantage_prediction_error_type (Literal["squared", "absolute"]): The type of error to use for the advantage prediction loss. Defaults to "squared".
            advantage_prediction_quantile (Optional[float]): Optional quantile to use for (quantile) advantage prediction loss. Defaults to None.
            advantage_prediction_quantile_weight (float): Weight for the quantile advantage prediction loss if advantage_prediction_quantile is not None. Defaults to 1.0.
            gae_gamma (float): The discount factor for GAE. Defaults to 1.0.
            gae_lam (float): The decay factor for GAE. Defaults to 0.95.
            gae_num_vectorized_iterations (int): The number of iterations to perform to for vectorized GAE. Defaults to 0 (unvectorized GAE).
            predicted_advantage_weight (float): How much to weight predicted advantages vs. estimated advantages. Defaults to 0.5.
            advantage_coef (float): Coefficient for the advantage loss. Defaults to 0.0.
            advantage_ratio (bool): If True, uses the ratio of the new and old logprobs for the advantage loss. Defaults to False.
            advantage_regularization (float): Regularization for the advantage loss. Defaults to 1.0.
            advantage_quantile (Optional[float]): Optional quantile to use for (quantile) advantage loss. Defaults to None.
            advantage_quantile_weight (float): Weight for the quantile advantage loss if advantage_quantile is not None. Defaults to 1.0.
            value_coef (float): Coefficient for the value loss (defaults to 0.0).
            value_quantile (Optional[float]): Optional quantile to use for (quantile) value loss. Defaults to None.
            model_id (int): The ID of the model to use for convergence/divergence losses. Defaults to 0.
            convergence_coef (float): Coefficient for the convergence loss. Defaults to 0.0.
            divergence_coef (float): Coefficient for the divergence loss. Defaults to 0.0.
            entropy_coef (float): Coefficient for the entropy bonus to encourage exploration.
            entropy_target (float): Target entropy (defaults to 0.5).
            entropy_target_coef (float): Coefficient for the entropy target loss.
            kl_coef (float): Coefficient for KL divergence penalty (defaults to 0.0).
            self_kl_coef (float): Coefficient for self KL divergence penalty (defaults to 0.0).
            peer_kl_coef (float): Coefficient for peer KL divergence penalty (defaults to 0.0).
            reverse_kl_coef (float): Coefficient for reverse KL divergence penalty (defaults to 0.0).
            ce_coef (float): Coefficient for cross entropy penalty (defaults to 0.0).
            weighted_entropy_coef (float): Coefficient for the weighted entropy bonus.
            weighted_kl_coef (float): Coefficient for the weighted KL divergence penalty.
            weighted_reverse_kl_coef (float): Coefficient for the weighted reverse KL divergence penalty.
            weighted_ce_coef (float): Coefficient for the weighted cross entropy loss.
            normalize_advantages (bool): Whether to normalize advantages before computing the loss (default True).
            normalize_advantage_predictions (bool): Whether to normalize advantage predictions before computing the loss (default True).
            normalize_values (bool): Whether to normalize values before computing the loss (default True).
            normalize_value_predictions (bool): Whether to normalize value predictions before computing the loss (default True).
        """
        super().__init__()
        self.policy_coef = policy_coef
        self.unclipped_policy_coef = unclipped_policy_coef
        self.tanh_log_policy_coef = tanh_log_policy_coef
        self.tanh_log_policy_beta = tanh_log_policy_beta
        self.tanh_log_advantages_first = tanh_log_advantages_first
        self.reinforce_coef = reinforce_coef
        self.clip_epsilon = clip_epsilon
        self.exploitation_penalty = exploitation_penalty
        self.use_reference_logprobs = use_reference_logprobs
        self.advantage_prediction_coef = advantage_prediction_coef
        self.advantage_prediction_error_type = advantage_prediction_error_type
        self.advantage_prediction_quantile = advantage_prediction_quantile
        self.advantage_prediction_quantile_weight = advantage_prediction_quantile_weight
        self.gae_gamma = gae_gamma
        self.gae_lam = gae_lam
        self.gae_num_vectorized_iterations = gae_num_vectorized_iterations
        self.predicted_advantage_weight = predicted_advantage_weight
        self.advantage_coef = advantage_coef
        self.advantage_ratio = advantage_ratio
        self.advantage_regularization = advantage_regularization
        self.advantage_quantile = advantage_quantile
        self.advantage_quantile_weight = advantage_quantile_weight
        self.value_coef = value_coef
        self.value_quantile = value_quantile
        self.model_id = model_id
        self.convergence_coef = convergence_coef
        self.divergence_coef = divergence_coef
        self.entropy_coef = entropy_coef
        self.entropy_target = entropy_target
        self.entropy_target_coef = entropy_target_coef
        self.kl_coef = kl_coef
        self.self_kl_coef = self_kl_coef
        self.peer_kl_coef = peer_kl_coef
        self.reverse_kl_coef = reverse_kl_coef
        self.ce_coef = ce_coef
        self.weighted_entropy_coef = weighted_entropy_coef
        self.weighted_kl_coef = weighted_kl_coef
        self.weighted_reverse_kl_coef = weighted_reverse_kl_coef
        self.weighted_ce_coef = weighted_ce_coef
        self.normalize_advantages = normalize_advantages
        self.normalize_advantage_predictions = normalize_advantage_predictions
        self.normalize_values = normalize_values
        self.normalize_value_predictions = normalize_value_predictions

    def forward(
        self,
        *,
        logits: Union[torch.Tensor, list[torch.Tensor]],
        value_predictions: torch.Tensor,
        tokens: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
        weights: torch.Tensor,
        model_ids: torch.Tensor,
        bos_id: int,
    ) -> PPOResult:
        """
        Computes the PPO loss for sequence data, supporting both regular and chunked inputs.

        Args:
            logits (Union[Tensor, List[Tensor]]):
                Either a single tensor of shape (batch_size, sequence_length, vocab_size)
                or a list of chunked tensors, each of shape
                (batch_size, sequence_length/num_chunks, vocab_size).

            value_predictions (Tensor):
                Shape: (batch_size, sequence_length)
                Value predictions for each token.

            tokens (Tensor):
                Shape: (batch_size, sequence_length)
                Token indices sampled under the old policy.

            values (Tensor):
                Shape: (batch_size, sequence_length)
                Value estimates for each token.

            advantages (Tensor):
                Shape: (batch_size, sequence_length)
                Advantage estimates for each token.

            logprobs (Tensor):
                Shape: (batch_size, sequence_length)
                Log probabilities of the sampled tokens under the old policy.

            reference_logprobs (Tensor):
                Shape: (batch_size, sequence_length)
                Log probabilities of the sampled tokens under the reference policy.

            weights (Tensor):
                Shape: (batch_size, sequence_length)
                Weights for each token in the sequence.

            model_ids (Tensor):
                Shape: (batch_size, sequence_length)
                The ID of the model for each token in the sequence.

            bos_id (int):
                Index of the beginning of sequence token in the vocabulary.

        Returns:
            PPOResult: The combined loss results across all chunks.
        """
        if isinstance(logits, list):
            result = PPOResult().to(logits[0].device)
            num_chunks = len(logits)
            for chunked_args in zip(
                logits,
                value_predictions.chunk(num_chunks, dim=1),
                tokens.chunk(num_chunks, dim=1),
                values.chunk(num_chunks, dim=1),
                advantages.chunk(num_chunks, dim=1),
                logprobs.chunk(num_chunks, dim=1),
                reference_logprobs.chunk(num_chunks, dim=1),
                weights.chunk(num_chunks, dim=1),
                model_ids.chunk(num_chunks, dim=1),
            ):
                result += self._forward_chunk(*chunked_args, bos_id=bos_id)
            return result

        return self._forward_chunk(
            logits,
            value_predictions,
            tokens,
            values,
            advantages,
            logprobs,
            reference_logprobs,
            weights,
            model_ids,
            bos_id=bos_id,
        )

    def _forward_chunk(
        self,
        logits: torch.Tensor,
        value_predictions: torch.Tensor,
        tokens: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
        weights: torch.Tensor,
        model_ids: torch.Tensor,
        bos_id: int,
    ) -> PPOResult:
        """
        Processes a single chunk of the PPO loss computation.
        """
        # Flatten logits tensor to shape (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        if self.normalize_value_predictions:
            value_predictions = (value_predictions - value_predictions.mean()) / (
                value_predictions.std() + 1e-8
            )
        advantage_predictions = (
            shift_tensor(value_predictions, ignore_label=bos_id) - value_predictions
        ).view(-1)
        # Shape: (batch_size * sequence_length,)
        tokens = shift_tensor(tokens, ignore_label=bos_id).view(-1)
        values = shift_tensor(values).view(-1)
        advantages = shift_tensor(advantages).view(-1)
        logprobs = shift_tensor(logprobs).view(-1)
        reference_logprobs = shift_tensor(reference_logprobs).view(-1)
        weights = shift_tensor(weights).view(-1)
        model_ids = shift_tensor(model_ids).view(-1)
        # Create a Categorical distribution from logits
        dist = torch.distributions.Categorical(logits=logits)

        # Calculate new log probabilities of the taken actions
        new_logprobs = dist.log_prob(tokens)  # Shape: (batch_size * sequence_length,)

        # Debugging
        if False:
            # Shape: (batch_size * sequence_length,)
            kl_divergence = torch.nn.functional.kl_div(
                new_logprobs, logprobs, reduction="none", log_target=True
            )
            # Debugging KL divergence distribution
            import matplotlib.pyplot as plt
            import numpy as np

            # Convert to numpy and remove NaN values
            kl_np = kl_divergence.detach().cpu().numpy()

            # Create line plot
            plt.figure(figsize=(10, 6))
            plt.plot(kl_np, linewidth=2)
            plt.xlabel("Token Index")
            plt.ylabel("KL Divergence")
            plt.title("KL Divergence Across Tokens")
            plt.grid(True, alpha=0.3)
            plt.show()

            # Find first token with KL divergence > 10
            high_kl_idx = (kl_np > 10).nonzero()[0]
            if len(high_kl_idx) > 0:
                print(f"First token with KL divergence > 10: {high_kl_idx[0]}")
                print(f"Token: {tokens[high_kl_idx[0]]}")
                print(f"Logprob: {logprobs[high_kl_idx[0]]}")
                print(f"New Logprob: {new_logprobs[high_kl_idx[0]]}")
            else:
                print("No tokens found with KL divergence > 1")

            raise ValueError("KL Divergence Debugging")

        # Calculate entropy for each token
        entropy = dist.entropy()

        # Create mask where advantages and logprobs are not NaN
        mask = ~torch.isnan(advantages) & ~torch.isnan(logprobs)
        num_tokens = mask.sum()
        if num_tokens == 0:
            logprobs = torch.zeros_like(logprobs)
            mask = ~torch.isnan(advantages)
            num_tokens = mask.sum()
        if num_tokens == 0:
            return PPOResult()

        # Apply mask to all tensors
        # Shape: (num_tokens,)
        new_logprobs = new_logprobs[mask]
        logprobs = logprobs[mask]
        advantage_predictions = advantage_predictions[mask]
        values = values[mask]
        advantages = advantages[mask]
        entropy = entropy[mask]
        reference_logprobs = reference_logprobs[mask]
        weights = weights[mask]
        model_ids = model_ids[mask]

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.exploitation_penalty > 0.0:
            # Reduce positive advantages to discourage greedy exploitation
            advantages = torch.where(
                advantages > 0, advantages * (1 - self.exploitation_penalty), advantages
            )

        old_logprobs = reference_logprobs if self.use_reference_logprobs else logprobs
        # Calculate the probability ratio (π_θ(a|s) / π_θ_old(a|s))
        log_ratio = new_logprobs - old_logprobs  # Shape: (num_tokens,)
        ratio = torch.exp(log_ratio)  # Shape: (num_tokens,)

        # Generalized Advantage Estimation
        gae_advantages = gae(
            advantages, self.gae_gamma, self.gae_lam, self.gae_num_vectorized_iterations
        )
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (
            gae_advantages.std() + 1e-8
        )

        advantage_estimates = advantages
        advantages = (
            self.predicted_advantage_weight * gae_advantages
            + (1 - self.predicted_advantage_weight) * advantage_estimates
        )

        # Calculate the surrogate losses
        surrogate1 = ratio * advantages  # Shape: (num_tokens,)
        surrogate2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )  # Shape: (num_tokens,)

        # Take the minimum of the two surrogate losses for clipped version
        policy_loss = -torch.min(surrogate1, surrogate2).mul(weights).sum()  # Scalar

        # Calculate unclipped policy loss
        unclipped_policy_loss = -surrogate1.mul(weights).sum()  # Scalar

        # Calculate tanh log policy loss
        if self.tanh_log_advantages_first:
            tanh_log_policy_loss = (
                -torch.tanh(self.tanh_log_policy_beta * log_ratio.mul(advantages))
                .mul(weights)
                .sum()
            )  # Scalar
        else:
            tanh_log_policy_loss = (
                -torch.tanh(self.tanh_log_policy_beta * log_ratio)
                .mul(advantages)
                .mul(weights)
                .sum()
            )  # Scalar

        # Calculate REINFORCE loss
        reinforce_loss = -(new_logprobs * advantages).mul(weights).sum()  # Scalar

        # Calculate advantage prediction loss between predicted and estimated advantages
        advantage_prediction_diff = advantage_predictions - advantage_estimates
        advantage_prediction_error = (
            advantage_prediction_diff.pow(2)
            if self.advantage_prediction_error_type == "squared"
            else advantage_prediction_diff.abs()
        )
        if self.advantage_prediction_quantile:
            advantage_prediction_quantile_error = (
                torch.where(
                    advantage_prediction_diff > 0,
                    self.advantage_prediction_quantile * advantage_prediction_error,
                    (1 - self.advantage_prediction_quantile)
                    * advantage_prediction_error,
                )
                * 2
            )
            advantage_prediction_error = (
                (1 - self.advantage_prediction_quantile_weight)
                * advantage_prediction_error
                + self.advantage_prediction_quantile_weight
                * advantage_prediction_quantile_error
            )
        advantage_prediction_loss = advantage_prediction_error.mul(
            weights
        ).sum()  # Scalar

        if self.advantage_coef:
            # Calculate the advantage loss
            # Shape: (num_tokens,)
            advantage_preds = new_logprobs
            if self.advantage_ratio:
                advantage_preds = advantage_preds - old_logprobs
            if self.normalize_advantage_predictions:
                advantage_preds = (advantage_preds - advantage_preds.mean()) / (
                    advantage_preds.std() + 1e-8
                )
            diff = advantages - self.advantage_regularization * advantage_preds
            advantage_loss = diff.abs()
            if self.advantage_quantile:
                quantile_loss = torch.where(
                    diff > 0,
                    self.advantage_quantile * diff,
                    (1 - self.advantage_quantile) * -diff,
                )
                advantage_loss = (
                    1 - self.advantage_quantile_weight
                ) * advantage_loss + self.advantage_quantile_weight * quantile_loss
            advantage_loss = advantage_loss.mul(weights).sum()
        else:
            advantage_loss = torch.tensor(0.0, device=logits.device)

        if self.value_coef:
            # Calculate the value loss
            # Shape: (num_tokens,)
            value_preds = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)[mask]
            if self.normalize_values:
                values = (values - values.mean()) / (values.std() + 1e-8)
            if self.normalize_value_predictions:
                value_preds = (value_preds - value_preds.mean()) / (
                    value_preds.std() + 1e-8
                )
            if self.value_quantile:
                diff = values - value_preds
                quantile_loss = torch.where(
                    diff > 0,
                    self.value_quantile * diff,
                    (1 - self.value_quantile) * -diff,
                )
                value_loss = quantile_loss.mul(weights).sum()
            else:
                value_loss = (value_preds - values).pow(2).mul(weights).sum()
        else:
            value_loss = torch.tensor(0.0, device=logits.device)

        # Convergence/Divergence Losses
        self_mask = model_ids == self.model_id
        peer_mask = model_ids != self.model_id
        convergence_loss = -new_logprobs.mul(self_mask).mul(weights).sum()
        divergence_loss = new_logprobs.mul(peer_mask).mul(weights).sum()

        # Entropy bonus
        entropy_bonus = entropy.mul(weights)
        weighted_entropy_bonus = entropy_bonus.mul(-advantages).sum()  # Scalar
        entropy_bonus = entropy_bonus.sum()  # Scalar

        # Calculate KL divergence between the old and new policies
        kl_divergence = torch.nn.functional.kl_div(
            new_logprobs,
            reference_logprobs,
            reduction="none",
            log_target=True,
        ).mul(
            weights
        )  # Shape: (num_tokens,)

        weighted_kl_divergence = kl_divergence.mul(-advantages).sum()  # Scalar
        kl_divergence = kl_divergence.sum()  # Scalar

        sample_kl_div = torch.nn.functional.kl_div(
            new_logprobs,
            logprobs,
            reduction="none",
            log_target=True,
        ).mul(weights)

        if self_mask.sum() > 0:
            self_kl_divergence = sample_kl_div[self_mask].sum()
        else:
            self_kl_divergence = torch.tensor(0.0, device=logits.device)

        if peer_mask.sum() > 0:
            peer_kl_divergence = sample_kl_div[peer_mask].sum()
        else:
            peer_kl_divergence = torch.tensor(0.0, device=logits.device)

        # Calculate reverse KL divergence between the old and new policies
        reverse_kl_divergence = torch.nn.functional.kl_div(
            reference_logprobs,
            new_logprobs,
            reduction="none",
            log_target=True,
        ).mul(
            weights
        )  # Shape: (num_tokens,)

        ce_loss = -(
            (
                values
                * torch.log(
                    torch.sigmoid(
                        torch.clamp(
                            self.tanh_log_policy_beta
                            * (new_logprobs - reference_logprobs),
                            min=-20.0,
                            max=20.0,
                        )
                    )
                    + 1e-10
                )
                + (1 - values)
                * torch.log(
                    1
                    - torch.sigmoid(
                        torch.clamp(
                            self.tanh_log_policy_beta
                            * (new_logprobs - reference_logprobs),
                            min=-20.0,
                            max=20.0,
                        )
                    )
                    + 1e-10
                )
            )
            .mul(weights)
            .sum()
        )

        weighted_reverse_kl_divergence = reverse_kl_divergence.mul(
            -advantages
        ).sum()  # Scalar
        reverse_kl_divergence = reverse_kl_divergence.sum()  # Scalar

        # Calculate cross entropy loss using log probabilities, weighted by advantages
        weighted_ce_loss = -new_logprobs.mul(advantages).mul(weights).sum()  # Scalar

        return PPOResult(
            policy_weight=self.policy_coef * num_tokens,
            unclipped_policy_weight=self.unclipped_policy_coef * num_tokens,
            tanh_log_policy_weight=self.tanh_log_policy_coef * num_tokens,
            reinforce_weight=self.reinforce_coef * num_tokens,
            advantage_prediction_weight=self.advantage_prediction_coef * num_tokens,
            advantage_weight=self.advantage_coef * num_tokens,
            value_weight=self.value_coef * num_tokens,
            convergence_weight=self.convergence_coef * num_tokens,
            divergence_weight=self.divergence_coef * num_tokens,
            entropy_weight=self.entropy_coef * num_tokens,
            entropy_target_weight=self.entropy_target_coef * num_tokens,
            kl_weight=self.kl_coef * num_tokens,
            self_kl_weight=self.self_kl_coef * num_tokens,
            peer_kl_weight=self.peer_kl_coef * num_tokens,
            reverse_kl_weight=self.reverse_kl_coef * num_tokens,
            ce_weight=self.ce_coef * num_tokens,
            weighted_entropy_weight=self.weighted_entropy_coef * num_tokens,
            weighted_kl_weight=self.weighted_kl_coef * num_tokens,
            weighted_reverse_kl_weight=self.weighted_reverse_kl_coef * num_tokens,
            weighted_ce_weight=self.weighted_ce_coef * num_tokens,
            policy_loss=policy_loss,
            unclipped_policy_loss=unclipped_policy_loss,
            tanh_log_policy_loss=tanh_log_policy_loss,
            reinforce_loss=reinforce_loss,
            advantage_prediction_loss=advantage_prediction_loss,
            advantage_loss=advantage_loss,
            value_loss=value_loss,
            convergence_loss=convergence_loss,
            divergence_loss=divergence_loss,
            entropy_bonus=entropy_bonus,
            entropy_target=self.entropy_target * num_tokens,
            kl_divergence=kl_divergence,
            self_kl_divergence=self_kl_divergence,
            peer_kl_divergence=peer_kl_divergence,
            reverse_kl_divergence=reverse_kl_divergence,
            ce_loss=ce_loss,
            weighted_entropy_bonus=weighted_entropy_bonus,
            weighted_kl_divergence=weighted_kl_divergence,
            weighted_reverse_kl_divergence=weighted_reverse_kl_divergence,
            weighted_ce_loss=weighted_ce_loss,
            num_tokens=num_tokens,
        )
