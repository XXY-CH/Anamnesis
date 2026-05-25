"""Tiny synthetic language-model training utilities.

The goal is not benchmark quality. This file gives the architecture a minimal
repeatable training surface before real datasets are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models import AnamnesisModel


@dataclass
class ToyBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor


def make_toy_lm_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str = "cpu",
    milestone_token_id: int | None = None,
) -> ToyBatch:
    """Create a deterministic-shape next-token prediction batch."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    if milestone_token_id is not None and seq_len >= 4:
        tokens[:, seq_len // 2] = milestone_token_id
    return ToyBatch(input_ids=tokens[:, :-1], target_ids=tokens[:, 1:])


def language_modeling_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy next-token loss."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
    )


def train_step(
    model: AnamnesisModel,
    optimizer: torch.optim.Optimizer,
    batch: ToyBatch,
    grad_clip: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one supervised next-token training step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, metrics = model(batch.input_ids, return_metrics=True)
    loss = language_modeling_loss(logits, batch.target_ids)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    scalar_metrics = {
        key: float(value.detach().cpu())
        for key, value in metrics.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }
    scalar_metrics["loss"] = float(loss.detach().cpu())
    return loss.detach(), scalar_metrics
