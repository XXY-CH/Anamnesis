"""Learned token importance gate — replaces MARK_THOUGHT oracle annotation.

Trains via distillation from oracle: during training the gate learns to predict
which positions the oracle would select, while the TCB continues using the
oracle mask directly. At inference the gate replaces the oracle.

Uses BCE-with-logits and per-sample pos_weight to handle the extreme class
imbalance (only ~1% of positions are oracle-selected).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedTokenGate(nn.Module):
    """Learn to identify which tokens are important for later retrieval.

    Trained with BCE against the oracle mask (positions before milestones).
    At inference, uses hard top-K from learned scores.
    """

    def __init__(
        self,
        d_model: int,
        max_store: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_store = max_store
        self.score_proj = nn.Linear(d_model, 1, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute importance scores and selection mask.

        Args:
            hidden: [B, T, d] model hidden states.

        Returns:
            mask: [B, T] bool mask (inference) or None (training).
            scores: [B, T] raw importance scores in (0, 1).
        """
        logits = self.score_proj(hidden).squeeze(-1)
        scores = torch.sigmoid(logits)

        if not self.training:
            _, indices = scores.topk(min(self.max_store, hidden.shape[1]), dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            return mask, scores

        return None, scores

    def compute_loss(
        self,
        hidden: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted BCE loss with automatic pos_weight for class balance.

        Args:
            hidden: [B, T, d] model hidden states (to compute fresh logits).
            target_mask: [B, T] boolean oracle mask (positive = important).
        """
        logits = self.score_proj(hidden).squeeze(-1)
        n_pos = target_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        n_neg = target_mask.shape[-1] - n_pos
        pos_weight = (n_neg / n_pos).clamp(max=200.0)
        return F.binary_cross_entropy_with_logits(
            logits, target_mask.float(), pos_weight=pos_weight,
        )
