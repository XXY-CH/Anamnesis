"""Kimi-style depth-axis Block Attention Residuals.

This module is intentionally not a token-time KV cache. It attends over a small
set of previous layer/block summaries and returns a scaled residual branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlockAttentionResidual(nn.Module):
    """Depth-axis attention over previous block/layer outputs.

    Args:
        d_model: Hidden width.
        max_sources: Maximum number of depth summaries retained.
        init_scale: Initial residual scale. Kept tiny for composition safety.
        distance_penalty: Optional ALiBi-style source-age penalty. This is a
            project extension, not part of the original AttnRes claim.
    """

    def __init__(
        self,
        d_model: int,
        max_sources: int = 8,
        init_scale: float = 1e-4,
        distance_penalty: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_sources = max_sources
        self.distance_penalty = distance_penalty

        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=d_model**-0.5)

        self.key_norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(
        self,
        x: torch.Tensor,
        sources: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read previous depth summaries.

        Args:
            x: Current hidden state [batch, seq_len, d_model]. Only shape/device
                are used; the query is layer-learned.
            sources: Previous layer/block outputs, each [batch, seq_len, d_model].

        Returns:
            Scaled residual branch and attention weights [batch, seq_len, m], or
            zeros/None when no source is available.
        """
        if not sources:
            return torch.zeros_like(x), None

        active_sources = sources[-self.max_sources :]
        stacked = torch.stack(active_sources, dim=2)  # [b, s, m, d]
        keys = self.key_norm(stacked)

        query = self.query.to(dtype=x.dtype, device=x.device)
        scores = torch.einsum("d,bsmd->bsm", query, keys)

        if self.distance_penalty:
            # Oldest retained source receives the largest penalty.
            m = scores.shape[-1]
            age = torch.arange(m - 1, -1, -1, device=x.device, dtype=x.dtype)
            scores = scores - self.distance_penalty * age.view(1, 1, m)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        readout = torch.einsum("bsm,bsmd->bsd", weights, stacked)
        return self.residual_scale.abs() * readout, weights
