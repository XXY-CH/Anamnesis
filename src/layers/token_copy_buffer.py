"""Token Copy Buffer -- direct token-embedding storage for exact-copy paths.

Unlike MilestoneSnapshotReadout which stores compressed hidden states,
this module stores raw token embeddings at marked positions and reads
them out via attention, projecting directly to logits through the
token embedding weight matrix.

This bypasses RetNet's recurrent state compression for marked tokens. Exact
prediction still depends on attention alignment and logit-margin conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sinusoidal_encoding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """Functional sinusoidal encoding — works at any position index."""
    half = d_model // 2
    freq = torch.arange(half, device=positions.device, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (freq / half))
    angles = positions.float().unsqueeze(-1) * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TokenCopyBuffer(nn.Module):
    """Store and read out token embeddings for conditional exact-copy recall."""

    def __init__(
        self,
        d_model: int,
        max_snapshots: int = 8,
        max_seq_len: int = 2048,
        init_scale: float = 1e-4,
        dropout: float = 0.0,
        use_pos_keys: bool = True,
        use_sinusoidal_pos: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_snapshots = max_snapshots
        self.use_pos_keys = use_pos_keys
        self.use_sinusoidal_pos = use_sinusoidal_pos
        self.query_norm = nn.RMSNorm(d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        if use_pos_keys and not use_sinusoidal_pos:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self._max_pos = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def collect(
        self,
        token_embeddings: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Collect token embeddings and position IDs at source positions.

        Handles boolean masks (inference / oracle) and float soft masks
        (training with LearnedTokenGate). Float masks use differentiable
        weighted gather so gradients flow through the gate.
        """
        if self.max_snapshots <= 0:
            return None

        batch, seq_len, d_model = token_embeddings.shape
        device = token_embeddings.device

        if source_mask.is_floating_point():
            k = min(self.max_snapshots, seq_len)
            topk_weights, topk_indices = source_mask.topk(k, dim=-1)
            idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, d_model)
            gathered = torch.gather(token_embeddings, 1, idx_expanded)
            stored = topk_weights.unsqueeze(-1) * gathered
            pos_ids = topk_indices
            valid = torch.ones(batch, k, device=device, dtype=torch.bool)
            return stored, valid, pos_ids

        stored = token_embeddings.new_zeros(batch, self.max_snapshots, d_model)
        pos_ids = torch.zeros(batch, self.max_snapshots, device=device, dtype=torch.long)
        valid = torch.zeros(batch, self.max_snapshots, device=device, dtype=torch.bool)

        any_stored = False
        for batch_idx in range(batch):
            positions = torch.nonzero(source_mask[batch_idx], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            positions = positions[-self.max_snapshots:]
            count = positions.numel()
            stored[batch_idx, :count] = token_embeddings[batch_idx, positions]
            pos_ids[batch_idx, :count] = positions
            valid[batch_idx, :count] = True
            any_stored = True

        if not any_stored:
            return None
        return stored, valid, pos_ids

    def forward(
        self,
        hidden: torch.Tensor,
        buffer_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend over stored token embeddings with positional keys.

        Returns:
            Scaled residual (in embedding space) and attention weights.
        """
        if buffer_cache is None:
            return torch.zeros_like(hidden), None

        stored, valid, pos_ids = buffer_cache
        q = self.query_norm(hidden)
        k = self.key_proj(stored)
        if self.use_pos_keys:
            if self.use_sinusoidal_pos:
                k = k + _sinusoidal_encoding(pos_ids, self.d_model)
            else:
                clamped = pos_ids.clamp(max=self._max_pos - 1)
                k = k + self.pos_embedding(clamped)
        scores = torch.einsum("bsd,bmd->bsm", q, k) / (self.d_model**0.5)
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        readout = torch.einsum("bsm,bmd->bsd", weights, stored)
        return self.residual_scale.abs() * readout, weights
