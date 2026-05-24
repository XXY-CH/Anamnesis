"""GCA-inspired chunk retriever for long context.

Scores chunks by query-dependent dot product attention with optional
chunk-level RoPE for breaking multi-needle symmetry.

Reference: Grouped Cross Attention (GCA) achieves 1000x length generalization
using a learned chunk retriever trained end-to-end with the generation loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Apply rotary position embedding."""
    half = dim // 2
    freq = 1.0 / (10000.0 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angles = positions.float().unsqueeze(-1) * freq
    cos = angles.cos()
    sin = angles.sin()
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ChunkRetriever(nn.Module):
    """Learn to select relevant chunks from a long context.

    Given chunk embeddings and a query embedding, computes soft attention
    over chunks and produces a logit correction for the answer positions.
    """

    def __init__(
        self,
        d_model: int,
        use_chunk_rope: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_chunk_rope = use_chunk_rope
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.chunk_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def score_chunks(
        self,
        query_emb: torch.Tensor,
        chunk_embs: torch.Tensor,
        query_chunk_idx: int | None = None,
    ) -> torch.Tensor:
        """Score chunks by content similarity with optional RoPE.

        Args:
            query_emb: [batch, d]
            chunk_embs: [batch, num_chunks, d]
            query_chunk_idx: index of the chunk containing the query.

        Returns:
            scores: [batch, num_chunks]
        """
        q = self.query_proj(query_emb)
        c = self.chunk_proj(chunk_embs)

        if self.use_chunk_rope and query_chunk_idx is not None:
            num_chunks = c.shape[1]
            device = c.device
            chunk_pos = torch.arange(num_chunks, device=device, dtype=torch.long)
            query_pos = torch.full(
                (q.shape[0],), query_chunk_idx, device=device, dtype=torch.long,
            )
            q = _apply_rope(q, query_pos, self.d_model)
            c = _apply_rope(
                c, chunk_pos.unsqueeze(0).expand(c.shape[0], -1), self.d_model,
            )

        return torch.einsum("bd,bnd->bn", q, c) / (self.d_model ** 0.5)

    def forward(
        self,
        query_emb: torch.Tensor,
        chunk_embs: torch.Tensor,
        chunk_hiddens: torch.Tensor,
        token_emb_weight: torch.Tensor,
        query_chunk_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft attention over chunks and produce logit correction."""
        scores = self.score_chunks(query_emb, chunk_embs, query_chunk_idx)
        weights = torch.softmax(scores, dim=-1)

        v = self.value_proj(chunk_hiddens)
        summary = torch.einsum("bn,bnd->bd", weights, v)

        logit_correction = F.linear(summary, token_emb_weight)
        return self.logit_scale * logit_correction, weights


def compute_chunk_embeddings(
    model: nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Process a long sequence in chunks and return per-chunk embeddings.

    Args:
        model: RetNetEngramModel (frozen).
        input_ids: [batch=1, seq_len] token IDs.
        chunk_size: size of each chunk.

    Returns:
        chunk_embs: [batch, num_chunks, d] mean-pooled hidden per chunk.
        chunk_hiddens_list: list of [batch, chunk_len, d] raw hiddens per chunk.
    """
    seq_len = input_ids.shape[1]
    chunk_embs_list: list[torch.Tensor] = []
    chunk_hiddens_list: list[torch.Tensor] = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]
        hidden = model(chunk_ids, return_hidden_only=True)  # [1, C, d]
        chunk_embs_list.append(hidden.mean(dim=1))  # [1, d]
        chunk_hiddens_list.append(hidden)

    chunk_embs = torch.stack(chunk_embs_list, dim=1)  # [1, num_chunks, d]
    return chunk_embs, chunk_hiddens_list
