"""GCA-inspired chunk retriever for long context.

Scores chunks by query-dependent dot product attention, selects relevant
context, and produces a logit correction. Fully differentiable via soft
attention (no STE needed).

Reference: Grouped Cross Attention (GCA) achieves 1000x length generalization
using a learned chunk retriever trained end-to-end with the generation loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkRetriever(nn.Module):
    """Learn to select relevant chunks from a long context.

    Given chunk embeddings and a query embedding, computes soft attention
    over chunks and produces a logit correction for the answer positions.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.chunk_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        query_emb: torch.Tensor,
        chunk_embs: torch.Tensor,
        chunk_hiddens: torch.Tensor,
        token_emb_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft attention over chunks and produce logit correction.

        Args:
            query_emb: [batch, d] mean-pooled hidden from query region.
            chunk_embs: [batch, num_chunks, d] mean-pooled hidden per chunk.
            chunk_hiddens: [batch, num_chunks, d] raw chunk summaries for value.
            token_emb_weight: [vocab, d] for projecting to vocab space.

        Returns:
            logit_correction: [batch, vocab] additive correction to logits.
            weights: [batch, num_chunks] attention weights for analysis.
        """
        q = self.query_proj(query_emb)
        c = self.chunk_proj(chunk_embs)

        scores = torch.einsum("bd,bnd->bn", q, c) / (self.d_model ** 0.5)
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
