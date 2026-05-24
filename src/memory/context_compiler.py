"""Context Compiler — preprocesses long context into queryable external memory.

Proof 34 implementation: three-stage compiler that converts a long sequence
into a fixed-size set of (key, value) pairs, queryable by the Small Reasoner
via gated attention. O(1) in sequence length during inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompiledMemory:
    """Fixed-size external memory store produced by the Context Compiler."""

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        self.keys = keys        # [K, d]
        self.values = values    # [K, d]
        self.positions = positions  # [K]

    @property
    def size(self) -> int:
        return self.keys.shape[0]

    def to(self, device: torch.device) -> CompiledMemory:
        return CompiledMemory(
            self.keys.to(device),
            self.values.to(device),
            self.positions.to(device),
        )


class MemoryQueryHead(nn.Module):
    """Gated attention head for querying compiled memory."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_proj.bias, -3.0)

        self.query_norm = nn.RMSNorm(d_model)
        self.key_norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(1e-4))

    def forward(
        self,
        hidden: torch.Tensor,
        memory: CompiledMemory | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if memory is None:
            return torch.zeros_like(hidden), None

        batch, seq_len, _ = hidden.shape
        K = memory.size

        q = self._split_heads(self.q_proj(self.query_norm(hidden)))
        k = self._split_heads(self.key_norm(memory.keys.unsqueeze(0).expand(batch, -1, -1)))

        scores = torch.einsum("bhsd,bhmd->bhsm", q, k) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        readout = torch.einsum("bhsm,bhmd->bhsd", weights, k)
        readout = self._merge_heads(readout)
        readout = self.out_proj(readout)

        gate = torch.sigmoid(self.gate_proj(hidden))
        output = self.residual_scale.abs() * gate * readout

        return output, weights.mean(dim=(1, 2))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)


class ImportanceScorer(nn.Module):
    """Score each position by predicted importance for memory selection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.scorer(hidden).squeeze(-1)


class ContextCompiler(nn.Module):
    """Compile long context into fixed-size external memory.

    Three stages:
    1. Chunk processing — run RetNet over chunks, collect hidden states
    2. Selection — score and select top-K important positions
    3. Structuring — organize into (key, value) pairs with position encoding
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 256,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.chunk_size = chunk_size

        self.importance_scorer = ImportanceScorer(d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

    @torch.no_grad()
    def compile(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> CompiledMemory:
        """Compile a long sequence into fixed-size memory.

        Args:
            model: RetNetEngramModel to use for chunk processing.
            input_ids: Token IDs [batch=1, seq_len].

        Returns:
            CompiledMemory with top-K (key, value) pairs.
        """
        batch, seq_len = input_ids.shape
        assert batch == 1, "Context Compiler processes one sequence at a time"

        all_hidden: list[torch.Tensor] = []
        all_positions: list[torch.Tensor] = []

        # Stage 1: Process in chunks, collect hidden states
        chunk_size = self.chunk_size
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_ids = input_ids[:, start:end]
            chunk_hidden = model(chunk_ids, return_hidden_only=True)
            all_hidden.append(chunk_hidden.squeeze(0))
            all_positions.append(torch.arange(start, end, device=input_ids.device))

        hidden = torch.cat(all_hidden, dim=0)       # [N, d]
        positions = torch.cat(all_positions, dim=0)  # [N]

        # Stage 2: Score and select top-K
        scores = self.importance_scorer(hidden.unsqueeze(0)).squeeze(0)  # [N]
        K = min(self.memory_size, seq_len)
        topk_indices = scores.topk(K, dim=0).indices

        selected_hidden = hidden[topk_indices]          # [K, d]
        selected_positions = positions[topk_indices]    # [K]

        # Stage 3: Create key-value pairs
        keys = self.key_proj(selected_hidden)
        values = selected_hidden

        return CompiledMemory(
            keys=keys,
            values=values,
            positions=selected_positions,
        )

    def compile_from_hidden(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> CompiledMemory:
        """Compile from pre-computed hidden states (avoids running model).

        Args:
            hidden: Hidden states [batch, seq_len, d_model] or [seq_len, d_model].
            positions: Optional position indices.

        Returns:
            CompiledMemory with top-K (key, value) pairs.
        """
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        batch, seq_len, _ = hidden.shape
        if positions is None:
            positions = torch.arange(seq_len, device=hidden.device)

        scores = self.importance_scorer(hidden).squeeze(-1)  # [batch, seq_len]
        K = min(self.memory_size, seq_len)
        topk_indices = scores.topk(K, dim=-1).indices  # [batch, K]

        idx = topk_indices[0]
        selected_hidden = hidden[0, idx]       # [K, d]
        selected_positions = positions[idx]     # [K]

        keys = self.key_proj(selected_hidden)
        values = selected_hidden

        return CompiledMemory(
            keys=keys,
            values=values,
            positions=selected_positions,
        )
