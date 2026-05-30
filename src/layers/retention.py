"""
Retention Layer - Core RetNet building block.

Implements parallel, recurrent, and chunkwise retention mechanisms.
"""

import torch
import torch.nn as nn


class RetentionLayer(nn.Module):
    """Single retention layer with multi-head retention.

    Supports three computation modes:
    - parallel: for training (full sequence at once)
    - recurrent: for autoregressive inference (step-by-step)
    - chunkwise: for long sequences (chunked processing)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 256,
        dropout: float = 0.0,
        input_dependent_gamma: bool = False,
        input_dependent_write: bool = False,
        learnable_gamma: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size
        self.input_dependent_gamma = input_dependent_gamma
        self.input_dependent_write = input_dependent_write

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_decay(learnable=learnable_gamma)

        if input_dependent_gamma:
            self.gamma_proj = nn.Linear(d_model, n_heads, bias=True)
            with torch.no_grad():
                nn.init.zeros_(self.gamma_proj.weight)
                self.gamma_proj.bias.copy_(
                    (self.gamma / (1 - self.gamma)).log()
                )

        if input_dependent_write:
            self.write_proj = nn.Linear(d_model, n_heads, bias=True)
            with torch.no_grad():
                nn.init.zeros_(self.write_proj.weight)
                nn.init.constant_(self.write_proj.bias, 2.0)

    def _init_decay(self, learnable: bool = False) -> None:
        """Initialize per-head decay rates below one.

        Args:
            learnable: If True, gamma is an nn.Parameter (optimized by AdamW).
                       If False, gamma is a fixed buffer (never updated).
        """
        if self.n_heads == 1:
            gamma = torch.tensor([1 - 2**-5], dtype=torch.float32)
        else:
            gamma = 1 - torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(1 / 32, dtype=torch.float32)),
                    torch.log(torch.tensor(1 / 512, dtype=torch.float32)),
                    self.n_heads,
                )
            )
        if learnable:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

    def _compute_dynamic_gamma(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gamma_proj(x))

    def _compute_write_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute input-dependent write gate: [batch, seq_len, n_heads]."""
        return torch.sigmoid(self.write_proj(x))

    def _resolve_gate(
        self,
        x: torch.Tensor,
        retention_gate: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.input_dependent_gamma:
            return retention_gate
        dynamic = self._compute_dynamic_gamma(x)
        if retention_gate is not None:
            return torch.maximum(retention_gate, dynamic)
        return dynamic

    def parallel_retention(
        self,
        x: torch.Tensor,
        retention_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel retention for training.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Retained output [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        retention_gate = self._resolve_gate(x, retention_gate)

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Apply input-dependent write gate: scale V per-position per-head
        if self.input_dependent_write:
            wg = self._compute_write_gate(x)  # [batch, seq_len, n_heads]
            wg = wg.permute(0, 2, 1).unsqueeze(-1)  # [batch, heads, seq, 1]
            v = v * wg

        positions = torch.arange(seq_len, device=x.device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [i, j] = i - j
        causal_mask = (diff >= 0).to(dtype=x.dtype)

        if retention_gate is None:
            # Causal decay mask: D[i,j] = gamma^(i-j) for i >= j
            decay = self.gamma.to(device=x.device, dtype=x.dtype)
            decay_mask = decay.unsqueeze(-1).unsqueeze(-1) ** diff.abs()
            decay_mask = decay_mask * causal_mask
        else:
            # Batch-specific gated decay:
            # D[b,h,i,j] = prod_{r=j+1..i} gate[b,r,h].
            gate = retention_gate.to(device=x.device, dtype=x.dtype)
            if gate.shape != (batch, seq_len, self.n_heads):
                raise ValueError(
                    "retention_gate must have shape "
                    f"{(batch, seq_len, self.n_heads)}, got {tuple(gate.shape)}"
                )
            log_gate = torch.log(gate.clamp_min(torch.finfo(x.dtype).tiny))
            prefix = torch.cumsum(log_gate, dim=1)  # [b, s, h]
            prefix_i = prefix.unsqueeze(2)
            prefix_j = prefix.unsqueeze(1)
            log_decay = (prefix_i - prefix_j).permute(0, 3, 1, 2)
            log_decay = log_decay.masked_fill(
                causal_mask.view(1, 1, seq_len, seq_len) < 0.5, float("-inf")
            )
            decay_mask = torch.exp(log_decay)

        # Retention: (Q K^T * D) V
        attn = torch.einsum("bhsd,bhtd->bhst", q, k) * decay_mask
        attn = self.dropout(attn)
        output = torch.einsum("bhst,bhtd->bhsd", attn, v)

        output = self._merge_heads(output)
        return self.out_proj(output)

    def recurrent_retention(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        retention_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent retention for autoregressive inference.

        Args:
            x: Input tensor [batch, 1, d_model]
            state: Previous state [batch, n_heads, head_dim, head_dim]

        Returns:
            Output tensor and updated state
        """
        batch = x.shape[0]

        retention_gate = self._resolve_gate(x, retention_gate)
        if retention_gate is not None and retention_gate.dim() == 3:
            retention_gate = retention_gate.squeeze(1)

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if state is None:
            state = torch.zeros(batch, self.n_heads, self.head_dim, self.head_dim, device=x.device)

        # Update state: S = gamma * S + write_gate * k^T v
        if retention_gate is None:
            gamma = self.gamma.to(device=x.device, dtype=x.dtype).view(1, self.n_heads, 1, 1)
        else:
            gamma = retention_gate.to(device=x.device, dtype=x.dtype).view(
                batch, self.n_heads, 1, 1
            )
        kv = torch.einsum("bhld,bhle->bhde", k, v)
        if self.input_dependent_write:
            wg = self._compute_write_gate(x)  # [batch, 1, n_heads]
            wg = wg.squeeze(1).view(batch, self.n_heads, 1, 1)
            kv = kv * wg
        new_state = gamma * state + kv

        # Output: q @ state
        output = torch.einsum("bhld,bhde->bhle", q, new_state)
        output = self._merge_heads(output)

        return self.out_proj(output), new_state

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "parallel",
        state: torch.Tensor | None = None,
        retention_gate: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with selectable computation mode.

        Args:
            x: Input tensor
            mode: 'parallel', 'recurrent', or 'chunkwise'
            state: Previous state (for recurrent mode)

        Returns:
            Retained output, optionally with updated state
        """
        if mode == "parallel":
            return self.parallel_retention(x, retention_gate=retention_gate)
        elif mode == "recurrent":
            return self.recurrent_retention(x, state, retention_gate=retention_gate)
        else:
            raise ValueError(f"Unknown retention mode: {mode}")

    def chunkwise_retention(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        retention_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise retention: parallel within chunk, recurrent across chunks.

        Processes a chunk with full parallel retention internally while
        incorporating cross-chunk context from the previous recurrent state.

        Args:
            x: Chunk input [batch, chunk_len, d_model]
            state: Previous recurrent state [batch, n_heads, head_dim, head_dim]
            retention_gate: Optional per-step gate [batch, chunk_len, n_heads]

        Returns:
            (output [batch, chunk_len, d_model], updated_state)
        """
        batch, chunk_len, _ = x.shape

        retention_gate = self._resolve_gate(x, retention_gate)

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Compute write gate once for both within-chunk and state update
        write_gate = None
        if self.input_dependent_write:
            write_gate = self._compute_write_gate(x)  # [batch, chunk_len, n_heads]
            wg = write_gate.permute(0, 2, 1).unsqueeze(-1)  # [batch, heads, chunk, 1]
            v = v * wg

        if state is None:
            state = torch.zeros(
                batch, self.n_heads, self.head_dim, self.head_dim, device=x.device
            )

        positions = torch.arange(chunk_len, device=x.device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        causal_mask = (diff >= 0).to(dtype=x.dtype)

        if retention_gate is None:
            decay = self.gamma.to(device=x.device, dtype=x.dtype)
            decay_mask = decay.unsqueeze(-1).unsqueeze(-1) ** diff.abs()
            decay_mask = decay_mask * causal_mask
        else:
            gate = retention_gate.to(device=x.device, dtype=x.dtype)
            log_gate = torch.log(gate.clamp_min(torch.finfo(x.dtype).tiny))
            prefix = torch.cumsum(log_gate, dim=1)
            log_decay = (prefix.unsqueeze(2) - prefix.unsqueeze(1)).permute(0, 3, 1, 2)
            log_decay = log_decay.masked_fill(
                causal_mask.view(1, 1, chunk_len, chunk_len) < 0.5, float("-inf")
            )
            decay_mask = torch.exp(log_decay)

        within_chunk = torch.einsum("bhsd,bhtd->bhst", q, k) * decay_mask
        within_chunk = self.dropout(within_chunk)
        within_output = torch.einsum("bhst,bhtd->bhsd", within_chunk, v)

        cross_chunk = torch.einsum("bhld,bhde->bhle", q, state)
        # Apply position-dependent decay from previous chunk boundary.
        # In the gated case, use cumulative gate product; otherwise use gamma.
        if retention_gate is None:
            decay_per_head = self.gamma.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            pos_offset = torch.arange(
                1, chunk_len + 1, device=x.device, dtype=x.dtype
            ).view(1, 1, -1, 1)
            cross_chunk = cross_chunk * (decay_per_head ** pos_offset)
        else:
            gate = retention_gate.to(device=x.device, dtype=x.dtype)
            log_gate = torch.log(gate.clamp_min(torch.finfo(x.dtype).tiny))
            cum_log = torch.cumsum(log_gate, dim=1)  # [batch, chunk_len, n_heads]
            # cross_decay[b, h, i] = exp(cum_log[b, i, h]) = product(gate[0..i])
            cross_decay = torch.exp(cum_log.permute(0, 2, 1).unsqueeze(-1))
            cross_chunk = cross_chunk * cross_decay
        output = within_output + cross_chunk
        output = self._merge_heads(output)
        output = self.out_proj(output)

        if retention_gate is None:
            gamma = self.gamma.to(device=x.device, dtype=x.dtype).view(
                1, self.n_heads, 1, 1
            )
            new_state = gamma.pow(chunk_len) * state
            for t in range(chunk_len):
                decay_t = gamma.pow(chunk_len - 1 - t)
                update = torch.einsum(
                    "bhld,bhle->bhde", k[:, :, t:t + 1], v[:, :, t:t + 1]
                )
                new_state = new_state + decay_t * update
        else:
            gate = retention_gate.to(device=x.device, dtype=x.dtype)
            log_gate = torch.log(gate.clamp_min(torch.finfo(x.dtype).tiny))
            total_log = log_gate.sum(dim=1)
            new_state = torch.exp(
                total_log.unsqueeze(-1).unsqueeze(-1)
            ) * state
            for t in range(chunk_len):
                remaining_log = log_gate[:, t + 1:].sum(dim=1)
                decay_t = torch.exp(remaining_log.unsqueeze(-1).unsqueeze(-1))
                update = torch.einsum(
                    "bhld,bhle->bhde", k[:, :, t:t + 1], v[:, :, t:t + 1]
                )
                new_state = new_state + decay_t * update

        return output, new_state

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
