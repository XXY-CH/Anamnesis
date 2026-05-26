"""Paper-faithful hashed N-gram Engram lookup.

Engram here is static conditional memory: deterministic N-gram hashes retrieve
embedding rows, then a context-aware gate injects a tiny residual branch.

The embedding tables can optionally live on a separate device (usually CPU).
That models the Engram paper's host-memory/offload direction: keep large static
tables out of accelerator memory, move only the retrieved memory activations
back to the residual stream device.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HashedNgramEngram(nn.Module):
    """Deterministic multi-head N-gram hash lookup with gated residual fusion."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_slots: int = 4096,
        max_ngram: int = 3,
        num_hash_heads: int = 4,
        init_scale: float = 1e-4,
        gate_bias: float = -3.0,
        dropout: float = 0.0,
        table_device: str | torch.device | None = None,
        use_conv: bool = True,
        vector_gate: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_slots = num_slots
        self.max_ngram = max_ngram
        self.num_hash_heads = num_hash_heads
        self.table_device = torch.device(table_device) if table_device is not None else None
        self.use_conv = use_conv
        self.vector_gate = vector_gate

        self.tables = nn.ModuleList(
            [nn.Embedding(num_slots, d_model) for _ in range(max_ngram * num_hash_heads)]
        )
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=d_model**-0.5)
        self._move_tables_to_table_device()

        self.hidden_norm = nn.RMSNorm(d_model)
        self.key_norm = nn.RMSNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias)))
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

        if use_conv:
            self.conv_norm = nn.RMSNorm(d_model)
            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=1,
                padding=0,
                dilation=3,
                groups=d_model,
                bias=False,
            )
            nn.init.normal_(self.conv.weight, std=d_model**-0.5)

        salts = torch.arange(1, max_ngram * num_hash_heads + 1, dtype=torch.long)
        self.register_buffer("salts", salts * 0x9E3779B1, persistent=False)

    @property
    def table_parameter_count(self) -> int:
        """Number of scalar parameters held by Engram lookup tables."""
        return self.max_ngram * self.num_hash_heads * self.num_slots * self.d_model

    def _move_tables_to_table_device(self) -> None:
        """Keep lookup tables on the requested offload device."""
        if self.table_device is None:
            return
        for table in self.tables:
            table.to(self.table_device)

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        """Apply module moves while preserving explicit table offload placement."""
        result = super()._apply(fn)
        self._move_tables_to_table_device()
        return result

    def table_memory_bytes(self) -> int:
        """Return lookup-table storage in bytes for the current table dtype."""
        if not self.tables:
            return 0
        element_size = self.tables[0].weight.element_size()
        return self.table_parameter_count * element_size

    def table_devices(self) -> set[torch.device]:
        """Return devices currently holding Engram lookup tables."""
        return {table.weight.device for table in self.tables}

    def _hash_suffixes(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Return hash indices for all n-gram orders and heads."""
        batch, seq_len = input_ids.shape
        device = input_ids.device
        ids = input_ids.to(torch.long)
        hashes: list[torch.Tensor] = []
        table_idx = 0

        for order in range(1, self.max_ngram + 1):
            h = torch.zeros(batch, seq_len, device=device, dtype=torch.long)
            for offset in range(order):
                shifted = torch.zeros_like(ids)
                if offset == 0:
                    shifted = ids
                else:
                    shifted[:, offset:] = ids[:, :-offset]
                multiplier = 0x85EBCA77 + 0xC2B2AE3D * (offset + 1)
                h = h ^ ((shifted + 1) * multiplier)

            # Avoid pretending incomplete prefixes are full n-grams.
            if order > 1:
                prefix = torch.arange(seq_len, device=device) < (order - 1)
                h[:, prefix] = ids[:, prefix]

            for _ in range(self.num_hash_heads):
                salt = self.salts[table_idx].to(device=device)
                hashes.append((h ^ salt) % self.num_slots)
                table_idx += 1

        return hashes

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled Engram residual and gate values.

        Args:
            hidden: Current hidden states [batch, seq_len, d_model].
            input_ids: Token IDs [batch, seq_len].
        """
        hashes = self._hash_suffixes(input_ids)
        retrieved = []
        for table, index in zip(self.tables, hashes):
            table_index = index.to(device=table.weight.device)
            retrieved.append(table(table_index))
        memory = torch.stack(retrieved, dim=0).mean(dim=0).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )

        norm_hidden = self.hidden_norm(hidden)
        key = self.key_proj(memory)
        norm_key = self.key_norm(key)

        if self.vector_gate:
            score = (norm_hidden * norm_key) / (self.d_model ** 0.5)
        else:
            score = (norm_hidden * norm_key).sum(dim=-1, keepdim=True) / (self.d_model ** 0.5)
        gate = torch.sigmoid(score + self.gate_bias)

        value = self.value_proj(memory)
        gated = gate * value

        if self.use_conv:
            # Causal depthwise 1D convolution (Equation 5)
            conv_in = self.conv_norm(gated).transpose(1, 2)
            padded = torch.nn.functional.pad(conv_in, (9, 0))
            conv_out = self.conv(padded).transpose(1, 2)
            Y = torch.nn.functional.silu(conv_out) + gated
        else:
            Y = gated

        residual = self.residual_scale.abs() * self.dropout(Y)
        return residual, gate
