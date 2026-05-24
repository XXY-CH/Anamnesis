"""Small Transformer language-model baseline for synthetic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int | None = None
    max_seq_len: int = 2048
    dropout: float = 0.0


class TransformerLM(nn.Module):
    """Minimal causal Transformer baseline."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        d_ff = config.d_ff or config.d_model * 4
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_metrics: bool = False,
        return_hidden_only: bool = False,
    ):
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.layers(x, mask=mask)
        x = self.final_norm(x)
        if return_hidden_only:
            return x
        logits = self.output_head(x)
        if return_metrics:
            return logits, {}
        return logits
