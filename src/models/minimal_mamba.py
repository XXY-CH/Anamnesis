"""Minimal Selective SSM (Mamba-like) baseline for comparison.

Implements a simplified selective state-space model with:
- Input-dependent B, C matrices (selection mechanism)
- Linear recurrence: h_t = A*h_{t-1} + B(x_t)*x_t
- Output: y_t = C(x_t)*h_t
- Sequential scan for training

No external mamba-ssm dependency needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSMConfig:
    vocab_size: int = 67
    d_model: int = 128
    n_layers: int = 8
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4    # Local convolution width
    expand_factor: int = 2  # Inner dimension = d_model * expand_factor
    max_seq_len: int = 2048


class SelectiveSSMBlock(nn.Module):
    """Single selective SSM block (simplified Mamba layer)."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        d = config.d_model
        d_inner = d * config.expand_factor
        d_state = config.d_state

        self.d_inner = d_inner
        self.d_state = d_state

        # Input projections
        self.in_proj = nn.Linear(d, d_inner * 2, bias=False)

        # Local convolution (causal depthwise)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=config.d_conv,
            padding=config.d_conv - 1, groups=d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(d_inner, 1))
        )

        # Output projection
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.norm = nn.RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        B, L, _ = x.shape
        residual = x

        # Input projection + split into two branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal conv1d
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_act = F.silu(x_conv)

        # Input-dependent SSM parameters
        ssm_params = self.x_proj(x_act)
        dt = F.softplus(ssm_params[:, :, 0:1])
        B_mat = ssm_params[:, :, 1:1 + self.d_state]
        C_mat = ssm_params[:, :, 1 + self.d_state:]

        # Discretize A
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)

        # Sequential scan
        h = x.new_zeros(B, self.d_inner, self.d_state)
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_act[:, t:t+1, :].transpose(1, 2)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return self.norm(residual + out)


class MinimalMambaModel(nn.Module):
    """Minimal Mamba-like model for baseline comparison."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            SelectiveSSMBlock(config) for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.embed.weight, std=std)
        nn.init.normal_(self.head.weight, std=std * 0.5)

    def forward(self, input_ids: torch.Tensor, return_metrics: bool = False, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        if return_metrics:
            return logits, {}
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
