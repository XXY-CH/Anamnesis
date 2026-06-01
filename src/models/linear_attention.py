"""Linear Attention baseline (Katharopoulos et al., 2020).

Elu-based linear attention for O(N) training and O(1) inference.
Used as a fair comparison point for efficient sequence models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttentionLayer(nn.Module):
    """Single-head linear attention with ELU feature map."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        # ELU feature map + 1 (ensure positive)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Linear attention: O(N) via cumulative sum
        # kv = cumsum(k^T v)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        qkv = torch.einsum("bhnd,bhde->bhne", q, kv)
        # Normalizer: cumsum(k)
        k_sum = k.cumsum(dim=2)
        qk_sum = torch.einsum("bhnd,bhnd->bhn", q, k_sum).unsqueeze(-1)
        out = qkv / (qk_sum + 1e-6)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return residual + out


class LinearAttentionBlock(nn.Module):
    """Pre-norm linear attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = LinearAttentionLayer(d_model, n_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LinearAttentionModel(nn.Module):
    """Linear Attention language model for baseline comparison."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 8,
        d_ff: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LinearAttentionBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        x = self.embedding(idx)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits
