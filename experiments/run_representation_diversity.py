#!/usr/bin/env python
"""Fig 6: Representation diversity heatmap.

Loads trained d=128 Shakespeare model, extracts per-(layer, head) retention
outputs from layers 0-3, computes pairwise cosine similarity, plots 16x16 heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.anamnesis import AnamnesisConfig, AnamnesisModel, sinusoidal_encoding
from experiments.train_real import CharTokenizer

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
    "axes.edgecolor": "#333333",
    "grid.color": "#EAEAEA",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
})

OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = ROOT / "experiments/results/real/scalar_engram_shakespeare/model.pt"
TOKENIZER_PATH = ROOT / "experiments/results/real/scalar_engram_shakespeare/tokenizer.json"

N_LAYERS_CAPTURE = 4
N_HEADS = 4


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device) -> tuple[AnamnesisModel, CharTokenizer, AnamnesisConfig]:
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    config = AnamnesisConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=N_HEADS,
        n_layers=8,
        d_ff=512,
        max_seq_len=512,
        position_encoding_type="sinusoidal",
        engram_layers=(2,),
        engram_num_slots=4096,
        engram_max_ngram=3,
        engram_hash_heads=4,
        engram_use_conv=True,
        engram_vector_gate=False,
        attnres_every=0,
        branch_init_scale=1e-4,
    )
    model = AnamnesisModel(config).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, config


def extract_per_head_representations(
    model: AnamnesisModel,
    input_ids: torch.Tensor,
    config: AnamnesisConfig,
    device: torch.device,
) -> dict[tuple[int, int], torch.Tensor]:
    """Extract mean-pooled per-head retention output from layers 0-3."""
    captured: dict[tuple[int, int], torch.Tensor] = {}

    with torch.no_grad():
        _batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_enc = sinusoidal_encoding(positions, config.d_model)
        x = model.dropout(model.token_embedding(input_ids) + pos_enc)

        depth_sources: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(model.layers):
            u = layer.retention_norm(x)

            q = layer.retention._split_heads(layer.retention.q_proj(u))
            k = layer.retention._split_heads(layer.retention.k_proj(u))
            v = layer.retention._split_heads(layer.retention.v_proj(u))

            diff = (torch.arange(seq_len, device=device).unsqueeze(1)
                    - torch.arange(seq_len, device=device).unsqueeze(0))
            causal_mask = (diff >= 0).to(dtype=u.dtype)
            decay = layer.retention.gamma.to(device=device, dtype=u.dtype)
            decay_mask = decay.unsqueeze(-1).unsqueeze(-1) ** diff.abs()
            decay_mask = decay_mask * causal_mask

            attn = torch.einsum("bhsd,bhtd->bhst", q, k) * decay_mask
            per_head = torch.einsum("bhst,bhtd->bhsd", attn, v)

            if layer_idx < N_LAYERS_CAPTURE:
                for h in range(N_HEADS):
                    captured[(layer_idx, h)] = per_head[0, h].mean(dim=0)

            ret_out = layer.retention.out_proj(layer.retention._merge_heads(per_head))
            ffn_out = layer.ffn(layer.ffn_norm(x))
            x = x + ret_out + ffn_out

            if layer.engram is not None:
                eng_res, _ = layer.engram(layer.ffn_norm(x), input_ids)
                x = x + eng_res

            if layer.attnres is not None and depth_sources:
                active = depth_sources[-layer.attnres.max_sources:]
                attnres_res, _ = layer.attnres(layer.ffn_norm(x), active)
                x = x + attnres_res

            depth_sources.append(x)

    return captured


def plot_diversity_heatmap(sim_matrix: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))

    cmap = sns.color_palette("YlOrBr", as_cmap=True)

    sns.heatmap(
        sim_matrix,
        ax=ax,
        cmap=cmap,
        vmin=-0.5,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        square=True,
        annot_kws={"fontsize": 7},
    )

    ax.set_title(
        "Representation Diversity: Per-Head Cosine Similarity Matrix",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Head (Layer, Head)", fontsize=10, labelpad=8)
    ax.set_ylabel("Head (Layer, Head)", fontsize=10, labelpad=8)
    ax.tick_params(axis="both", labelsize=8.5, rotation=45)

    fig.savefig(OUT_DIR / "fig6_representation_diversity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print("Loading model...")
    model, tokenizer, config = load_model(device)
    print(f"Config: d={config.d_model}, heads={config.n_heads}, layers={config.n_layers}")

    text = (
        "To be, or not to be, that is the question: "
        "Whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, "
        "or to take arms against a sea of troubles"
    )
    input_ids = torch.tensor([tokenizer.encode(text)], device=device)
    print(f"Input: {input_ids.shape[1]} tokens")

    print("Extracting per-head representations...")
    captured = extract_per_head_representations(model, input_ids, config, device)
    print(f"Captured {len(captured)} (layer, head) representations")

    keys = [(l, h) for l in range(N_LAYERS_CAPTURE) for h in range(N_HEADS)]
    vectors = torch.stack([captured[k] for k in keys])
    norms = F.normalize(vectors, dim=-1)
    sim = torch.mm(norms, norms.t()).cpu().numpy()

    labels = [f"L{l+1}H{h+1}" for l, h in keys]

    off_diag = sim[~np.eye(len(keys), dtype=bool)]
    print(f"Similarity range: [{sim.min():.3f}, {sim.max():.3f}]")
    print(f"Mean off-diagonal: {off_diag.mean():.3f}")
    print(f"Std off-diagonal: {off_diag.std():.3f}")

    plot_diversity_heatmap(sim, labels)
    print(f"Saved to {OUT_DIR / 'fig6_representation_diversity.pdf'}")


if __name__ == "__main__":
    main()
