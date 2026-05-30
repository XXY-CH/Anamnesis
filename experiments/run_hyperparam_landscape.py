#!/usr/bin/env python
"""Fig 7: Hyperparameter landscape heatmap.

Grid search over Engram gate bias and residual scale init on Shakespeare d=128.
Each configuration trains for 100 steps and records validation perplexity.
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

from src.models.anamnesis import AnamnesisConfig, AnamnesisModel
from src.layers.engram import HashedNgramEngram
from experiments.train_real import CharTokenizer, TextDataset, load_dataset, evaluate
from torch.utils.data import DataLoader

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

BIAS_VALUES = [-4.5, -4.0, -3.5, -3.0, -2.5, -2.0]
SCALE_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

STEPS = 500
D_MODEL = 128
SEQ_LEN = 512


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_data(
    tokenizer: CharTokenizer,
) -> tuple[DataLoader, DataLoader]:
    train_text = load_dataset("shakespeare", "train", max_chars=10_000_000)
    valid_text = load_dataset("shakespeare", "valid", max_chars=500_000)

    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)

    train_ds = TextDataset(train_ids, SEQ_LEN)
    valid_ds = TextDataset(valid_ids, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, valid_loader


def create_model_with_params(
    vocab_size: int,
    gate_bias: float,
    init_scale: float,
    device: torch.device,
) -> AnamnesisModel:
    config = AnamnesisConfig(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=4,
        n_layers=8,
        d_ff=512,
        max_seq_len=SEQ_LEN,
        position_encoding_type="sinusoidal",
        engram_layers=(2,),
        engram_num_slots=4096,
        engram_max_ngram=3,
        engram_hash_heads=4,
        engram_use_conv=True,
        engram_vector_gate=False,
        attnres_every=4,
        branch_init_scale=init_scale,
    )
    model = AnamnesisModel(config).to(device)

    for module in model.modules():
        if isinstance(module, HashedNgramEngram):
            module.gate_bias.data.fill_(gate_bias)
            module.residual_scale.data.fill_(init_scale)

    return model


def train_and_eval(
    model: AnamnesisModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
) -> float:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01
    )

    data_iter = iter(train_loader)
    for _ in range(STEPS):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, return_metrics=True)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    val_loss = evaluate(model, valid_loader, device, max_batches=20)
    return torch.exp(torch.tensor(val_loss)).item()


def plot_hyperparam_heatmap(
    ppl_matrix: np.ndarray,
    bias_labels: list[str],
    scale_labels: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = sns.color_palette("YlGn_r", as_cmap=True)

    sns.heatmap(
        ppl_matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        linewidths=1,
        linecolor="white",
        xticklabels=scale_labels,
        yticklabels=bias_labels,
        cbar_kws={"label": "Validation PPL (Lower is Better)", "shrink": 0.8},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    opt_bias_idx = BIAS_VALUES.index(-3.0)
    opt_scale_idx = SCALE_VALUES.index(1e-4)
    ax.add_patch(plt.Rectangle(
        (opt_scale_idx, opt_bias_idx), 1, 1,
        fill=False, edgecolor="#E74C3C", linewidth=3, linestyle="--",
    ))

    ax.set_title(
        "Gate Bias vs. Init Scale: Validation PPL Landscape",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Residual Scale Init ($s_0$)", fontsize=11, labelpad=8)
    ax.set_ylabel("Gate Bias ($b$)", fontsize=11, labelpad=8)
    ax.tick_params(axis="both", labelsize=10)

    ax.annotate(
        "Theory\nprediction",
        xy=(opt_scale_idx + 0.5, opt_bias_idx + 0.5),
        xytext=(opt_scale_idx + 2.5, opt_bias_idx - 0.8),
        fontsize=9, color="#E74C3C", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
        ha="center",
    )

    fig.savefig(OUT_DIR / "fig7_hyperparam_landscape.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print("Loading Shakespeare dataset...")
    train_text = load_dataset("shakespeare", "train", max_chars=10_000_000)
    tokenizer = CharTokenizer(train_text)
    train_loader, valid_loader = prepare_data(tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    n_bias = len(BIAS_VALUES)
    n_scale = len(SCALE_VALUES)
    ppl_matrix = np.zeros((n_bias, n_scale))
    total = n_bias * n_scale

    print(f"\n=== Grid search: {total} configurations x {STEPS} steps ===")

    for i, bias in enumerate(BIAS_VALUES):
        for j, scale in enumerate(SCALE_VALUES):
            idx = i * n_scale + j + 1
            print(
                f"[{idx}/{total}] bias={bias:.1f}, scale={scale:.0e} ... ",
                end="", flush=True,
            )

            torch.manual_seed(42)
            model = create_model_with_params(
                tokenizer.vocab_size, bias, scale, device
            )
            ppl = train_and_eval(model, train_loader, valid_loader, device)
            ppl_matrix[i, j] = ppl
            print(f"PPL={ppl:.2f}")

            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    best_idx = np.unravel_index(ppl_matrix.argmin(), ppl_matrix.shape)
    print(f"\nBest: bias={BIAS_VALUES[best_idx[0]]}, "
          f"scale={SCALE_VALUES[best_idx[1]]}, "
          f"PPL={ppl_matrix[best_idx]:.2f}")
    default_ppl = ppl_matrix[
        BIAS_VALUES.index(-3.0), SCALE_VALUES.index(1e-4)
    ]
    print(f"Default (bias=-3.0, scale=1e-4): PPL={default_ppl:.2f}")

    bias_labels = [f"{b:.1f}" for b in BIAS_VALUES]
    scale_labels = [f"{s:.0e}" for s in SCALE_VALUES]

    import json
    data_out = {
        "description": "Gate bias vs init scale grid search on Shakespeare d=128",
        "steps": STEPS,
        "d_model": D_MODEL,
        "bias_values": BIAS_VALUES,
        "scale_values": SCALE_VALUES,
        "ppl_matrix": ppl_matrix.tolist(),
        "best_bias": BIAS_VALUES[best_idx[0]],
        "best_scale": SCALE_VALUES[best_idx[1]],
        "best_ppl": float(ppl_matrix[best_idx]),
        "default_ppl": float(default_ppl),
    }
    data_path = OUT_DIR / "fig7_hyperparam_landscape_data.json"
    data_path.write_text(json.dumps(data_out, indent=2))
    print(f"Data saved to {data_path}")

    plot_hyperparam_heatmap(ppl_matrix, bias_labels, scale_labels)
    print(f"\nSaved to {OUT_DIR / 'fig7_hyperparam_landscape.pdf'}")


if __name__ == "__main__":
    main()
