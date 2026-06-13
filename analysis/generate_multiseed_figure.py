#!/usr/bin/env python3
"""Generate multi-seed comparison figure from REAL CSV data.

Shows mean +/- std validation PPL across seeds for d=128 and d=256.
Directly addresses reviewer concern about statistical validation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "experiments" / "results" / "real"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

C_ANAM = "#2563EB"
C_RET = "#059669"
C_TF = "#DC2626"


def get_final_ppl(csv_path: Path) -> float:
    last: float | None = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = row.get("val_ppl", "")
            if vp:
                last = float(vp)
    if last is None:
        raise ValueError(f"No val_ppl in {csv_path}")
    return last


multiseed = {
    r"$d_{model}=128$": {
        "Anamnesis": [
            "hparam_lr1e3_5k",
            "hparam_lr1e3_5k_s100",
            "hparam_lr1e3_5k_s200",
        ],
        "Bare RetNet": ["retnet_8hlw_lr1e3_s42"],
        "Transformer": [
            "transformer_d128_lr1e3_s42",
            "transformer_d128_lr1e3_s100",
            "transformer_d128_lr1e3_s200",
        ],
    },
    r"$d_{model}=256$": {
        "Anamnesis": [
            "anamnesis_d256_lr1e3_s42",
            "anamnesis_d256_lr1e3_s100",
            "anamnesis_d256_lr1e3_s200",
        ],
        "Bare RetNet": ["retnet_bare_d256_lr1e3_s42_5k"],
        "Transformer": [
            "transformer_d256_lr1e3_s42",
            "transformer_d256_lr1e3_s100",
        ],
    },
}

model_colors = {"Anamnesis": C_ANAM, "Bare RetNet": C_RET, "Transformer": C_TF}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (panel_title, models) in zip([ax1, ax2], multiseed.items()):
    x_labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    n_seeds: list[int] = []
    colors: list[str] = []
    all_vals: list[list[float]] = []

    for model_name, dirs in models.items():
        vals: list[float] = []
        for d in dirs:
            csv_path = REAL / d / "results.csv"
            if csv_path.exists():
                ppl = get_final_ppl(csv_path)
                vals.append(ppl)
                print(f"{panel_title} {model_name} {d}: {ppl:.3f}")

        all_vals.append(vals)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        n_seeds.append(len(vals))
        colors.append(model_colors[model_name])
        x_labels.append(model_name)

    x = np.arange(len(x_labels))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.75,
           capsize=5, edgecolor="black", linewidth=0.5, width=0.6)

    for i, vals in enumerate(all_vals):
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(x[i] + jitter, vals, color="black", zorder=5,
                   s=20, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Validation PPL (5K steps)")
    ax.set_title(panel_title)
    ax.grid(True, axis="y", alpha=0.2)

    for i, (m, s, n) in enumerate(zip(means, stds, n_seeds)):
        label = f"{m:.2f}" if n == 1 else f"{m:.2f}+/-{s:.2f}"
        ax.text(x[i], m + s + 0.08, f"{label}\n(n={n})",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(means) + max(stds) + 0.8)

fig.suptitle(
    "Multi-Seed Validation (Shakespeare char-level, lr=$10^{-3}$, 5K steps)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_paths = [
    ROOT / "analysis" / "figures" / "fig_multiseed_real.pdf",
    ROOT / "analysis" / "figures" / "fig_multiseed_real.png",
    ROOT / "neurocomputing_submission_package" / "figures" / "fig_multiseed_real.pdf",
]
for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), bbox_inches="tight",
                dpi=150 if p.suffix == ".png" else 300)
    print(f"Saved: {p}")

plt.close(fig)
