#!/usr/bin/env python3
"""Generate training-curve figures from REAL CSV data.

Plots actual val_ppl trajectories (logged every 500 steps) from the
20K-step Shakespeare char-level runs. No hardcoded values.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_val_ppl(csv_path: Path) -> tuple[list[int], list[float]]:
    """Load validation PPL from a results CSV."""
    steps: list[int] = []
    ppls: list[float] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = row.get("val_ppl", "")
            if vp:
                steps.append(int(row["step"]))
                ppls.append(float(vp))
    return steps, ppls


def main() -> None:
    runs_128 = [
        ("Anamnesis", REAL / "anamnesis_shakespeare_char_lr1e3_s42_20k" / "results.csv", C_ANAM, "D-"),
        ("Bare RetNet 8h+lw", REAL / "retnet_shakespeare_char_lr1e3_s42_20k" / "results.csv", C_RET, "s--"),
        ("Transformer", REAL / "transformer_shakespeare_char_lr1e3_s42_20k" / "results.csv", C_TF, "o-"),
    ]

    runs_256 = [
        ("Anamnesis", REAL / "anamnesis_shakespeare_char_d256_lr1e3_s42_20k" / "results.csv", C_ANAM, "D-"),
        ("Transformer", REAL / "transformer_shakespeare_char_d256_lr1e3_s42_20k" / "results.csv", C_TF, "o-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, csv_path, color, style in runs_128:
        if not csv_path.exists():
            print(f"MISS: {csv_path}")
            continue
        steps, ppls = load_val_ppl(csv_path)
        ax1.plot(steps, ppls, style, color=color, label=label,
                 markersize=5, linewidth=1.8, alpha=0.85)
        print(f"d=128 {label}: {len(steps)} val points, final={ppls[-1]:.3f}")

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Validation Perplexity")
    ax1.set_title(r"(a) $d_{model}=128$")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-500, 21000)

    for label, csv_path, color, style in runs_256:
        if not csv_path.exists():
            print(f"MISS: {csv_path}")
            continue
        steps, ppls = load_val_ppl(csv_path)
        ax2.plot(steps, ppls, style, color=color, label=label,
                 markersize=5, linewidth=1.8, alpha=0.85)
        print(f"d=256 {label}: {len(steps)} val points, final={ppls[-1]:.3f}")

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title(r"(b) $d_{model}=256$")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(-500, 21000)

    fig.suptitle(
        "Validation PPL Trajectories (Shakespeare char-level, lr=$10^{-3}$, seed=42, 20K steps)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_paths = [
        ROOT / "analysis" / "figures" / "fig_training_curves_real.pdf",
        ROOT / "analysis" / "figures" / "fig_training_curves_real.png",
        ROOT / "neurocomputing_submission_package" / "figures" / "fig_training_curves_real.pdf",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), bbox_inches="tight",
                    dpi=150 if p.suffix == ".png" else 300)
        print(f"Saved: {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()
