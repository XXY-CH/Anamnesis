#!/usr/bin/env python3
"""Generate cross-dataset comparison figure from REAL CSV data.

Shows that Anamnesis advantage scales with data diversity:
  Shakespeare (diverse) > WikiText-2 (moderate) > TinyStories (repetitive).
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


datasets = [
    ("Shakespeare\n(diverse)", {
        "Anamnesis": "anamnesis_shakespeare_char_lr1e3_s42_10k",
        "Transformer": "transformer_shakespeare_char_lr1e3_s42_10k",
    }),
    ("WikiText-2\n(moderate)", {
        "Anamnesis": "anamnesis_wikitext2_char_lr1e3_s42_10k",
        "Transformer": "transformer_wikitext2_char_lr1e3_s42_10k",
    }),
    ("TinyStories\n(repetitive)", {
        "Anamnesis": "anamnesis_tinystories_char_lr1e3_s42_10k",
        "Transformer": "transformer_tinystories_char_lr1e3_s42_10k",
    }),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

x_labels: list[str] = []
anam_ppls: list[float] = []
tf_ppls: list[float] = []
deltas: list[float] = []

for ds_name, dirs in datasets:
    a_ppl = get_final_ppl(REAL / dirs["Anamnesis"] / "results.csv")
    t_ppl = get_final_ppl(REAL / dirs["Transformer"] / "results.csv")
    delta = (a_ppl - t_ppl) / t_ppl * 100
    anam_ppls.append(a_ppl)
    tf_ppls.append(t_ppl)
    deltas.append(delta)
    x_labels.append(ds_name)
    print(f"{ds_name.replace(chr(10), ' ')}: Anam={a_ppl:.3f} TF={t_ppl:.3f} delta={delta:+.1f}%")

x = np.arange(len(x_labels))
w = 0.32
bars_a = ax1.bar(x - w/2, anam_ppls, w, color=C_ANAM, alpha=0.8,
                 edgecolor="black", linewidth=0.5, label="Anamnesis")
bars_t = ax1.bar(x + w/2, tf_ppls, w, color=C_TF, alpha=0.8,
                 edgecolor="black", linewidth=0.5, label="Transformer")

for bar, val in zip(bars_a, anam_ppls):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars_t, tf_ppls):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=9.5)
ax1.set_ylabel("Validation PPL (10K steps)")
ax1.set_title(r"(a) Cross-Dataset PPL ($d=128$)")
ax1.legend()
ax1.grid(True, axis="y", alpha=0.2)
ax1.set_ylim(0, max(tf_ppls) + 0.5)

colors = ["#2E7D32" if d < 0 else "#C62828" for d in deltas]
bars_d = ax2.bar(x, deltas, color=colors, alpha=0.8,
                 edgecolor="black", linewidth=0.5, width=0.5)

for bar, val in zip(bars_d, deltas):
    offset = 1.5 if val >= 0 else -2.5
    ax2.text(bar.get_x() + bar.get_width()/2, val + offset,
             f"{val:+.1f}%", ha="center",
             va="bottom" if val >= 0 else "top",
             fontsize=10, fontweight="bold")

ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=9.5)
ax2.set_ylabel(r"$\Delta$ vs Transformer (\%)")
ax2.set_title(r"(b) Advantage $\propto$ Data Diversity")
ax2.grid(True, axis="y", alpha=0.2)

ax2.annotate(
    "Anamnesis wins\n(diverse data)",
    xy=(0, deltas[0]), xytext=(0.5, deltas[0] - 8),
    fontsize=8.5, ha="center", color="#1B5E20",
    arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1),
)
ax2.annotate(
    "Transformer wins\n(repetitive data)",
    xy=(2, deltas[2]), xytext=(1.5, deltas[2] + 8),
    fontsize=8.5, ha="center", color="#B71C1C",
    arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1),
)

fig.suptitle(
    "Cross-Dataset Validation (char-level, $d=128$, lr=$10^{-3}$, seed=42, 10K steps)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_paths = [
    ROOT / "analysis" / "figures" / "fig_cross_dataset_real.pdf",
    ROOT / "analysis" / "figures" / "fig_cross_dataset_real.png",
    ROOT / "neurocomputing_submission_package" / "figures" / "fig_cross_dataset_real.pdf",
]
for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), bbox_inches="tight",
                dpi=150 if p.suffix == ".png" else 300)
    print(f"Saved: {p}")

plt.close(fig)
