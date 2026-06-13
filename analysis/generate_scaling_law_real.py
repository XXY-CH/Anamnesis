#!/usr/bin/env python3
"""Generate scaling law figure from REAL CSV endpoint data.

Reads final val_ppl from 5K/10K/20K runs and fits power laws.
Exponents quantify RetNet's steeper scaling vs Transformer.
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
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

C_ANAM = "#2563EB"
C_RET = "#059669"
C_TF = "#DC2626"


def get_final_ppl(csv_path: Path) -> float:
    """Return the last non-empty val_ppl from a results CSV."""
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


runs = {
    "Anamnesis": {
        5: "hparam_lr1e3_5k",
        10: "anamnesis_shakespeare_char_lr1e3_s42_10k",
        20: "anamnesis_shakespeare_char_lr1e3_s42_20k",
        "color": C_ANAM, "marker": "D",
    },
    "Bare RetNet 8h+lw": {
        5: "retnet_8hlw_lr1e3_s42",
        10: "retnet_shakespeare_char_lr1e3_s42_10k",
        20: "retnet_shakespeare_char_lr1e3_s42_20k",
        "color": C_RET, "marker": "s",
    },
    "Transformer": {
        5: "transformer_d128_lr1e3_s42",
        10: "transformer_shakespeare_char_lr1e3_s42_10k",
        20: "transformer_shakespeare_char_lr1e3_s42_20k",
        "color": C_TF, "marker": "o",
    },
}

steps = np.array([5.0, 10.0, 20.0])
fig, ax = plt.subplots(figsize=(7, 5))
exponents: dict[str, float] = {}

for label, info in runs.items():
    ppls = []
    for k in [5, 10, 20]:
        csv_path = REAL / info[k] / "results.csv"
        ppl = get_final_ppl(csv_path)
        ppls.append(ppl)
        print(f"{label} {k}K: {ppl:.3f} ({info[k]})")

    ppls_arr = np.array(ppls)
    log_s = np.log(steps)
    log_p = np.log(ppls_arr)
    b, log_a = np.polyfit(log_s, log_p, 1)
    a = np.exp(log_a)
    exponents[label] = b

    ax.plot(steps, ppls_arr, info["marker"], color=info["color"],
            markersize=10, zorder=5, label=f"{label} (data)")
    s_fine = np.linspace(4.5, 22, 100)
    ax.plot(s_fine, a * s_fine ** b, "--", color=info["color"], alpha=0.6,
            linewidth=1.5, label=f"{label} fit ($b$={b:.3f})")

ax.set_xlabel("Training Steps (K)")
ax.set_ylabel("Validation PPL")
ax.set_title(
    r"Scaling Law: PPL $\propto$ steps$^b$"
    f"\n(Shakespeare char-level, $d=128$, lr=$10^{{-3}}$, seed=42)"
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20])
ax.set_xticklabels(["5K", "10K", "20K"])
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, alpha=0.2, which="both")

ratio = abs(exponents["Bare RetNet 8h+lw"] / exponents["Transformer"])
ax.annotate(
    f"RetNet $|b|$ = {ratio:.1f}$\\times$ Transformer\n"
    f"$\\rightarrow$ steeper scaling = faster improvement",
    xy=(13, 2.0), fontsize=9, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
              edgecolor="#F9A825", alpha=0.8),
)

fig.tight_layout()

out_paths = [
    ROOT / "analysis" / "figures" / "fig_scaling_law_real.pdf",
    ROOT / "analysis" / "figures" / "fig_scaling_law_real.png",
    ROOT / "neurocomputing_submission_package" / "figures" / "fig_scaling_law_real.pdf",
]
for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), bbox_inches="tight",
                dpi=150 if p.suffix == ".png" else 300)
    print(f"Saved: {p}")

plt.close(fig)

print(f"\nFitted exponents (b):")
for label, b in exponents.items():
    print(f"  {label}: {b:.3f}")
print(f"\nRetNet/Transformer ratio: {ratio:.2f}x steeper")
