#!/usr/bin/env python3
"""Generate retrieval baseline comparison figure for Neurocomputing paper.

Shows: BM25/exact-match baselines (EM=1.000) vs model-based retrieval (EM=0.000).
This is the key negative finding that reframes the retrieval contribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def main() -> None:
    # Load P0 BM25/exact-match baselines (flat structure per entry)
    with open(RESULTS / "retrieval_baselines.json") as f:
        baselines = json.load(f)

    # Load P0 encoder ablation
    with open(RESULTS / "retriever_encoder_ablation.json") as f:
        ablation = json.load(f)

    # Aggregate baselines: average over seeds per context length
    ctx_set = sorted(set(e["total_tokens"] for e in baselines))
    baseline_keys = ["random_em", "exact_match_em", "bm25_em", "needle_token_em"]
    baseline_labels = ["Random", "Exact Match", "BM25", "Needle Token"]
    baseline_colors = ["#888888", "#4CAF50", "#2196F3", "#FF9800"]

    avg_by_len: dict[int, dict[str, float]] = {}
    for cl in ctx_set:
        entries = [e for e in baselines if e["total_tokens"] == cl]
        avg_by_len[cl] = {k: sum(e[k] for e in entries) / len(entries) for k in baseline_keys}

    # Figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: Non-neural baselines all EM=1.000 ---
    x_labels = []
    for cl in ctx_set:
        if cl >= 1_000_000:
            x_labels.append(f"{cl // 1_000_000}M")
        elif cl >= 1000:
            x_labels.append(f"{cl // 1000}K")
        else:
            x_labels.append(str(cl))

    for key, label, color in zip(baseline_keys, baseline_labels, baseline_colors):
        ems = [avg_by_len[cl][key] for cl in ctx_set]
        ax1.plot(range(len(ctx_set)), ems, "o-", label=label, color=color,
                 markersize=5, linewidth=1.5, alpha=0.8)

    ax1.set_xticks(range(len(ctx_set)))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Exact Match (EM)")
    ax1.set_title("(a) Non-Neural Retrieval Baselines")
    ax1.legend(loc="lower left")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax1.grid(True, alpha=0.2)

    # --- Panel B: Neural encoder + same retriever fails ---
    retnet_results = {r["context_length"]: r["em"] for r in ablation[0]["results"]}
    anamnesis_results = {r["context_length"]: r["em"] for r in ablation[1]["results"]}

    ctx_ablation = [r["context_length"] for r in ablation[0]["results"]]
    x_labels_b = []
    for cl in ctx_ablation:
        x_labels_b.append(f"{cl // 1000}K" if cl >= 1000 else str(cl))

    retnet_ems = [retnet_results[cl] for cl in ctx_ablation]
    anamnesis_ems = [anamnesis_results[cl] for cl in ctx_ablation]

    ax2.plot(range(len(ctx_ablation)), retnet_ems, "s-",
             label="Bare RetNet + Retriever",
             color="#E53935", markersize=6, linewidth=2)
    ax2.plot(range(len(ctx_ablation)), anamnesis_ems, "^-",
             label="Anamnesis + Retriever",
             color="#1565C0", markersize=6, linewidth=2)
    ax2.axhline(y=1.0, color="#4CAF50", linestyle="--", alpha=0.5,
                label="BM25 (reference)")

    ax2.set_xticks(range(len(ctx_ablation)))
    ax2.set_xticklabels(x_labels_b, rotation=45, ha="right")
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Exact Match (EM)")
    ax2.set_title("(b) Neural Encoder + Same Retriever")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    ax2.annotate(
        "Both neural models\nfail (EM = 0.000)\nat all lengths",
        xy=(4, 0.0), xytext=(4, 0.5),
        fontsize=9, ha="center", color="#B71C1C",
        arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                  edgecolor="#B71C1C", alpha=0.8),
    )

    fig.tight_layout()

    out_paths = [
        ROOT / "analysis" / "figures" / "fig_retrieval_baselines.pdf",
        ROOT / "analysis" / "figures" / "fig_retrieval_baselines.png",
        ROOT / "neurocomputing_submission_package" / "figures" / "fig_retrieval_baselines.pdf",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), bbox_inches="tight",
                    dpi=150 if p.suffix == ".png" else 300)
        print(f"Saved: {p}")

    plt.close(fig)

    print(f"\nChunk Discrimination Diagnostic:")
    print(f"  Bare RetNet: acc={ablation[0]['diag_acc']:.3f}  top3={ablation[0]['diag_top3']:.3f}")
    print(f"  Anamnesis:   acc={ablation[1]['diag_acc']:.3f}  top3={ablation[1]['diag_top3']:.3f}")
    print(f"  Random baseline: 0.250")
    print(f"\nConclusion: Both encoders BELOW random -> content-based retrieval impossible")


if __name__ == "__main__":
    main()
