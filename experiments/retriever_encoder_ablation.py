#!/usr/bin/env python3
"""P0: RetNet + same retriever baseline for chunk retrieval.

Tests whether retrieval EM=1.000 comes from Engram-enhanced embeddings
or from the retriever pipeline alone.

Runs the full pipeline (train model → freeze → diagnostic → train retriever → evaluate)
with bare RetNet (no Engram) and Anamnesis side by side.

Output: experiments/results/retriever_encoder_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.memory.chunk_retriever import ChunkRetriever
from src.models import AnamnesisConfig, AnamnesisModel
from experiments.test_chunk_retriever import (
    MARK_THOUGHT,
    diagnostic_chunk_discrimination,
    evaluate,
    train_chunk_retriever,
    train_model,
)
from experiments.train_synthetic import set_seed

ENGRAM_LAYERS = (2, 5)  # Default Anamnesis Engram layers

EVAL_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def run_variant(
    name: str,
    engram_layers: tuple[int, ...],
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Run full pipeline for one model variant."""
    print(f"\n{'=' * 60}")
    print(f"  VARIANT: {name}")
    print(f"{'=' * 60}")

    set_seed(seed)

    config = AnamnesisConfig(
        vocab_size=192,
        d_model=64,
        n_heads=4,
        n_layers=8,
        max_seq_len=512,
        engram_layers=engram_layers,
        use_token_copy_buffer=True,
        milestone_token_ids=(MARK_THOUGHT,),
        token_copy_sinusoidal_pos=True,
        position_encoding_type="sinusoidal",
    )
    model = AnamnesisModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  engram_layers={engram_layers}")

    # Phase 1: Train on needle@512 (fixed-needle, matching original pipeline)
    print(f"\n  Phase 1: Train on fixed-needle@512 ({name})")
    train_model(
        model, device,
        steps=800, seq_len=512, use_random_needle=False, lr=1e-3,
    )

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Phase 2: Chunk discrimination diagnostic
    print(f"\n  Phase 2: Chunk discrimination diagnostic ({name})")
    diag_acc, diag_top3 = diagnostic_chunk_discrimination(
        model, device, seq_len=2048, chunk_size=512,
    )

    # Phase 3: Train retriever
    print(f"\n  Phase 3: Train retriever on random-needle@2048 ({name})")
    retriever = ChunkRetriever(d_model=64).to(device)
    train_chunk_retriever(
        model, retriever, device,
        steps=200, seq_len=2048, chunk_size=512, contrastive=True,
    )

    # Phase 4: Evaluate at increasing context lengths
    results = []
    print(f"\n  Phase 4: Evaluation ({name})")
    for ctx_len in EVAL_CONTEXT_LENGTHS:
        try:
            em = evaluate(
                model, retriever, device, ctx_len,
                mode="retriever", use_random_needle=True,
            )
            n_chunks = ctx_len // 512
            print(f"    {ctx_len:>10,} tok ({n_chunks:>5} chunks): EM={em:.3f}")
            results.append({"context_length": ctx_len, "n_chunks": n_chunks, "em": round(em, 4)})
        except Exception as e:
            print(f"    {ctx_len:>10,} tok: FAILED ({e})")
            results.append({
                "context_length": ctx_len,
                "n_chunks": ctx_len // 512,
                "em": -1,
                "error": str(e),
            })

    return {
        "variant": name,
        "engram_layers": list(engram_layers),
        "params": n_params,
        "diag_acc": round(diag_acc, 4),
        "diag_top3": round(diag_top3, 4),
        "results": results,
    }


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    all_results = []

    # Variant 1: Bare RetNet (no Engram) — the baseline question
    r1 = run_variant("bare_retnet", engram_layers=(), device=device, seed=42)
    all_results.append(r1)

    # Variant 2: Anamnesis (with Engram) — reference
    r2 = run_variant("anamnesis", engram_layers=ENGRAM_LAYERS, device=device, seed=42)
    all_results.append(r2)

    # Save results
    out = Path(ROOT) / "experiments" / "results" / "retriever_encoder_ablation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY: Encoder Ablation (RetNet vs Anamnesis)")
    print(f"{'=' * 60}")
    print(f"  {'Context':>10} | {'Bare RetNet':>12} | {'Anamnesis':>12} | Delta")
    print(f"  {'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 7}")
    for item1, item2 in zip(r1["results"], r2["results"]):
        ctx = item1["context_length"]
        em1 = item1.get("em", -1)
        em2 = item2.get("em", -1)
        em1_str = f"{em1:.3f}" if em1 >= 0 else "FAIL"
        em2_str = f"{em2:.3f}" if em2 >= 0 else "FAIL"
        if em1 >= 0 and em2 >= 0:
            delta = em2 - em1
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "N/A"
        print(f"  {ctx:>10,} | {em1_str:>12} | {em2_str:>12} | {delta_str}")

    print(f"\n  Diagnostics:")
    print(f"    Bare RetNet discrimination: acc={r1['diag_acc']:.3f} top3={r1['diag_top3']:.3f}")
    print(f"    Anamnesis  discrimination: acc={r2['diag_acc']:.3f} top3={r2['diag_top3']:.3f}")


if __name__ == "__main__":
    main()
