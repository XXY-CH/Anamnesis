"""Benchmark recurrent (O(1)) vs parallel inference for RetNet/Anamnesis.

Demonstrates that recurrent mode has constant-time per-token inference,
while Transformer's parallel mode scales quadratically with seq_len.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anamnesis import AnamnesisConfig, AnamnesisModel
from src.models.transformer_baseline import TransformerConfig, TransformerLM


def benchmark_parallel(model, seq_len, vocab_size, device, n_warmup=2, n_measured=5):
    """Benchmark parallel forward pass."""
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_measured):
            _ = model(x)
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
    total_tokens = seq_len * n_measured
    return total_tokens / elapsed


def benchmark_recurrent(model, seq_len, vocab_size, device, n_warmup=1, n_measured=3):
    """Benchmark recurrent (one-token-at-a-time) inference."""
    model.eval()
    bos = torch.randint(0, vocab_size, (1,), device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            state = model.init_recurrent_state(1, torch.device(device))
            x_warm = torch.randint(0, vocab_size, (64,), device=device)
            for t in range(64):
                _, state = model.forward_recurrent_step(x_warm[t:t+1], state)
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_measured):
            state = model.init_recurrent_state(1, torch.device(device))
            x = torch.randint(0, vocab_size, (seq_len,), device=device)
            for t in range(seq_len):
                _, state = model.forward_recurrent_step(x[t:t+1], state)
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

    total_tokens = seq_len * n_measured
    return total_tokens / elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=67)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--output", type=str, default="experiments/results/real/recurrent_benchmark.csv")
    args = p.parse_args()

    import csv

    seq_lens = [128, 256, 512, 1024, 2048]
    results = []

    # Anamnesis (with recurrent support)
    print("=" * 60)
    print("Anamnesis (8h+lw+Engram)")
    print("=" * 60)
    config = AnamnesisConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 4, max_seq_len=max(seq_lens),
        layerwise_gamma=True, engram_layers=(2,),
        engram_num_slots=8192, engram_use_conv=True,
    )
    model = AnamnesisModel(config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n_params:.1f}M")

    for sl in seq_lens:
        print(f"  seq_len={sl:>5d} parallel: ", end="", flush=True)
        try:
            tok_s = benchmark_parallel(model, sl, args.vocab_size, args.device)
            print(f"{tok_s:>8,.0f} tok/s  recurrent: ", end="", flush=True)
            rec_tok_s = benchmark_recurrent(model, sl, args.vocab_size, args.device)
            print(f"{rec_tok_s:>8,.0f} tok/s")
            results.append({
                "model": "anamnesis", "d_model": args.d_model,
                "n_params_m": n_params, "seq_len": sl,
                "parallel_tok_s": tok_s, "recurrent_tok_s": rec_tok_s,
                "mode": "both",
            })
        except Exception as e:
            print(f"FAILED: {e}")

    del model
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    # Bare RetNet (no Engram, recurrent supported)
    print("\n" + "=" * 60)
    print("Bare RetNet (8h+lw, no Engram)")
    print("=" * 60)
    config2 = AnamnesisConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 4, max_seq_len=max(seq_lens),
        layerwise_gamma=True, engram_layers=(),
    )
    model2 = AnamnesisModel(config2).to(args.device)
    n_params2 = sum(p.numel() for p in model2.parameters()) / 1e6
    print(f"Params: {n_params2:.1f}M")

    for sl in seq_lens:
        print(f"  seq_len={sl:>5d} parallel: ", end="", flush=True)
        try:
            tok_s = benchmark_parallel(model2, sl, args.vocab_size, args.device)
            print(f"{tok_s:>8,.0f} tok/s  recurrent: ", end="", flush=True)
            rec_tok_s = benchmark_recurrent(model2, sl, args.vocab_size, args.device)
            print(f"{rec_tok_s:>8,.0f} tok/s")
            results.append({
                "model": "retnet_bare", "d_model": args.d_model,
                "n_params_m": n_params2, "seq_len": sl,
                "parallel_tok_s": tok_s, "recurrent_tok_s": rec_tok_s,
                "mode": "both",
            })
        except Exception as e:
            print(f"FAILED: {e}")

    del model2
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    # Transformer (parallel only)
    print("\n" + "=" * 60)
    print("Transformer")
    print("=" * 60)
    tf_config = TransformerConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
    )
    tf_model = TransformerLM(tf_config).to(args.device)
    n_params3 = sum(p.numel() for p in tf_model.parameters()) / 1e6
    print(f"Params: {n_params3:.1f}M")

    for sl in seq_lens:
        print(f"  seq_len={sl:>5d} parallel: ", end="", flush=True)
        try:
            tok_s = benchmark_parallel(tf_model, sl, args.vocab_size, args.device)
            print(f"{tok_s:>8,.0f} tok/s")
            results.append({
                "model": "transformer", "d_model": args.d_model,
                "n_params_m": n_params3, "seq_len": sl,
                "parallel_tok_s": tok_s, "recurrent_tok_s": 0,
                "mode": "parallel_only",
            })
        except Exception as e:
            print(f"FAILED: {e}")

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
