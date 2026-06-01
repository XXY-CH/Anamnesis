"""Benchmark inference throughput and memory across seq_len sweep.

Measures real tok/s and peak memory for Transformer, RetNet, and Anamnesis
at varying sequence lengths to demonstrate O(1) inference scaling.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anamnesis import AnamnesisModel
from src.models.transformer_baseline import TransformerConfig, TransformerLM


def measure_inference(model, x, device, n_warmup=3, n_measured=10):
    """Measure inference throughput (tok/s) and peak memory."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(x)
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

        # Reset memory tracking
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        # Measured runs
        t0 = time.perf_counter()
        for _ in range(n_measured):
            _ = model(x)
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    total_tokens = x.shape[0] * x.shape[1] * n_measured
    tok_per_sec = total_tokens / elapsed

    # Peak memory
    peak_mb = 0.0
    if device.startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    elif hasattr(torch, "mps") and device == "mps":
        # MPS doesn't have direct peak memory API, use process RSS
        try:
            import resource
            peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: bytes→KB→MB
        except Exception:
            peak_mb = -1

    return tok_per_sec, peak_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=67)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--output", type=str, default="experiments/results/real/seq_len_sweep.csv")
    args = p.parse_args()

    seq_lens = [128, 256, 512, 1024, 2048]
    models_config = [
        ("transformer", False, False),
        ("retnet", True, False),
        ("anamnesis", True, True),
    ]

    results = []
    device = args.device

    for model_name, layerwise, engram in models_config:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} (d={args.d_model}, h={args.n_heads}, L={args.n_layers})")
        print(f"{'='*60}")

        if model_name == "transformer":
            tf_config = TransformerConfig(
                vocab_size=args.vocab_size, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers,
            )
            model = TransformerLM(tf_config).to(device)
        else:
            from src.models.anamnesis import AnamnesisConfig
            ana_config = AnamnesisConfig(
                vocab_size=args.vocab_size, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers,
                d_ff=args.d_model * 4,
                max_seq_len=max(seq_lens),
                dropout=0.0,
                layerwise_gamma=layerwise,
                engram_layers=(2,) if engram else (),
                engram_num_slots=8192 if engram else 0,
                engram_use_conv=True,
            )
            model = AnamnesisModel(ana_config).to(device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Params: {n_params:.1f}M")

        for seq_len in seq_lens:
            print(f"  seq_len={seq_len:>5d} ... ", end="", flush=True)
            try:
                x = torch.randint(0, args.vocab_size, (args.batch_size, seq_len), device=device)
                tok_s, mem_mb = measure_inference(model, x, device)
                print(f"tok/s={tok_s:>10,.0f}  mem={mem_mb:>8.1f}MB")
                results.append({
                    "model": model_name,
                    "d_model": args.d_model,
                    "n_params_m": n_params,
                    "seq_len": seq_len,
                    "tokens_per_sec": tok_s,
                    "peak_memory_mb": mem_mb,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "model": model_name,
                    "d_model": args.d_model,
                    "n_params_m": n_params,
                    "seq_len": seq_len,
                    "tokens_per_sec": 0,
                    "peak_memory_mb": 0,
                })

        del model
        if hasattr(torch, "mps") and device == "mps":
            torch.mps.empty_cache()

    # Save CSV
    import csv
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
