"""VRAM and latency profiling for paper figures."""

import argparse
import time
import torch
from src.models.anamnesis import AnamnesisModel, AnamnesisConfig
from src.models.transformer_baseline import TransformerLM, TransformerConfig
from src.models.linear_attention import LinearAttentionModel


def profile_model(model, vocab_size, d_model, seq_len, batch_size, device, label):
    """Profile VRAM and latency for a single model."""
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

    model = model.to(device)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            out = model(x, return_metrics=False)
            logits = out[0] if isinstance(out, tuple) else out
    if device.type == 'mps':
        torch.mps.synchronize()

    # Measure latency
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            out = model(x, return_metrics=False)
            logits = out[0] if isinstance(out, tuple) else out
            if device.type == 'mps':
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)

    latency_ms = (sum(times) / len(times)) * 1000

    # Memory: MPS doesn't have cuda.memory_allocated, use param count as proxy
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"{label:30s} | params={n_params/1e6:.1f}M | param_mem={param_mb:.1f}MB | "
          f"latency={latency_ms:.1f}ms | tok/s={batch_size*seq_len/(latency_ms/1000):.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    vocab_size = 94
    d_models = [128, 256]
    seq_lens = [128, 512, 1024, 2048]

    for d in d_models:
        print(f"\n{'='*80}")
        print(f"d_model={d}")
        print(f"{'='*80}")

        # Build models
        ana_cfg = AnamnesisConfig(
            vocab_size=vocab_size, d_model=d, n_heads=8, n_layers=8,
            d_ff=d*4, max_seq_len=2048, dropout=0.0,
            position_encoding_type="sinusoidal", layerwise_gamma=True,
            engram_layers=(2,),
            engram_num_slots=8192, engram_max_ngram=3, engram_hash_heads=4,
        )
        tf_cfg = TransformerConfig(
            vocab_size=vocab_size, d_model=d, n_heads=8, n_layers=8,
            d_ff=d*4, max_seq_len=2048, dropout=0.0,
        )

        for seq_len in seq_lens:
            batch = 1
            print(f"\n--- seq_len={seq_len}, batch={batch} ---")
            profile_model(
                AnamnesisModel(ana_cfg), vocab_size, d, seq_len, batch, device,
                f"Anamnesis d={d}"
            )
            profile_model(
                TransformerLM(tf_cfg), vocab_size, d, seq_len, batch, device,
                f"Transformer d={d}"
            )
            profile_model(
                LinearAttentionModel(vocab_size, d, 8, 8, d*4), vocab_size, d,
                seq_len, batch, device, f"LinearAttn d={d}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
