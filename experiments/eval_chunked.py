#!/usr/bin/env python
"""Chunked Context Compiler evaluation.

Processes long sequences in chunks using chunkwise retention, storing
token embeddings from every chunk, then retrieving using the last
chunk's hidden state (which is in-distribution because the last chunk
is processed in parallel mode).

Usage:
  python experiments/eval_chunked.py --train-seq-len 1024 --eval-lengths 1024,2048,4096,8192
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.models import AnamnesisConfig, AnamnesisModel
from experiments.train_synthetic import (
    make_needle_batch,
    masked_exact_match,
    masked_lm_loss,
    set_seed,
    SyntheticBatch,
)


@torch.no_grad()
def chunked_eval(
    model: AnamnesisModel,
    input_ids: torch.Tensor,
    chunk_size: int = 512,
    max_buffer_slots: int = 256,
) -> torch.Tensor:
    """Evaluate using the model's forward_chunked method."""
    return model.forward_chunked(input_ids, chunk_size=chunk_size)


def train_and_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    eval_lengths = [int(x) for x in args.eval_lengths.split(",")]

    set_seed(args.seed)

    config = AnamnesisConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.train_seq_len * 2,
        dropout=0.0,
        engram_layers=(args.engram_layer,),
        engram_num_slots=args.engram_slots,
        engram_max_ngram=3,
        engram_hash_heads=4,
        attnres_every=args.attnres_every,
        branch_init_scale=args.branch_init_scale,
        milestone_token_ids=(2,) if args.use_milestones else (),
        milestone_ttl=args.milestone_ttl,
        milestone_gamma=args.milestone_gamma,
        use_milestone_snapshots=args.use_snapshots,
        max_milestone_snapshots=args.max_snapshots,
        use_token_copy_buffer=args.use_token_copy_buffer,
        token_copy_sinusoidal_pos=args.tcb_sinusoidal_pos,
        position_encoding_type="sinusoidal",
    )

    model = AnamnesisModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    # --- Train at base seq_len ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    print(f"Training at seq_len={args.train_seq_len} for {args.train_steps} steps", flush=True)
    for step in range(1, args.train_steps + 1):
        if args.mixed_length:
            cur_len = random.randint(args.min_seq_len, args.max_seq_len)
        else:
            cur_len = args.train_seq_len
        batch = make_needle_batch(
            args.batch_size, cur_len, args.vocab_size, device
        )
        logits = model(batch.input_ids)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"  step={step:4d} loss={loss.item():.4f}", flush=True)

    # --- Chunked eval at each length ---
    print(f"\n--- Chunked Context Compiler Eval ---", flush=True)
    print(f"{'seq_len':>8s} {'chunks':>7s} {'buffer':>7s} {'eval_em':>8s} {'time_s':>8s}", flush=True)
    print("-" * 45, flush=True)

    for seq_len in eval_lengths:
        n_chunks = (seq_len + args.chunk_size - 1) // args.chunk_size
        buffer_slots = min(args.max_buffer_slots, seq_len)

        t0 = time.time()
        ems: list[float] = []
        set_seed(20000 + seq_len)
        model.eval()

        for _ in range(args.eval_batches):
            batch = make_needle_batch(
                args.batch_size, seq_len, args.vocab_size, device
            )
            logits = chunked_eval(
                model, batch.input_ids,
                chunk_size=args.chunk_size,
                max_buffer_slots=args.max_buffer_slots,
            )
            ems.append(masked_exact_match(logits, batch.target_ids, batch.loss_mask))

        elapsed = time.time() - t0
        avg_em = sum(ems) / len(ems)
        print(
            f"{seq_len:8d} {n_chunks:7d} {buffer_slots:7d} "
            f"{avg_em:8.3f} {elapsed:8.1f}",
            flush=True,
        )

    # Recurrent baseline skipped — O(S) per token is too slow at S>2048.
    # Use eval_context_scaling.py for recurrent baseline comparisons.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--eval-lengths", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-buffer-slots", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--engram-layer", type=int, default=2)
    parser.add_argument("--engram-slots", type=int, default=8192)
    parser.add_argument("--attnres-every", type=int, default=4)
    parser.add_argument("--branch-init-scale", type=float, default=1e-4)
    parser.add_argument("--use-milestones", action="store_true")
    parser.add_argument("--milestone-ttl", type=int, default=256)
    parser.add_argument("--milestone-gamma", type=float, default=0.999)
    parser.add_argument("--use-snapshots", action="store_true")
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--use-token-copy-buffer", action="store_true")
    parser.add_argument("--tcb-sinusoidal-pos", action="store_true")
    parser.add_argument("--mixed-length", action="store_true")
    parser.add_argument("--min-seq-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    train_and_eval(args)


if __name__ == "__main__":
    main()
