#!/usr/bin/env python
"""Evaluate recurrent inference at increasing context lengths.

Trains at seq_len=1024 with sinusoidal PE, then measures recurrent eval_em
at progressively longer sequences. This establishes the baseline degradation
curve before we build the Context Compiler.

Usage:
  python experiments/eval_context_scaling.py
  python experiments/eval_context_scaling.py --train-seq-len 512 --max-eval-len 65536
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
import torch.nn.functional as F

from src.models import AnamnesisConfig, AnamnesisModel
from src.models.recurrent_state import RecurrentState

from experiments.train_synthetic import (
    make_needle_batch,
    masked_exact_match,
    masked_lm_loss,
    set_seed,
    SyntheticBatch,
)


def make_needle_at_length(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    password_len: int = 3,
) -> SyntheticBatch:
    return make_needle_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
        password_len=password_len,
    )


@torch.no_grad()
def recurrent_eval_at_length(
    model: AnamnesisModel,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    n_batches: int = 8,
    seed: int = 20000,
) -> dict[str, float]:
    """Recurrent evaluation at a specific sequence length."""
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()

    try:
        set_seed(seed)
        model.eval()
        losses: list[float] = []
        exact_matches: list[float] = []

        for b in range(n_batches):
            batch = make_needle_at_length(
                batch_size, seq_len, vocab_size, device
            )
            state = model.init_recurrent_state(batch_size, device=device)
            all_logits: list[torch.Tensor] = []

            for t in range(seq_len):
                step_logits, state = model.forward_recurrent_step(
                    batch.input_ids[:, t], state
                )
                all_logits.append(step_logits)

            logits = torch.stack(all_logits, dim=1)
            loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
            losses.append(float(loss.detach().cpu()))
            exact_matches.append(
                masked_exact_match(logits, batch.target_ids, batch.loss_mask)
            )

        return {
            "seq_len": seq_len,
            "eval_loss": sum(losses) / len(losses),
            "eval_em": sum(exact_matches) / len(exact_matches),
            "n_batches": n_batches,
        }
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)


def build_config(args: argparse.Namespace) -> AnamnesisConfig:
    return AnamnesisConfig(
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
        engram_table_device=None,
        attnres_every=args.attnres_every,
        branch_init_scale=args.branch_init_scale,
        milestone_token_ids=(2,) if args.use_milestones else (),
        milestone_ttl=args.milestone_ttl,
        milestone_gamma=args.milestone_gamma,
        use_milestone_snapshots=args.use_snapshots,
        max_milestone_snapshots=args.max_snapshots,
        use_token_copy_buffer=args.use_token_copy_buffer,
        position_encoding_type="sinusoidal",
    )


def train(
    args: argparse.Namespace,
    model: AnamnesisModel,
    device: torch.device,
) -> None:
    """Train at the base seq_len for the specified number of steps."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model.train()

    for step in range(1, args.train_steps + 1):
        batch = make_needle_at_length(
            args.batch_size, args.train_seq_len, args.vocab_size, device
        )
        logits = model(batch.input_ids)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            print(
                f"  train step={step:4d} loss={loss.item():.4f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--eval-lengths", type=str,
                        default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--max-eval-len", type=int, default=None,
                        help="Override eval-lengths with powers of 2 up to this value.")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=100)
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
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)

    if args.max_eval_len is not None:
        lengths = []
        l = args.train_seq_len
        while l <= args.max_eval_len:
            lengths.append(l)
            l *= 2
        args.eval_lengths = ",".join(str(l) for l in lengths)
    eval_lengths = [int(x) for x in args.eval_lengths.split(",")]

    set_seed(args.seed)

    print(f"=== Context Scaling Experiment ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Train: seq_len={args.train_seq_len}, steps={args.train_steps}", flush=True)
    print(f"Eval lengths: {eval_lengths}", flush=True)
    print(f"Config: d_model={args.d_model}, n_layers={args.n_layers}", flush=True)
    print(flush=True)

    config = build_config(args)
    model = AnamnesisModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)
    print(flush=True)

    # --- Train ---
    print("--- Training ---", flush=True)
    t0 = time.time()
    train(args, model, device)
    print(f"Training done in {time.time() - t0:.1f}s", flush=True)
    print(flush=True)

    # --- Evaluate at each length ---
    print("--- Recurrent Eval Scaling ---", flush=True)
    print(f"{'seq_len':>8s} {'eval_em':>8s} {'eval_loss':>10s} {'time_s':>8s}", flush=True)
    print("-" * 40, flush=True)

    results: list[dict[str, float]] = []
    for seq_len in eval_lengths:
        t0 = time.time()
        result = recurrent_eval_at_length(
            model=model,
            seq_len=seq_len,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            device=device,
            n_batches=args.eval_batches,
            seed=20000 + seq_len,
        )
        elapsed = time.time() - t0
        result["time_s"] = elapsed
        results.append(result)

        print(
            f"{result['seq_len']:8d} {result['eval_em']:8.3f} "
            f"{result['eval_loss']:10.4f} {elapsed:8.1f}",
            flush=True,
        )

    print(flush=True)
    print("=== Summary ===", flush=True)
    for r in results:
        print(
            f"  seq_len={r['seq_len']:6d}  eval_em={r['eval_em']:.3f}  "
            f"loss={r['eval_loss']:.4f}  time={r['time_s']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
