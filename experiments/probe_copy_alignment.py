#!/usr/bin/env python
"""Probe TokenCopyBuffer alignment on Needle-style exact-copy tasks.

This is a debugging probe for the ARC work. It asks whether the bounded copy
memory contains the expected source positions and whether answer positions put
attention mass on the correct copy slot.

Example:
  python experiments/probe_copy_alignment.py --train-steps 50 --eval-lengths 128,256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.train_synthetic import (
    make_needle_batch,
    masked_exact_match,
    masked_lm_loss,
    set_seed,
)
from src.models import AnamnesisConfig, AnamnesisModel


@torch.no_grad()
def copy_alignment_metrics(
    diagnostics: dict[str, torch.Tensor],
    seq_len: int,
    password_len: int,
) -> dict[str, float]:
    weights = diagnostics["token_copy_weights"]
    valid = diagnostics["token_copy_valid"]
    pos_ids = diagnostics["token_copy_pos_ids"]

    masses: list[torch.Tensor] = []
    hits: list[torch.Tensor] = []
    for answer_offset in range(password_len):
        answer_pos = seq_len - password_len + answer_offset
        source_pos = 1 + answer_offset
        slot_match = pos_ids == source_pos
        mass = (weights[:, answer_pos, :] * slot_match.float()).sum(dim=-1)
        masses.append(mass)
        hits.append(slot_match.any(dim=-1).float())

    stacked_mass = torch.stack(masses, dim=1)
    stacked_hits = torch.stack(hits, dim=1)
    answer_weights = weights[:, seq_len - password_len : seq_len, :]
    entropy = -(answer_weights.clamp_min(1e-9).log() * answer_weights).sum(dim=-1)

    return {
        "copy_correct_slot_mass": float(stacked_mass.mean().cpu()),
        "copy_source_slot_hit": float(stacked_hits.mean().cpu()),
        "copy_answer_entropy": float(entropy.mean().cpu()),
        "copy_valid_count": float(valid.sum(dim=-1).float().mean().cpu()),
    }


def build_model(args: argparse.Namespace, device: torch.device) -> AnamnesisModel:
    config = AnamnesisConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff or args.d_model * 4,
        max_seq_len=max(args.train_seq_len, max(args.eval_lengths)) * 2,
        dropout=0.0,
        engram_layers=(),
        milestone_token_ids=(2,),
        use_token_copy_buffer=True,
        max_milestone_snapshots=args.max_snapshots,
        position_encoding_type="sinusoidal",
    )
    return AnamnesisModel(config).to(device)


def train(args: argparse.Namespace, model: AnamnesisModel, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for step in range(1, args.train_steps + 1):
        batch = make_needle_batch(
            args.batch_size,
            args.train_seq_len,
            args.vocab_size,
            device,
            password_len=args.password_len,
        )
        logits = model(batch.input_ids)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_interval == 0 or step == args.train_steps:
            print(f"train step={step:4d} loss={loss.item():.4f}", flush=True)


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: AnamnesisModel, device: torch.device) -> None:
    print(
        f"{'seq_len':>8s} {'em':>6s} {'loss':>8s} {'slot_hit':>9s} "
        f"{'slot_mass':>10s} {'entropy':>8s} {'valid':>6s} {'time_s':>7s}",
        flush=True,
    )
    print("-" * 74, flush=True)
    model.eval()
    for seq_len in args.eval_lengths:
        losses: list[float] = []
        exact_matches: list[float] = []
        alignments: list[dict[str, float]] = []
        t0 = time.time()
        set_seed(args.eval_seed + seq_len)
        for _ in range(args.eval_batches):
            batch = make_needle_batch(
                args.batch_size,
                seq_len,
                args.vocab_size,
                device,
                password_len=args.password_len,
            )
            logits, _, diagnostics = model(
                batch.input_ids,
                return_metrics=True,
                return_diagnostics=True,
            )
            losses.append(float(masked_lm_loss(logits, batch.target_ids, batch.loss_mask).cpu()))
            exact_matches.append(masked_exact_match(logits, batch.target_ids, batch.loss_mask))
            alignments.append(copy_alignment_metrics(diagnostics, seq_len, args.password_len))

        avg = {
            key: sum(row[key] for row in alignments) / len(alignments)
            for key in alignments[0]
        }
        print(
            f"{seq_len:8d} {sum(exact_matches)/len(exact_matches):6.3f} "
            f"{sum(losses)/len(losses):8.4f} {avg['copy_source_slot_hit']:9.3f} "
            f"{avg['copy_correct_slot_mass']:10.3f} {avg['copy_answer_entropy']:8.3f} "
            f"{avg['copy_valid_count']:6.2f} {time.time()-t0:7.1f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seq-len", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--eval-lengths", default="128,256,512")
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--password-len", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.eval_lengths = [int(item) for item in args.eval_lengths.split(",")]
    return args


def main() -> None:
    args = parse_args()
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)
    model = build_model(args, device)
    print(f"Device: {device}", flush=True)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    train(args, model, device)
    evaluate(args, model, device)


if __name__ == "__main__":
    main()

