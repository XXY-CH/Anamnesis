#!/usr/bin/env python
"""Curriculum experiments with aggressive hyperparams for length generalization."""

from __future__ import annotations

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


def run_curriculum(
    phases: list[tuple[int, int, int]],  # (seq_len, steps, batch_size)
    ttl: int,
    gamma: float,
    label: str,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(42)

    max_len = max(p[0] for p in phases) * 4
    config = AnamnesisConfig(
        vocab_size=192, d_model=64, n_heads=4, n_layers=8,
        max_seq_len=max_len, dropout=0.0,
        engram_layers=(2,), engram_num_slots=8192,
        engram_max_ngram=3, engram_hash_heads=4,
        attnres_every=4, branch_init_scale=1e-4,
        milestone_token_ids=(2,), milestone_ttl=ttl,
        milestone_gamma=gamma, use_milestone_snapshots=True,
        max_milestone_snapshots=8, use_token_copy_buffer=True,
        position_encoding_type="sinusoidal",
    )
    model = AnamnesisModel(config).to(device)
    print(f"\n=== {label} ===", flush=True)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()

    global_step = 0
    for phase_idx, (seq_len, steps, bs) in enumerate(phases):
        phase_label = f"Phase {phase_idx + 1}: seq_len={seq_len}, {steps} steps, bs={bs}"
        print(phase_label, flush=True)
        for step in range(1, steps + 1):
            global_step += 1
            batch = make_needle_batch(bs, seq_len, 192, device)
            logits = model(batch.input_ids)
            loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            log_every = max(steps // 6, 1)
            if step % log_every == 0:
                print(f"  step={global_step} loss={loss.item():.4f}", flush=True)

    # Eval
    print(f"--- Eval ({label}) ---", flush=True)
    model.eval()
    for sl in [1024, 2048, 4096, 8192]:
        set_seed(20000 + sl)
        ems: list[float] = []
        t0 = time.time()
        for _ in range(8):
            batch = make_needle_batch(4, sl, 192, device)
            logits = model.forward_chunked(batch.input_ids, chunk_size=512)
            ems.append(masked_exact_match(logits, batch.target_ids, batch.loss_mask))
        print(
            f"  seq_len={sl:5d} eval_em={sum(ems)/len(ems):.3f} time={time.time()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    # Experiment 1: 2-phase curriculum, aggressive decay protection
    run_curriculum(
        phases=[(1024, 300, 8), (2048, 200, 4)],
        ttl=4096, gamma=0.99999,
        label="2-phase curriculum, ttl=4096, g=0.99999",
    )

    # Experiment 2: 2-phase curriculum, wider TTL
    run_curriculum(
        phases=[(1024, 300, 8), (2048, 200, 4)],
        ttl=8192, gamma=0.9999,
        label="2-phase curriculum, ttl=8192, g=0.9999",
    )
