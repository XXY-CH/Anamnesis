#!/usr/bin/env python
"""Test Context Compiler as long-context extension.

Two-phase experiment:
1. Train Small Reasoner on needle@512 (short context)
2. Freeze model, fine-tune MemoryQueryHead on needle@4096 via Context Compiler
3. Compare: model alone@512 vs model+compiler@4096

Also tests oracle compiler (always selects password positions) as upper bound.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.memory import ContextCompiler, CompiledMemory, MemoryQueryHead
from src.models import AnamnesisModel, AnamnesisConfig
from experiments.train_synthetic import (
    make_needle_batch,
    masked_lm_loss,
    masked_exact_match,
    set_seed,
)

START = 1
MARK_THOUGHT = 2
QUERY = 3


def train_short_context(model, device, steps=400, seq_len=512, batch_size=16, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_needle_batch(batch_size, seq_len, vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.input_ids, return_metrics=True)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                em = masked_exact_match(logits, batch.target_ids, batch.loss_mask)
            print(f"  train step={step:4d} loss={loss.item():.4f} em={em:.3f}")


@torch.no_grad()
def oracle_compile(model, input_ids, password_len=3, chunk_size=512):
    """Oracle compiler: always select the password positions. Processes in chunks."""
    seq_len = input_ids.shape[1]
    all_hidden = []
    all_positions = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_hidden = model(input_ids[:, start:end], return_hidden_only=True)
        all_hidden.append(chunk_hidden.squeeze(0))
        all_positions.append(torch.arange(start, end, device=input_ids.device))

    hidden = torch.cat(all_hidden, dim=0)  # [N, d]
    positions = torch.cat(all_positions, dim=0)

    # Password is at positions 1..password_len, QUERY is near the end
    query_pos = (input_ids == QUERY).nonzero(as_tuple=False)[0, 1].item()
    key_positions = list(range(1, password_len + 1)) + [password_len + 1, query_pos]

    selected_hidden = hidden[key_positions]
    selected_positions = positions[key_positions]

    keys = F.normalize(selected_hidden, dim=-1)
    return CompiledMemory(keys=keys, values=selected_hidden, positions=selected_positions)


def forward_with_memory(model, query_head, input_ids, memory):
    """Forward pass: model logits + memory query logits."""
    logits, _ = model(input_ids, return_metrics=True)
    hidden = model(input_ids, return_hidden_only=True)

    memory_out, _ = query_head(hidden, memory)
    memory_logits = F.linear(memory_out, model.token_embedding.weight)

    alpha = query_head.residual_scale.abs()
    return logits + alpha * memory_logits


def fine_tune_query_head(model, compiler, query_head, device, steps=200,
                         seq_len=4096, batch_size=1, lr=1e-3, password_len=3):
    """Fine-tune only the query head on long sequences. Model stays frozen."""
    model.eval()
    query_head.train()

    optimizer = torch.optim.AdamW(query_head.parameters(), lr=lr)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_needle_batch(batch_size, seq_len, vocab_size, device)
        input_ids = batch.input_ids
        targets = batch.target_ids
        mask = batch.loss_mask

        # Find answer start position
        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            continue
        answer_start = query_pos[0, 1].item() + 1

        # Compile context before answer
        context_ids = input_ids[:, :answer_start]
        compiled = compiler.compile(model, context_ids)

        # Run model on last chunk (answer region)
        max_chunk = model.config.max_seq_len
        chunk_start = max(0, answer_start - max_chunk + 64)
        chunk_ids = input_ids[:, chunk_start:]
        chunk_targets = targets[:, chunk_start:]
        chunk_mask = mask[:, chunk_start:]

        combined_logits = forward_with_memory(model, query_head, chunk_ids, compiled)
        loss = masked_lm_loss(combined_logits, chunk_targets, chunk_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query_head.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                em = masked_exact_match(combined_logits, chunk_targets, chunk_mask)
            print(f"  finetune step={step:4d} loss={loss.item():.4f} em={em:.3f}")


@torch.no_grad()
def evaluate(model, query_head, compiler, device, seq_len, eval_batches=16,
             max_chunk=512, password_len=3, use_oracle=False, model_only=False):
    """Evaluate with Context Compiler. model_only skips memory entirely."""
    model.eval()
    if query_head is not None:
        query_head.eval()
    vocab_size = model.config.vocab_size
    exact_matches = []

    for _ in range(eval_batches):
        batch = make_needle_batch(1, seq_len, vocab_size, device)
        input_ids = batch.input_ids
        targets = batch.target_ids
        mask = batch.loss_mask

        if model_only:
            # Model only — must fit within max_seq_len
            if seq_len > model.config.max_seq_len:
                exact_matches.append(0.0)
                continue
            logits, _ = model(input_ids, return_metrics=True)
            em = masked_exact_match(logits, targets, mask)
            exact_matches.append(em)
            continue

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            exact_matches.append(0.0)
            continue
        answer_start = query_pos[0, 1].item() + 1

        if use_oracle:
            compiled = oracle_compile(model, input_ids, password_len=password_len)
        else:
            context_ids = input_ids[:, :answer_start]
            compiled = compiler.compile(model, context_ids)

        chunk_start = max(0, answer_start - max_chunk + 64)
        chunk_ids = input_ids[:, chunk_start:]
        chunk_targets = targets[:, chunk_start:]
        chunk_mask = mask[:, chunk_start:]

        combined_logits = forward_with_memory(model, query_head, chunk_ids, compiled)
        em = masked_exact_match(combined_logits, chunk_targets, chunk_mask)
        exact_matches.append(em)

    avg_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    return avg_em


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--finetune-steps", type=int, default=200)
    parser.add_argument("--train-seq-len", type=int, default=512)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)

    set_seed(args.seed)

    config = AnamnesisConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.train_seq_len,
        engram_layers=(),
        use_token_copy_buffer=True,
        milestone_token_ids=(MARK_THOUGHT,),
    )
    model = AnamnesisModel(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Train on short context
    print(f"\n=== Phase 1: Train on needle@{args.train_seq_len} ===")
    train_short_context(model, device, steps=args.train_steps, seq_len=args.train_seq_len)

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Build compiler and query head
    compiler = ContextCompiler(
        d_model=args.d_model,
        memory_size=args.memory_size,
        chunk_size=512,
    ).to(device)

    query_head = MemoryQueryHead(
        d_model=args.d_model,
        n_heads=4,
    ).to(device)

    print(f"QueryHead params: {sum(p.numel() for p in query_head.parameters()):,}")

    # Phase 2: Test zero-shot at training length
    print(f"\n=== Phase 2: Zero-shot test at {args.train_seq_len} ===")
    em_model = evaluate(model, None, compiler, device,
                        args.train_seq_len, model_only=True)
    print(f"  Model only (no memory):    EM = {em_model:.3f}")

    # Phase 3: Fine-tune query head on long sequences
    print(f"\n=== Phase 3: Fine-tune query head on needle@{args.eval_seq_len} ===")
    fine_tune_query_head(model, compiler, query_head, device,
                         steps=args.finetune_steps, seq_len=args.eval_seq_len)

    # Phase 4: Final evaluation
    print(f"\n=== Phase 4: Final evaluation ===")
    print(f"{'Length':>8s} {'Model Only':>11s} {'Compiler':>10s} {'Oracle':>10s}")
    print("-" * 42)

    for length in [args.train_seq_len, args.eval_seq_len]:
        em_base = evaluate(model, None, compiler, device, length, model_only=True)
        em_compiled = evaluate(model, query_head, compiler, device, length, use_oracle=False)
        em_oracle = evaluate(model, query_head, compiler, device, length, use_oracle=True)

        base_str = f"{em_base:>11.3f}" if em_base >= 0 else "        N/A"
        print(f"{length:>8d} {base_str} {em_compiled:>10.3f} {em_oracle:>10.3f}")


if __name__ == "__main__":
    main()
