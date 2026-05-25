#!/usr/bin/env python
"""Real-data pipeline transfer: Shakespeare LM + chunk retrieval.

Tests whether a RetNet trained on real language modeling produces chunk
embeddings that support the contrastive retrieval pipeline.

Pipeline:
1. Train bare RetNet on Shakespeare character-level LM (no TCB/milestones)
2. Create needle-in-Shakespeare batches (Shakespeare filler + synthetic needle)
3. Train retriever on chunk embeddings from the LM-trained model
4. Evaluate pipeline at increasing context lengths
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

from src.memory.chunk_retriever import ChunkRetriever, compute_chunk_embeddings
from src.models import AnamnesisConfig, AnamnesisModel
from experiments.train_synthetic import (
    masked_exact_match,
    masked_lm_loss,
    set_seed,
    SyntheticBatch,
)
from experiments.train_real import CharTokenizer, load_dataset

PAD = 0
START = 1
MARK_THOUGHT = 2
QUERY = 3
SEP = 4


def make_shakespeare_needle_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    shakespeare_text: str,
    tokenizer: CharTokenizer,
    password_len: int = 3,
    needle_region: float = 0.8,
) -> SyntheticBatch:
    """Needle-in-a-haystack with Shakespeare filler instead of random tokens."""
    total = seq_len + 1
    tokens = torch.randint(16, vocab_size, (batch_size, total), device=device)
    passwords = torch.randint(5, vocab_size, (batch_size, password_len), device=device)

    for b in range(batch_size):
        start_pos = random.randint(0, max(0, len(shakespeare_text) - total - 1))
        chunk = shakespeare_text[start_pos : start_pos + total]
        for i, c in enumerate(chunk):
            tid = tokenizer.stoi.get(c, None)
            if tid is not None and tid < vocab_size:
                tokens[b, i] = tid

        max_pos = int(seq_len * needle_region) - password_len - 1
        needle_start = random.randint(1, max(1, max_pos))
        tokens[b, 0] = START
        tokens[b, needle_start : needle_start + password_len] = passwords[b]
        tokens[b, needle_start + password_len] = MARK_THOUGHT
        tokens[b, -password_len - 1] = QUERY
        tokens[b, -password_len:] = passwords[b]

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, -password_len:] = True
    return SyntheticBatch(tokens[:, :-1], tokens[:, 1:], mask)


def train_lm(model, text, tokenizer, device, steps=2000, seq_len=512, batch_size=16):
    """Train model on character-level language modeling."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    encoded = tokenizer.encode(text)
    n_tokens = len(encoded)

    for step in range(1, steps + 1):
        indices = torch.randint(0, n_tokens - seq_len - 1, (batch_size,))
        input_ids = torch.stack([
            torch.tensor(encoded[i : i + seq_len], dtype=torch.long) for i in indices
        ]).to(device)
        target_ids = torch.stack([
            torch.tensor(encoded[i + 1 : i + seq_len + 1], dtype=torch.long) for i in indices
        ]).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, metrics = model(input_ids, return_metrics=True)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=tokenizer.pad_id,
        )
        if isinstance(metrics, dict) and metrics.get("gate_loss") is not None:
            loss = loss + 0.5 * metrics["gate_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step == steps:
            print(f"  step={step:4d} loss={loss.item():.4f}")

    # Validation perplexity
    val_start = n_tokens // 2
    val_data = encoded[val_start : val_start + 10000]
    val_losses = []
    with torch.no_grad():
        for i in range(0, len(val_data) - seq_len, seq_len):
            ids = torch.tensor([val_data[i : i + seq_len]], dtype=torch.long).to(device)
            tgt = torch.tensor([val_data[i + 1 : i + seq_len + 1]], dtype=torch.long).to(device)
            logits, _ = model(ids, return_metrics=True)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1),
                ignore_index=tokenizer.pad_id,
            )
            val_losses.append(loss.item())
    avg_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"  val_ppl={ppl:.2f}")
    return ppl


def train_on_needle(model, text, tokenizer, device, steps=1200, seq_len=512, batch_size=16):
    """Fine-tune model on needle-in-Shakespeare task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_shakespeare_needle_batch(
            batch_size, seq_len, vocab_size, device, text, tokenizer,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, metrics = model(batch.input_ids, return_metrics=True)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        if isinstance(metrics, dict):
            if metrics.get("gate_loss") is not None:
                loss = loss + 0.5 * metrics["gate_loss"]
            if metrics.get("engram_tcb_distill_loss") is not None:
                loss = loss + 0.5 * metrics["engram_tcb_distill_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == steps:
            with torch.no_grad():
                em = masked_exact_match(logits, batch.target_ids, batch.loss_mask)
            print(f"  step={step:4d} loss={loss.item():.4f} em={em:.3f}")


def train_retriever(model, retriever, text, tokenizer, device, steps=500, seq_len=8192, chunk_size=512):
    """Train retriever with contrastive loss on Shakespeare-filled batches."""
    model.eval()
    retriever.train()
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=3e-3)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_shakespeare_needle_batch(
            1, seq_len, vocab_size, device, text, tokenizer,
        )
        input_ids = batch.input_ids

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            continue
        mark_positions = (input_ids[0] == MARK_THOUGHT).nonzero(as_tuple=False)
        if len(mark_positions) == 0:
            continue
        target_chunk = mark_positions[0, 0].item() // chunk_size

        with torch.no_grad():
            chunk_embs, _ = compute_chunk_embeddings(model, input_ids, chunk_size)
            qp = query_pos[0, 1].item()
            query_chunk_idx = qp // chunk_size
            qs = query_chunk_idx * chunk_size
            qe = min(qs + chunk_size, input_ids.shape[1])
            query_hidden = model(input_ids[:, qs:qe], return_hidden_only=True)
            query_emb = query_hidden[:, qp - qs : qp - qs + 1, :].mean(dim=1)

        q = retriever.query_proj(query_emb)
        c = retriever.chunk_proj(chunk_embs)
        scores = torch.einsum("bd,bnd->bn", q, c) / (retriever.d_model ** 0.5)
        loss = F.cross_entropy(scores, torch.tensor([target_chunk], device=device))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                pred = scores[0].argmax().item()
                hit = 1.0 if pred == target_chunk else 0.0
            print(f"  step={step:4d} loss={loss.item():.4f} chunk_acc={hit:.0f}")


@torch.no_grad()
def evaluate_pipeline(
    model, retriever, text, tokenizer, device, seq_len,
    chunk_size=512, eval_batches=8, selection_temperature=1.0,
):
    """Evaluate pipeline on needle-in-Shakespeare."""
    model.eval()
    retriever.eval()
    vocab_size = model.config.vocab_size
    exact_matches = []
    chunk_correct = []

    for _ in range(eval_batches):
        batch = make_shakespeare_needle_batch(
            1, seq_len, vocab_size, device, text, tokenizer,
        )
        input_ids = batch.input_ids
        targets = batch.target_ids
        mask = batch.loss_mask

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            exact_matches.append(0.0)
            chunk_correct.append(0.0)
            continue
        answer_start = query_pos[0, 1].item()

        mark_positions = (input_ids[0] == MARK_THOUGHT).nonzero(as_tuple=False)
        if len(mark_positions) == 0:
            exact_matches.append(0.0)
            chunk_correct.append(0.0)
            continue
        target_chunk = mark_positions[0, 0].item() // chunk_size

        chunk_embs, _ = compute_chunk_embeddings(model, input_ids, chunk_size)

        qp = query_pos[0, 1].item()
        query_chunk_idx = qp // chunk_size
        query_chunk_start = query_chunk_idx * chunk_size
        query_chunk_end = min(query_chunk_start + chunk_size, input_ids.shape[1])
        query_ids = input_ids[:, query_chunk_start:query_chunk_end]
        query_hidden = model(query_ids, return_hidden_only=True)
        local_qp = qp - query_chunk_start
        query_emb = query_hidden[:, local_qp : local_qp + 1, :].mean(dim=1)

        scores = retriever.score_chunks(query_emb, chunk_embs, query_chunk_idx=query_chunk_idx)
        scaled_scores = scores / selection_temperature
        weights = torch.softmax(scaled_scores, dim=-1)
        selected_idx = weights[0].argmax().item()
        chunk_correct.append(1.0 if selected_idx == target_chunk else 0.0)

        chunk_start_pos = selected_idx * chunk_size
        chunk_end_pos = min(chunk_start_pos + chunk_size, seq_len)
        chunk_tokens = input_ids[0, chunk_start_pos:chunk_end_pos]

        mark_positions = (chunk_tokens == MARK_THOUGHT).nonzero(as_tuple=False)
        if len(mark_positions) == 0:
            exact_matches.append(0.0)
            continue

        mark_pos = mark_positions[0, 0].item()
        password_len = 3
        pw_start = max(0, mark_pos - password_len)
        pw_tokens = chunk_tokens[pw_start:mark_pos]

        chunk_ids_start = max(0, answer_start - model.config.max_seq_len + 64)
        chunk_ids = input_ids[:, chunk_ids_start:]
        chunk_targets = targets[:, chunk_ids_start:]
        chunk_mask = mask[:, chunk_ids_start:]
        base_logits, _ = model(chunk_ids, return_metrics=True)

        readout_scale = 1.0 / (model.config.d_model ** 0.5)
        combined_logits = base_logits.clone()
        local_answer = answer_start - chunk_ids_start
        for pw_i in range(min(password_len, len(pw_tokens))):
            pos = local_answer + pw_i
            if pos >= combined_logits.shape[1]:
                break
            emb = model.token_embedding(pw_tokens[pw_i : pw_i + 1])
            combined_logits[0, pos] += readout_scale * F.linear(
                emb, model.token_embedding.weight
            ).squeeze(0)
        em = masked_exact_match(combined_logits, chunk_targets, chunk_mask)
        exact_matches.append(em)

    pipeline_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    chunk_acc = sum(chunk_correct) / len(chunk_correct) if chunk_correct else 0.0
    return pipeline_em, chunk_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-steps", type=int, default=2000)
    parser.add_argument("--needle-steps", type=int, default=1200)
    parser.add_argument("--retriever-steps", type=int, default=500)
    parser.add_argument("--train-seq-len", type=int, default=512)
    parser.add_argument("--retriever-seq-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-eval-len", type=int, default=131072)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--proj-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lm", action="store_true", help="Skip LM training, use random init")
    parser.add_argument("--skip-needle", action="store_true", help="Skip needle fine-tuning")
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

    # Load Shakespeare
    text = load_dataset("shakespeare", "train")
    tokenizer = CharTokenizer(text)
    actual_vocab = min(args.vocab_size, tokenizer.vocab_size)
    print(f"Shakespeare: {len(text)} chars, vocab={tokenizer.vocab_size}")

    # Build model with TCB for needle task
    config = AnamnesisConfig(
        vocab_size=actual_vocab,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.train_seq_len,
        use_token_copy_buffer=True,
        use_learned_gate=False,
        milestone_token_ids=(MARK_THOUGHT,),
        token_copy_sinusoidal_pos=True,
        position_encoding_type="sinusoidal",
    )
    model = AnamnesisModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Phase 1: Train on Shakespeare LM
    if not args.skip_lm:
        print(f"\n=== Phase 1: Shakespeare LM ({args.lm_steps} steps) ===")
        train_lm(model, text, tokenizer, device, steps=args.lm_steps, seq_len=args.train_seq_len)
    else:
        print("\n=== Phase 1: SKIPPED (random init) ===")

    # Phase 2: Fine-tune on needle-in-Shakespeare
    if not args.skip_needle:
        print(f"\n=== Phase 2: Needle fine-tuning ({args.needle_steps} steps) ===")
        train_on_needle(
            model, text, tokenizer, device,
            steps=args.needle_steps, seq_len=args.train_seq_len,
        )
    else:
        print("\n=== Phase 2: SKIPPED ===")

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Phase 3: Train retriever
    retriever = ChunkRetriever(
        d_model=args.d_model,
        proj_dim=args.proj_dim,
    ).to(device)
    print(f"\nChunkRetriever params: {sum(p.numel() for p in retriever.parameters()):,}")
    print(f"\n=== Phase 3: Train retriever @ {args.retriever_seq_len} ===")
    train_retriever(
        model, retriever, text, tokenizer, device,
        steps=args.retriever_steps, seq_len=args.retriever_seq_len,
        chunk_size=args.chunk_size,
    )

    # Phase 4: Scaling evaluation
    lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    if args.max_eval_len > 131072:
        lengths.extend([262144, 524288, 1048576])
    lengths = [l for l in lengths if l <= args.max_eval_len]
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05]

    print(f"\n=== Phase 4: Scaling evaluation (temperature sweep) ===")
    print(f"{'Length':>8s} {'Chunks':>7s} {'Temp':>5s} {'ChunkAcc':>9s} {'PipelineEM':>11s} {'Time':>6s}")
    print("-" * 55)

    for length in lengths:
        n_chunks = (length + args.chunk_size - 1) // args.chunk_size
        best_em = 0.0
        best_temp = 1.0
        for temp in temperatures:
            t0 = time.time()
            pipeline_em, chunk_acc = evaluate_pipeline(
                model, retriever, text, tokenizer, device, length,
                chunk_size=args.chunk_size, eval_batches=args.eval_batches,
                selection_temperature=temp,
            )
            elapsed = time.time() - t0
            print(
                f"{length:>8d} {n_chunks:>7d} {temp:>5.2f} {chunk_acc:>9.3f} {pipeline_em:>11.3f} {elapsed:>5.1f}s"
            )
            if pipeline_em > best_em:
                best_em = pipeline_em
                best_temp = temp
        print(f"  Best: temp={best_temp:.2f} EM={best_em:.3f}")
        if best_em < 0.1:
            print(f"  Pipeline collapsed at {length}, stopping.")
            break


if __name__ == "__main__":
    main()
