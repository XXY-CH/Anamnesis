#!/usr/bin/env python
"""Transformer vs RetNet pipeline comparison.

Trains both models on needle@512, trains retrievers on each,
then evaluates the chunk retrieval pipeline at increasing lengths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.memory.chunk_retriever import ChunkRetriever, compute_chunk_embeddings
from src.models import RetNetEngramConfig, RetNetEngramModel
from src.models.transformer_baseline import TransformerConfig, TransformerLM
from experiments.train_synthetic import (
    masked_exact_match,
    masked_lm_loss,
    set_seed,
)
from experiments.test_chunk_retriever import (
    make_random_needle_batch,
    get_needle_chunk_index,
)

PAD = 0
START = 1
MARK_THOUGHT = 2
QUERY = 3
SEP = 4


def train_model(model, device, steps=1200, seq_len=512, batch_size=16):
    """Train on fixed-needle task."""
    from experiments.train_synthetic import make_needle_batch

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_needle_batch(batch_size, seq_len, vocab_size, device)
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


def train_retriever(model, retriever, device, steps=500, seq_len=8192, chunk_size=512):
    """Train retriever with contrastive loss."""
    import torch.nn.functional as F

    model.eval()
    retriever.train()
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=3e-3)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_random_needle_batch(1, seq_len, vocab_size, device)
        input_ids = batch.input_ids

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            continue
        target_chunk = get_needle_chunk_index(input_ids, chunk_size)

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
    model, retriever, device, seq_len, chunk_size=512, eval_batches=8,
    selection_temperature=1.0,
):
    """Evaluate pipeline: contrastive chunk selection + token embedding readout."""
    import torch.nn.functional as F

    model.eval()
    retriever.eval()
    vocab_size = model.config.vocab_size
    exact_matches = []
    chunk_correct = []

    for _ in range(eval_batches):
        batch = make_random_needle_batch(1, seq_len, vocab_size, device)
        input_ids = batch.input_ids
        targets = batch.target_ids
        mask = batch.loss_mask

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            exact_matches.append(0.0)
            chunk_correct.append(0.0)
            continue
        answer_start = query_pos[0, 1].item()
        target_chunk = get_needle_chunk_index(input_ids, chunk_size)

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


def make_retnet(args, device):
    config = RetNetEngramConfig(
        vocab_size=args.vocab_size,
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
    return RetNetEngramModel(config).to(device)


def make_transformer(args, device):
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.train_seq_len,
    )
    return TransformerLM(config).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=1200)
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

    lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    if args.max_eval_len > 131072:
        lengths.extend([262144, 524288, 1048576])
    lengths = [l for l in lengths if l <= args.max_eval_len]
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05]

    results = {}

    for model_name, make_fn in [("RetNet", make_retnet), ("Transformer", make_transformer)]:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        model = make_fn(args, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")

        print(f"\n--- Train @ {args.train_seq_len} ---")
        train_model(model, device, steps=args.train_steps, seq_len=args.train_seq_len)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        retriever = ChunkRetriever(d_model=args.d_model).to(device)
        print(f"Retriever params: {sum(p.numel() for p in retriever.parameters()):,}")
        print(f"\n--- Train retriever @ {args.retriever_seq_len} ---")
        train_retriever(
            model, retriever, device,
            steps=args.retriever_steps, seq_len=args.retriever_seq_len,
            chunk_size=args.chunk_size,
        )

        print(f"\n--- Scaling evaluation ---")
        print(f"{'Length':>8s} {'Chunks':>7s} {'Temp':>5s} {'ChunkAcc':>9s} {'PipelineEM':>11s}")
        print("-" * 50)

        model_results = []
        for length in lengths:
            n_chunks = (length + args.chunk_size - 1) // args.chunk_size
            best_em = 0.0
            best_temp = 1.0
            for temp in temperatures:
                pipeline_em, chunk_acc = evaluate_pipeline(
                    model, retriever, device, length,
                    chunk_size=args.chunk_size, eval_batches=args.eval_batches,
                    selection_temperature=temp,
                )
                print(
                    f"{length:>8d} {n_chunks:>7d} {temp:>5.2f} {chunk_acc:>9.3f} {pipeline_em:>11.3f}"
                )
                if pipeline_em > best_em:
                    best_em = pipeline_em
                    best_temp = temp
            model_results.append((length, best_em, best_temp))
            print(f"  Best: temp={best_temp:.2f} EM={best_em:.3f}")

        results[model_name] = model_results

        del model, retriever
        if args.device == "mps":
            torch.mps.empty_cache()

    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Length':>8s} {'RetNet EM':>10s} {'Trans EM':>10s} {'Delta':>8s}")
    print("-" * 42)
    retnet_dict = {r[0]: r[1] for r in results.get("RetNet", [])}
    trans_dict = {r[0]: r[1] for r in results.get("Transformer", [])}
    for length in lengths:
        r_em = retnet_dict.get(length, 0.0)
        t_em = trans_dict.get(length, 0.0)
        delta = r_em - t_em
        print(f"{length:>8d} {r_em:>10.3f} {t_em:>10.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
