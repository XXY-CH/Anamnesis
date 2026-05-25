#!/usr/bin/env python
"""Test GCA-style Chunk Retrieval for long-context extension.

Three-phase experiment:
1. Train Small Reasoner on random-needle@512
2. Diagnostic: can frozen model distinguish needle chunk from filler?
3. Train ChunkRetriever on random-needle@2048, evaluate vs controls

Random needle: password at random position (not always at start).
This forces the retriever to actually learn content-based selection,
not just position bias.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
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
)

PAD = 0
START = 1
MARK_THOUGHT = 2
QUERY = 3
SEP = 4


@dataclass
class SyntheticBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor


def make_random_needle_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    filler_low: int = 16,
    password_len: int = 3,
    needle_region: float = 0.8,
) -> SyntheticBatch:
    """Needle-in-a-haystack with random needle position.

    Password is placed at a random position in [1, seq_len*needle_region].
    Forces the model/retriever to find the needle by content, not position.
    """
    total = seq_len + 1
    tokens = torch.randint(filler_low, vocab_size, (batch_size, total), device=device)
    passwords = torch.randint(5, vocab_size, (batch_size, password_len), device=device)

    for b in range(batch_size):
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


def get_needle_chunk_index(
    input_ids: torch.Tensor,
    chunk_size: int,
) -> int:
    """Find which chunk contains the password (MARK_THOUGHT marks its end)."""
    mark_positions = (input_ids[0] == MARK_THOUGHT).nonzero(as_tuple=False)
    if len(mark_positions) == 0:
        return 0
    mark_pos = mark_positions[0, 0].item()
    return mark_pos // chunk_size


def train_model(model, device, steps=400, seq_len=512, batch_size=16, lr=3e-4,
                use_random_needle=False):
    """Train on needle task. Fixed needle by default; random if flag set."""
    from experiments.train_synthetic import make_needle_batch as make_fixed_batch

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        if use_random_needle:
            batch = make_random_needle_batch(batch_size, seq_len, vocab_size, device)
        else:
            batch = make_fixed_batch(batch_size, seq_len, vocab_size, device)
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
def diagnostic_chunk_discrimination(
    model, device, seq_len=2048, chunk_size=512, num_trials=32,
):
    """Test: can frozen model's hidden states distinguish the needle chunk?

    If the model's chunk embeddings carry no signal about which chunk contains
    the needle, then content-based retrieval is impossible. This diagnostic
    checks whether chunk retrieval is even feasible before training a retriever.
    """
    vocab_size = model.config.vocab_size
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    correct = 0
    top3_correct = 0

    for _ in range(num_trials):
        batch = make_random_needle_batch(1, seq_len, vocab_size, device)
        input_ids = batch.input_ids
        target_chunk = get_needle_chunk_index(input_ids, chunk_size)

        chunk_embs, _ = compute_chunk_embeddings(model, input_ids, chunk_size)

        # Query embedding from QUERY position
        query_pos = (input_ids[0] == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            continue
        qp = query_pos[0, 0].item()
        # Get hidden at query position by running the chunk containing it
        query_chunk_idx = qp // chunk_size
        query_chunk_start = query_chunk_idx * chunk_size
        query_chunk_end = min(query_chunk_start + chunk_size, seq_len)
        query_ids = input_ids[:, query_chunk_start:query_chunk_end]
        query_hidden = model(query_ids, return_hidden_only=True)
        local_qp = qp - query_chunk_start
        query_emb = query_hidden[:, local_qp : local_qp + 1, :].mean(dim=1)

        # Score chunks by dot product with query
        scores = torch.einsum(
            "bd,bnd->bn", query_emb, chunk_embs
        ).squeeze(0)

        predicted = scores.argmax().item()
        top3 = scores.topk(min(3, num_chunks)).indices.tolist()

        if predicted == target_chunk:
            correct += 1
        if target_chunk in top3:
            top3_correct += 1

    acc = correct / num_trials
    top3_acc = top3_correct / num_trials
    random_baseline = 1.0 / num_chunks
    print(f"  Chunk discrimination (needle chunk identification):")
    print(f"    Accuracy:       {acc:.3f} (random baseline: {random_baseline:.3f})")
    print(f"    Top-3 accuracy: {top3_acc:.3f}")
    return acc, top3_acc


def train_chunk_retriever(
    model, retriever, device, steps=200, seq_len=2048, batch_size=1,
    lr=1e-3, chunk_size=512, contrastive=True,
):
    """Train ChunkRetriever on long sequences. Model stays frozen.

    If contrastive=True, trains the retriever to identify the needle chunk
    via cross-entropy (no generation loss needed). Otherwise trains via
    generation loss (requires model to already work).
    """
    model.eval()
    retriever.train()
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=lr)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_random_needle_batch(batch_size, seq_len, vocab_size, device)
        input_ids = batch.input_ids

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            continue
        target_chunk = get_needle_chunk_index(input_ids, chunk_size)

        with torch.no_grad():
            chunk_embs, _ = compute_chunk_embeddings(model, input_ids, chunk_size)

            qp = query_pos[0, 1].item()
            query_chunk_idx = qp // chunk_size
            query_chunk_start = query_chunk_idx * chunk_size
            query_chunk_end = min(query_chunk_start + chunk_size, input_ids.shape[1])
            query_ids = input_ids[:, query_chunk_start:query_chunk_end]
            query_hidden = model(query_ids, return_hidden_only=True)
            local_qp = qp - query_chunk_start
            query_emb = query_hidden[:, local_qp : local_qp + 1, :].mean(dim=1)

        # Score chunks
        q = retriever.query_proj(query_emb)
        c = retriever.chunk_proj(chunk_embs)
        scores = torch.einsum("bd,bnd->bn", q, c) / (retriever.d_model ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        if contrastive:
            loss = F.cross_entropy(scores, torch.tensor([target_chunk], device=device))
        else:
            targets = batch.target_ids
            mask = batch.loss_mask
            answer_start = query_pos[0, 1].item() + 1

            with torch.no_grad():
                chunk_start = max(0, answer_start - model.config.max_seq_len + 64)
                chunk_ids = input_ids[:, chunk_start:]
                chunk_targets = targets[:, chunk_start:]
                chunk_mask = mask[:, chunk_start:]
                base_logits, _ = model(chunk_ids, return_metrics=True)

            v = retriever.value_proj(chunk_embs)
            summary = torch.einsum("bn,bnd->bd", weights, v)
            logit_corr = retriever.logit_scale * F.linear(
                summary, model.token_embedding.weight
            )
            combined_logits = base_logits + logit_corr.unsqueeze(1)
            loss = masked_lm_loss(combined_logits, chunk_targets, chunk_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                pred_chunk = weights[0].argmax().item()
                hit = 1.0 if pred_chunk == target_chunk else 0.0
            extra = ""
            if not contrastive:
                with torch.no_grad():
                    em = masked_exact_match(combined_logits, chunk_targets, chunk_mask)
                extra = f" em={em:.3f}"
            print(
                f"  step={step:4d} loss={loss.item():.4f} "
                f"chunk_acc={hit:.0f}{extra} "
                f"top_weights={weights[0].topk(3).values.tolist()}"
            )


@torch.no_grad()
def evaluate(
    model, retriever, device, seq_len, chunk_size=512,
    eval_batches=16, mode="retriever", use_random_needle=True,
):
    """Evaluate with different modes.

    Modes:
        model_only: model alone (fails if seq_len > max_seq_len)
        retriever: model + trained ChunkRetriever (contrastive selection + position readout)
        oracle_retriever: model + oracle chunk selection (always picks needle chunk)
        mean_pool: model + uniform weight over all chunks (retriever ablation)
    """
    from experiments.train_synthetic import make_needle_batch as make_fixed_batch

    model.eval()
    if retriever is not None:
        retriever.eval()
    vocab_size = model.config.vocab_size
    exact_matches = []

    for _ in range(eval_batches):
        if use_random_needle:
            batch = make_random_needle_batch(1, seq_len, vocab_size, device)
        else:
            batch = make_fixed_batch(1, seq_len, vocab_size, device)
        input_ids = batch.input_ids
        targets = batch.target_ids
        mask = batch.loss_mask

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            exact_matches.append(0.0)
            continue
        answer_start = query_pos[0, 1].item() + 1

        if mode == "model_only":
            if seq_len > model.config.max_seq_len:
                exact_matches.append(0.0)
                continue
            logits, _ = model(input_ids, return_metrics=True)
            em = masked_exact_match(logits, targets, mask)
            exact_matches.append(em)
            continue

        chunk_embs, chunk_hiddens_list = compute_chunk_embeddings(
            model, input_ids, chunk_size
        )

        # Query embedding
        qp = query_pos[0, 1].item()
        query_chunk_idx = qp // chunk_size
        query_chunk_start = query_chunk_idx * chunk_size
        query_chunk_end = min(query_chunk_start + chunk_size, input_ids.shape[1])
        query_ids = input_ids[:, query_chunk_start:query_chunk_end]
        query_hidden = model(query_ids, return_hidden_only=True)
        local_qp = qp - query_chunk_start
        query_emb = query_hidden[:, local_qp : local_qp + 1, :].mean(dim=1)

        # Model logits on answer chunk
        chunk_start = max(0, answer_start - model.config.max_seq_len + 64)
        chunk_ids = input_ids[:, chunk_start:]
        chunk_targets = targets[:, chunk_start:]
        chunk_mask = mask[:, chunk_start:]
        base_logits, _ = model(chunk_ids, return_metrics=True)

        if mode == "retriever":
            q = retriever.query_proj(query_emb)
            c = retriever.chunk_proj(chunk_embs)
            scores = torch.einsum("bd,bnd->bn", q, c) / (retriever.d_model ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            selected_idx = weights[0].argmax().item()
        elif mode == "oracle_retriever":
            selected_idx = get_needle_chunk_index(input_ids, chunk_size)
        elif mode == "mean_pool":
            # Average all chunk hiddens
            all_hiddens = torch.cat(
                [h.squeeze(0) for h in chunk_hiddens_list], dim=0
            )
            summary = all_hiddens.mean(dim=0, keepdim=True)  # [1, d]
            logit_corr = retriever.logit_scale * F.linear(
                summary, model.token_embedding.weight
            )
            combined_logits = base_logits + logit_corr
            em = masked_exact_match(combined_logits, chunk_targets, chunk_mask)
            exact_matches.append(em)
            continue
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Position-level readout from selected chunk
        selected_hidden = chunk_hiddens_list[selected_idx]  # [1, C, d]
        # Cross-attention: query attends to selected chunk positions
        attn_scores = torch.einsum(
            "bd,bnd->bn", query_emb, selected_hidden.squeeze(0).unsqueeze(0)
        ) / (retriever.d_model ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [1, C]
        # Weighted readout of selected chunk
        readout = torch.einsum(
            "bn,bnd->bd", attn_weights, selected_hidden
        )  # [1, d]
        logit_corr = retriever.logit_scale * F.linear(
            readout, model.token_embedding.weight
        )
        combined_logits = base_logits + logit_corr
        em = masked_exact_match(combined_logits, chunk_targets, chunk_mask)
        exact_matches.append(em)

    return sum(exact_matches) / len(exact_matches) if exact_matches else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--retriever-steps", type=int, default=200)
    parser.add_argument("--train-seq-len", type=int, default=512)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--contrastive", action="store_true", default=True)
    parser.add_argument("--no-contrastive", dest="contrastive", action="store_false")
    parser.add_argument("--random-train", action="store_true", default=False)
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
        token_copy_sinusoidal_pos=True,
        position_encoding_type="sinusoidal",
    )
    model = AnamnesisModel(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Train on needle@512
    needle_type = "random-needle" if args.random_train else "fixed-needle"
    print(f"\n=== Phase 1: Train on {needle_type}@{args.train_seq_len} ===")
    train_model(
        model, device,
        steps=args.train_steps, seq_len=args.train_seq_len,
        use_random_needle=args.random_train,
    )

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Phase 2: Diagnostic — chunk discrimination
    print(f"\n=== Phase 2: Chunk discrimination diagnostic@{args.eval_seq_len} ===")
    diag_acc, _ = diagnostic_chunk_discrimination(
        model, device, seq_len=args.eval_seq_len, chunk_size=args.chunk_size,
    )

    if diag_acc < 0.15:
        print("  WARNING: frozen model cannot distinguish needle chunk.")
        print("  Chunk retrieval unlikely to work. Results below are exploratory.")

    # Build retriever
    retriever = ChunkRetriever(d_model=args.d_model).to(device)
    print(f"\nChunkRetriever params: {sum(p.numel() for p in retriever.parameters()):,}")

    # Phase 3: Baseline at training length (same distribution as training)
    print(f"\n=== Phase 3: Baseline@{args.train_seq_len} ===")
    em_fixed = evaluate(
        model, None, device, args.train_seq_len, mode="model_only",
        use_random_needle=False,
    )
    em_random = evaluate(
        model, None, device, args.train_seq_len, mode="model_only",
        use_random_needle=True,
    )
    print(f"  Model (fixed-needle): EM = {em_fixed:.3f}")
    print(f"  Model (random-needle): EM = {em_random:.3f}")

    # Phase 4: Train retriever on long sequences
    mode_str = "contrastive" if args.contrastive else "generation"
    print(f"\n=== Phase 4: Train retriever ({mode_str}) on random-needle@{args.eval_seq_len} ===")
    train_chunk_retriever(
        model, retriever, device,
        steps=args.retriever_steps, seq_len=args.eval_seq_len,
        chunk_size=args.chunk_size, contrastive=args.contrastive,
    )

    # Phase 5: Final evaluation with all controls
    print(f"\n=== Phase 5: Final evaluation ===")
    header = f"{'Len':>5s} {'Type':>6s} {'Model':>7s} {'MeanP':>6s} {'Retrv':>6s} {'Oracle':>7s}"
    print(header)
    print("-" * len(header))

    for length in [args.train_seq_len, args.eval_seq_len]:
        for ntype, random_flag in [("fixed", False), ("random", True)]:
            em_model = evaluate(
                model, retriever, device, length, mode="model_only",
                use_random_needle=random_flag,
            )
            if length <= model.config.max_seq_len:
                em_mean = evaluate(
                    model, retriever, device, length, mode="mean_pool",
                    use_random_needle=random_flag,
                )
                em_retr = evaluate(
                    model, retriever, device, length, mode="retriever",
                    use_random_needle=random_flag,
                )
                em_orcl = evaluate(
                    model, retriever, device, length, mode="oracle_retriever",
                    use_random_needle=random_flag,
                )
                print(
                    f"{length:>5d} {ntype:>6s} {em_model:>7.3f} {em_mean:>6.3f} "
                    f"{em_retr:>6.3f} {em_orcl:>7.3f}"
                )
            else:
                em_mean = evaluate(
                    model, retriever, device, length, mode="mean_pool",
                    use_random_needle=random_flag,
                )
                em_retr = evaluate(
                    model, retriever, device, length, mode="retriever",
                    use_random_needle=random_flag,
                )
                em_orcl = evaluate(
                    model, retriever, device, length, mode="oracle_retriever",
                    use_random_needle=random_flag,
                )
                print(
                    f"{length:>5d} {ntype:>6s} {'N/A':>7s} {em_mean:>6.3f} "
                    f"{em_retr:>6.3f} {em_orcl:>7.3f}"
                )


if __name__ == "__main__":
    main()
