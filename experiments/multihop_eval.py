"""Multi-needle retrieval evaluation.

Tests whether the retrieval pipeline can find MULTIPLE needles in long context.
Extends the single-needle EM=1.000@1M result to multi-needle scenarios.

Protocol:
1. Train small Anamnesis on needle task (400 steps, d=64)
2. Freeze model, extract chunk embeddings
3. Train retriever on multi-needle data (200 steps)
4. Evaluate: top-K recall at various lengths
"""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anamnesis import AnamnesisConfig, AnamnesisModel
from src.memory.chunk_retriever import ChunkRetriever


# --- Special tokens ---
PAD, START, MARK, QUERY, SEP = 0, 1, 2, 3, 4
VOCAB = 192


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_single_needle(
    seq_len: int, rng: random.Random, password_len: int = 3
) -> tuple:
    """Generate single-needle data."""
    password = [rng.randint(10, VOCAB // 2) for _ in range(password_len)]
    filler = [rng.randint(5, 9) for _ in range(seq_len)]
    position = rng.randint(seq_len // 4, 3 * seq_len // 4)

    ids = filler.copy()
    ids[0] = START
    ids[position] = MARK
    for i, t in enumerate(password):
        if position + 1 + i < seq_len - password_len - 2:
            ids[position + 1 + i] = t

    query_pos = position + 1 + password_len
    if query_pos + 2 < seq_len:
        ids[query_pos] = QUERY
        ids[query_pos + 1] = SEP

    answer_start = query_pos + 2

    input_ids = torch.tensor([ids], dtype=torch.long)
    target_ids = torch.roll(input_ids, -1, dims=-1)

    loss_mask = torch.zeros(1, seq_len, dtype=torch.long)
    for i in range(password_len):
        if answer_start + i < seq_len:
            loss_mask[0, answer_start + i] = 1

    return input_ids, target_ids, loss_mask, answer_start, password


def generate_multineedle(
    seq_len: int,
    n_needles: int,
    chunk_size: int,
    rng: random.Random,
    password_len: int = 3,
) -> dict:
    """Generate multi-needle data with needles in different chunks."""
    n_chunks = seq_len // chunk_size
    needle_chunks = rng.sample(range(n_chunks), n_needles + 1)
    query_chunk = needle_chunks[-1]
    needle_chunks = needle_chunks[:-1]

    passwords = []
    ids = [rng.randint(5, 9) for _ in range(seq_len)]
    ids[0] = START

    for ci in needle_chunks:
        pw = [rng.randint(10, VOCAB // 2) for _ in range(password_len)]
        passwords.append(pw)
        offset = ci * chunk_size
        pos = offset + rng.randint(4, chunk_size // 2)
        ids[pos] = MARK
        for i, t in enumerate(pw):
            if pos + 1 + i < seq_len:
                ids[pos + 1 + i] = t

    # Query at end of query chunk
    q_start = query_chunk * chunk_size + chunk_size - password_len * n_needles - 3
    if q_start < query_chunk * chunk_size:
        q_start = query_chunk * chunk_size
    ids[q_start] = QUERY
    ids[q_start + 1] = SEP
    answer_start = q_start + 2

    all_pw = []
    for pw in passwords:
        all_pw.extend(pw)
    for i, t in enumerate(all_pw):
        if answer_start + i < seq_len:
            ids[answer_start + i] = t

    input_ids = torch.tensor([ids], dtype=torch.long)
    target_ids = torch.roll(input_ids, -1, dims=-1)
    loss_mask = torch.zeros(1, seq_len, dtype=torch.long)
    for i in range(len(all_pw)):
        if answer_start + i < seq_len:
            loss_mask[0, answer_start + i] = 1

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "needle_chunks": needle_chunks,
        "query_chunk": query_chunk,
        "passwords": passwords,
    }


def train_needle_model(
    steps: int = 400, seq_len: int = 512, device: str = "mps", seed: int = 42
) -> AnamnesisModel:
    """Train a small Anamnesis model on single-needle task."""
    set_seed(seed)
    config = AnamnesisConfig(
        vocab_size=VOCAB, d_model=64, n_heads=8, n_layers=8,
        d_ff=256, max_seq_len=2048, layerwise_gamma=True,
        engram_layers=(2,), engram_num_slots=4096, engram_use_conv=True,
    )
    model = AnamnesisModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(seed)

    for step in range(1, steps + 1):
        input_ids, target_ids, loss_mask, _, _ = generate_single_needle(seq_len, rng)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB), target_ids.view(-1), reduction="none"
        )
        loss = (loss.view(1, -1) * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"  Needle training step {step}: loss={loss.item():.4f}")

    return model


def extract_chunk_embeddings(
    model: AnamnesisModel,
    input_ids: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Extract per-chunk embeddings from frozen model (chunkwise to avoid max_seq_len)."""
    with torch.no_grad():
        seq_len = input_ids.shape[1]
        n_chunks = seq_len // chunk_size
        embs = []
        for c in range(n_chunks):
            start = c * chunk_size
            end = start + chunk_size
            chunk_ids = input_ids[:, start:end]
            hidden = model(chunk_ids, return_hidden_only=True)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            chunk_emb = hidden[0].mean(dim=0)
            embs.append(chunk_emb)
        return torch.stack(embs)


def get_query_embedding(
    model: AnamnesisModel,
    input_ids: torch.Tensor,
    query_pos: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Get query position hidden state (process only query chunk)."""
    with torch.no_grad():
        chunk_idx = query_pos // chunk_size
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_ids = input_ids[:, start:end]
        hidden = model(chunk_ids, return_hidden_only=True)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        local_pos = query_pos - start
        return hidden[0, local_pos]


def train_retriever(
    model: AnamnesisModel,
    retriever: ChunkRetriever,
    n_steps: int = 200,
    seq_len: int = 2048,
    chunk_size: int = 512,
    n_needles: int = 2,
    device: str = "mps",
    seed: int = 42,
):
    """Train retriever on multi-needle data with contrastive loss."""
    rng = random.Random(seed + 1000)
    optimizer = torch.optim.Adam(retriever.parameters(), lr=3e-3)

    for step in range(1, n_steps + 1):
        data = generate_multineedle(seq_len, n_needles, chunk_size, rng)
        input_ids = data["input_ids"].to(device)
        needle_chunks = data["needle_chunks"]

        chunk_embs = extract_chunk_embeddings(model, input_ids, chunk_size)
        n_chunks = chunk_embs.shape[0]

        query_pos_candidates = (input_ids[0] == QUERY).nonzero(as_tuple=True)[0]
        if len(query_pos_candidates) == 0:
            continue
        query_pos = query_pos_candidates[0].item()
        query_emb = get_query_embedding(model, input_ids, query_pos, chunk_size)

        scores = retriever.score_chunks(
            query_emb.unsqueeze(0), chunk_embs.unsqueeze(0)
        ).squeeze(0)

        labels = torch.zeros(n_chunks, device=device)
        for nc in needle_chunks:
            if nc < n_chunks:
                labels[nc] = 1.0

        n_pos = labels.sum().clamp(min=1)
        n_neg = n_chunks - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        loss = F.binary_cross_entropy_with_logits(
            scores, labels, pos_weight=pos_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            preds = (scores > 0).float()
            acc = (preds == labels).float().mean().item()
            print(f"  Retriever step {step}: loss={loss.item():.4f}, acc={acc:.3f}")


def evaluate_multineedle(
    model: AnamnesisModel,
    retriever: ChunkRetriever,
    seq_len: int,
    chunk_size: int,
    n_needles: int,
    n_eval: int = 50,
    device: str = "mps",
    seed: int = 42,
) -> dict:
    """Evaluate multi-needle recall at K."""
    rng = random.Random(seed + 9999)
    top1_recall = 0
    topk_recall = 0
    total = 0

    for i in range(n_eval):
        data = generate_multineedle(seq_len, n_needles, chunk_size, rng)
        input_ids = data["input_ids"].to(device)
        needle_chunks = data["needle_chunks"]

        chunk_embs = extract_chunk_embeddings(model, input_ids, chunk_size)
        n_chunks = chunk_embs.shape[0]

        query_pos_candidates = (input_ids[0] == QUERY).nonzero(as_tuple=True)[0]
        if len(query_pos_candidates) == 0:
            continue
        query_pos = query_pos_candidates[0].item()
        query_emb = get_query_embedding(model, input_ids, query_pos, chunk_size)

        with torch.no_grad():
            scores = retriever.score_chunks(
                query_emb.unsqueeze(0), chunk_embs.unsqueeze(0)
            ).squeeze(0)

        _, topk_indices = scores.topk(n_needles)
        topk_set = set(topk_indices.tolist())
        needle_set = set(needle_chunks)

        if needle_set.issubset(topk_set):
            topk_recall += 1

        if topk_indices[0].item() in needle_set:
            top1_recall += 1

        total += 1

    return {
        "top1_recall": top1_recall / total if total > 0 else 0,
        f"top{n_needles}_recall": topk_recall / total if total > 0 else 0,
        "total": total,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--needle-steps", type=int, default=400)
    p.add_argument("--retriever-steps", type=int, default=200)
    p.add_argument("--n-needles", type=int, default=2)
    p.add_argument("--n-eval", type=int, default=50)
    args = p.parse_args()

    device = args.device
    seed = args.seed
    print(f"=== Multi-Needle Retrieval Evaluation (seed={seed}) ===")
    print(f"Device: {device}, Needles: {args.n_needles}")

    # Step 1: Train needle model
    print("\n[1] Training needle model...")
    model = train_needle_model(
        steps=args.needle_steps, seq_len=512, device=device, seed=seed
    )
    model.eval()
    print("  Done.")

    # Step 2: Train retriever on multi-needle data
    print(f"\n[2] Training retriever on {args.n_needles}-needle data...")
    retriever = ChunkRetriever(d_model=64, proj_dim=256).to(device)
    train_retriever(
        model, retriever,
        n_steps=args.retriever_steps,
        seq_len=2048, chunk_size=512,
        n_needles=args.n_needles,
        device=device, seed=seed,
    )
    print("  Done.")

    # Step 3: Evaluate at multiple lengths
    print(f"\n[3] Evaluating multi-needle recall...")
    results = {}
    for seq_len in [2048]:
        chunk_size = 512
        n_chunks = seq_len // chunk_size
        if n_chunks < args.n_needles + 1:
            continue
        r = evaluate_multineedle(
            model, retriever,
            seq_len=seq_len, chunk_size=chunk_size,
            n_needles=args.n_needles,
            n_eval=args.n_eval,
            device=device, seed=seed,
        )
        results[seq_len] = r
        print(f"  {seq_len:>6} ({n_chunks:>4} chunks): "
              f"top-1={r['top1_recall']:.3f}, "
              f"top-{args.n_needles}={r[f'top{args.n_needles}_recall']:.3f}")

    print("\n=== Summary ===")
    print(f"Needles: {args.n_needles}, Eval samples: {args.n_eval}")
    for sl, r in results.items():
        print(f"  {sl}: top-1={r['top1_recall']:.3f}, "
              f"top-{args.n_needles} recall={r[f'top{args.n_needles}_recall']:.3f}")
