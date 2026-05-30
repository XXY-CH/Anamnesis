#!/usr/bin/env python
"""Multi-needle recall capacity sweep at 128K.

Trains model on single-needle@512, retriever on single-needle@8192,
then evaluates multi-needle recall at 128K with top-K_max chunk selection.
Sweeps N_needles in [1..8] and K_max in {2, 4, 8, 16}.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.memory.chunk_retriever import ChunkRetriever, compute_chunk_embeddings
from src.models import AnamnesisConfig, AnamnesisModel
from experiments.train_synthetic import (
    masked_exact_match,
    masked_lm_loss,
    set_seed,
)
from experiments.test_chunk_retriever import (
    make_random_needle_batch,
    get_needle_chunk_index,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
    "axes.edgecolor": "#333333",
    "grid.color": "#EAEAEA",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
})

OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAD = 0
START = 1
MARK_THOUGHT = 2
QUERY = 3
SEP = 4

K_COLORS = {
    2: "#B8A9C9",
    4: "#7EB5D6",
    8: "#A3C4A8",
    16: "#E8B87D",
}
K_MARKERS = {2: "o", 4: "s", 8: "^", 16: "D"}


def make_multi_needle_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    n_needles: int,
    chunk_size: int = 512,
    filler_low: int = 16,
    password_len: int = 3,
) -> tuple[torch.Tensor, list[list[int]]]:
    n_chunks = seq_len // chunk_size
    tokens = torch.randint(
        filler_low, vocab_size, (batch_size, seq_len + 1), device=device
    )
    all_needle_chunks: list[list[int]] = []

    for b in range(batch_size):
        tokens[b, 0] = START
        available = list(range(2, n_chunks - 2))
        n_actual = min(n_needles, len(available))
        needle_chunks = sorted(random.sample(available, n_actual))

        last_pw = None
        for i, ci in enumerate(needle_chunks):
            pw = torch.randint(5, vocab_size, (password_len,), device=device)
            start = ci * chunk_size
            offset = random.randint(10, chunk_size - password_len - 10)
            pos = start + offset
            tokens[b, pos:pos + password_len] = pw
            tokens[b, pos + password_len] = MARK_THOUGHT
            if i == n_actual - 1:
                last_pw = pw

        tokens[b, -password_len - 1] = QUERY
        if last_pw is not None:
            tokens[b, -password_len:] = last_pw

        all_needle_chunks.append(needle_chunks)

    return tokens[:, :-1], all_needle_chunks


def train_model(
    model: AnamnesisModel,
    device: torch.device,
    steps: int = 1200,
    seq_len: int = 512,
    batch_size: int = 16,
) -> None:
    from experiments.train_synthetic import make_needle_batch

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    vocab_size = model.config.vocab_size

    for step in range(1, steps + 1):
        batch = make_needle_batch(batch_size, seq_len, vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, metrics = model(batch.input_ids, return_metrics=True)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            with torch.no_grad():
                em = masked_exact_match(logits, batch.target_ids, batch.loss_mask)
            print(f"  step={step:4d} loss={loss.item():.4f} em={em:.3f}")


def train_retriever(
    model: AnamnesisModel,
    retriever: ChunkRetriever,
    device: torch.device,
    steps: int = 500,
    seq_len: int = 8192,
    chunk_size: int = 512,
) -> None:
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
            qci = qp // chunk_size
            qs = qci * chunk_size
            qe = min(qs + chunk_size, input_ids.shape[1])
            qh = model(input_ids[:, qs:qe], return_hidden_only=True)
            query_emb = qh[:, qp - qs:qp - qs + 1, :].mean(dim=1)

        q = retriever.query_proj(query_emb)
        c = retriever.chunk_proj(chunk_embs)
        scores = torch.einsum("bd,bnd->bn", q, c) / (retriever.d_model ** 0.5)
        loss = F.cross_entropy(
            scores, torch.tensor([target_chunk], device=device)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"  step={step:4d} loss={loss.item():.4f}")


@torch.no_grad()
def evaluate_multi_needle(
    model: AnamnesisModel,
    retriever: ChunkRetriever,
    device: torch.device,
    seq_len: int,
    n_needles: int,
    k_max: int,
    chunk_size: int = 512,
    eval_batches: int = 8,
) -> float:
    model.eval()
    retriever.eval()
    vocab_size = model.config.vocab_size
    recalls: list[float] = []

    for _ in range(eval_batches):
        input_ids, needle_chunks_list = make_multi_needle_batch(
            1, seq_len, vocab_size, device, n_needles, chunk_size
        )

        chunk_embs, _ = compute_chunk_embeddings(model, input_ids, chunk_size)

        query_pos = (input_ids == QUERY).nonzero(as_tuple=False)
        if len(query_pos) == 0:
            recalls.append(0.0)
            continue
        qp = query_pos[0, 1].item()
        qci = qp // chunk_size
        qs = qci * chunk_size
        qe = min(qs + chunk_size, input_ids.shape[1])
        qh = model(input_ids[:, qs:qe], return_hidden_only=True)
        query_emb = qh[:, qp - qs:qp - qs + 1, :].mean(dim=1)

        scores = retriever.score_chunks(query_emb, chunk_embs)
        k = min(k_max, scores.shape[1])
        _, top_idx = scores[0].topk(k)
        top_set = set(top_idx.tolist())

        needle_set = set(needle_chunks_list[0])
        found = len(top_set & needle_set)
        recall = found / len(needle_set) if needle_set else 0.0
        recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0.0


def plot_capacity_curve(
    results: dict[int, dict[int, float]], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    needles = sorted(results.keys())
    k_values = sorted(set(k for v in results.values() for k in v.keys()))

    for k_max in k_values:
        ems = [results[n][k_max] for n in needles]
        ax.plot(
            needles, ems,
            marker=K_MARKERS[k_max],
            color=K_COLORS[k_max],
            linewidth=2,
            markersize=8,
            label=f"$K_{{max}}$ = {k_max}",
        )

    ax.set_title(
        "Multi-Needle Recall Capacity at 128K Context",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Number of Needles (N)", fontsize=11, labelpad=8)
    ax.set_ylabel("Recall@$K_{max}$ (EM)", fontsize=11, labelpad=8)
    ax.set_xticks(needles)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        frameon=True, facecolor="#FDFDFD", edgecolor="#E2E2E2",
        fontsize=10, title="Retrieval Budget",
    )
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--use-engram", action="store_true", default=False)
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
        vocab_size=192,
        d_model=64,
        n_heads=4,
        n_layers=8,
        max_seq_len=512,
        engram_layers=(2,) if args.use_engram else (),
        use_token_copy_buffer=True,
        milestone_token_ids=(MARK_THOUGHT,),
        token_copy_sinusoidal_pos=True,
        position_encoding_type="sinusoidal",
        attnres_every=0,
    )
    model = AnamnesisModel(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n=== Phase 1: Train model on needle@512 ===")
    train_model(model, device, steps=1200, seq_len=512)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    retriever = ChunkRetriever(d_model=64, proj_dim=256).to(device)
    print(f"\n=== Phase 2: Train retriever @8192 ===")
    train_retriever(model, retriever, device, steps=500, seq_len=8192)

    seq_len = 131072
    needle_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    k_max_values = [2, 4, 8, 16]

    results: dict[int, dict[int, float]] = {}

    print(f"\n=== Phase 3: Multi-needle sweep at 128K ===")
    for n_needles in needle_counts:
        results[n_needles] = {}
        for k_max in k_max_values:
            t0 = time.time()
            recall = evaluate_multi_needle(
                model, retriever, device, seq_len, n_needles, k_max,
                eval_batches=args.eval_batches,
            )
            elapsed = time.time() - t0
            results[n_needles][k_max] = recall
            print(f"  N={n_needles} K_max={k_max:2d} Recall={recall:.3f} ({elapsed:.1f}s)")

    out_path = OUT_DIR / "fig_multi_needle_capacity.pdf"
    plot_capacity_curve(results, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
