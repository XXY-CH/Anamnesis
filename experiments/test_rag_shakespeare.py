"""RAG evaluation on real data (Shakespeare).

Validates Proof 41 (RAG Separation Principle):
- Retriever finds relevant chunks (content-based cosine similarity)
- Model processes retrieved chunks as native text context
- Measures perplexity improvement from retrieved context

Uses the model's own hidden states as chunk embeddings (no separate retriever training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

# Ensure project root is on sys.path for `src` and `experiments` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from train_real import CharTokenizer, load_dataset
from src.models import AnamnesisModel
from src.models.anamnesis import AnamnesisConfig


def encode_chunks(
    model: AnamnesisModel,
    chunk_ids: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Encode chunks into mean-pooled hidden state embeddings."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for ids in chunk_ids:
            ids = ids.unsqueeze(0).to(device)
            hidden = model(ids, return_hidden_only=True)
            embeddings.append(hidden.mean(dim=1))
    return torch.cat(embeddings, dim=0)


def retrieve_top_k(
    query_emb: torch.Tensor,
    chunk_embs: torch.Tensor,
    k: int = 1,
    exclude_indices: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Find top-K most similar chunks by cosine similarity."""
    sim = F.cosine_similarity(query_emb.unsqueeze(0), chunk_embs, dim=-1)
    if exclude_indices:
        for idx in exclude_indices:
            sim[idx] = -float("inf")
    topk = torch.topk(sim, min(k, len(sim)))
    return list(zip(topk.indices.tolist(), topk.values.tolist()))


@torch.no_grad()
def evaluate_ppl(
    model: AnamnesisModel,
    input_ids: torch.Tensor,
    device: torch.device,
    use_chunked: bool = False,
    target_offset: int = 0,
) -> float:
    """Compute perplexity, optionally only on target positions after target_offset."""
    model.eval()
    input_ids = input_ids.unsqueeze(0).to(device)
    if use_chunked:
        logits = model.forward_chunked(input_ids, chunk_size=512)
    else:
        logits = model(input_ids)

    if isinstance(logits, tuple):
        logits = logits[0]

    # Only compute loss on positions after target_offset
    if target_offset > 0:
        logits = logits[:, target_offset:, :]
        input_ids = input_ids[:, target_offset:]

    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_targets = input_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=0)
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation on Shakespeare")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = CharTokenizer.load(Path(args.tokenizer_path))
    print(f"Vocab size: {tokenizer.vocab_size}")

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    d_model = state_dict["token_embedding.weight"].shape[1]
    vocab_size = state_dict["token_embedding.weight"].shape[0]

    config = AnamnesisConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=8,
        max_seq_len=8192,
        position_encoding_type="sinusoidal",
        engram_layers=(2,),
        engram_num_slots=4096,
        engram_max_ngram=3,
        engram_hash_heads=4,
        attnres_every=0,
    )
    model = AnamnesisModel(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded: d_model={d_model}")

    print("Loading Shakespeare data...")
    train_text = load_dataset("shakespeare", "train")
    valid_text = load_dataset("shakespeare", "valid")

    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    print(f"Train: {len(train_ids):,} tokens, Valid: {len(valid_ids):,} tokens")

    chunk_size = args.chunk_size
    train_chunks = []
    for i in range(0, len(train_ids) - chunk_size, chunk_size):
        train_chunks.append(torch.tensor(train_ids[i : i + chunk_size], dtype=torch.long))
    print(f"Retrieval DB: {len(train_chunks)} chunks of {chunk_size} tokens")

    print("Encoding retrieval database...")
    chunk_embs = encode_chunks(model, train_chunks, device)
    print(f"Chunk embeddings: {chunk_embs.shape}")

    max_eval = min(args.max_eval_tokens, len(valid_ids))
    stride = args.stride

    baseline_losses = []
    rag_losses = []
    oracle_losses = []

    print(f"\nEvaluating {max_eval // stride} windows (stride={stride})...")

    for start in range(0, max_eval - chunk_size * 2, stride):
        target_start = start + chunk_size
        target_end = target_start + chunk_size
        if target_end > len(valid_ids):
            break

        target_ids = torch.tensor(
            valid_ids[target_start:target_end], dtype=torch.long
        )

        # Baseline: model sees only target chunk
        baseline_loss = evaluate_ppl(model, target_ids, device)
        baseline_losses.append(baseline_loss)

        # RAG: retrieve relevant chunk, prepend as context
        query_hidden = model(target_ids.unsqueeze(0).to(device), return_hidden_only=True)
        query_emb = query_hidden.mean(dim=1).squeeze(0)

        results = retrieve_top_k(query_emb, chunk_embs, k=args.top_k)
        if results:
            best_idx, _sim = results[0]
            retrieved_ids = train_chunks[best_idx]
            rag_ids = torch.cat([retrieved_ids, target_ids])
            rag_loss = evaluate_ppl(model, rag_ids, device, use_chunked=True, target_offset=chunk_size)
            rag_losses.append(rag_loss)

        # Oracle: prepend the actual preceding validation chunk
        prev_ids = torch.tensor(
            valid_ids[start : start + chunk_size], dtype=torch.long
        )
        oracle_ids = torch.cat([prev_ids, target_ids])
        oracle_loss = evaluate_ppl(model, oracle_ids, device, use_chunked=True, target_offset=chunk_size)
        oracle_losses.append(oracle_loss)

        if len(baseline_losses) % 10 == 0:
            bl = sum(baseline_losses[-10:]) / min(len(baseline_losses), 10)
            rl = sum(rag_losses[-10:]) / min(len(rag_losses), 10) if rag_losses else 0
            ol = sum(oracle_losses[-10:]) / min(len(oracle_losses), 10) if oracle_losses else 0
            print(
                f"  [{len(baseline_losses):3d}] "
                f"baseline={torch.exp(torch.tensor(bl)):.2f} "
                f"rag={torch.exp(torch.tensor(rl)):.2f} "
                f"oracle={torch.exp(torch.tensor(ol)):.2f}"
            )

    avg_baseline = sum(baseline_losses) / len(baseline_losses)
    avg_rag = sum(rag_losses) / len(rag_losses) if rag_losses else float("inf")
    avg_oracle = sum(oracle_losses) / len(oracle_losses) if oracle_losses else float("inf")

    print("\n=== RESULTS ===")
    print(f"Baseline (target only):      loss={avg_baseline:.4f}  ppl={torch.exp(torch.tensor(avg_baseline)):.2f}")
    print(f"RAG (+ retrieved chunk):      loss={avg_rag:.4f}  ppl={torch.exp(torch.tensor(avg_rag)):.2f}")
    print(f"Oracle (+ preceding chunk):   loss={avg_oracle:.4f}  ppl={torch.exp(torch.tensor(avg_oracle)):.2f}")
    print(f"RAG improvement:              {(avg_baseline - avg_rag) / avg_baseline * 100:+.1f}%")
    print(f"Oracle improvement:           {(avg_baseline - avg_oracle) / avg_baseline * 100:+.1f}%")

    results = {
        "baseline_loss": avg_baseline,
        "baseline_ppl": torch.exp(torch.tensor(avg_baseline)).item(),
        "rag_loss": avg_rag,
        "rag_ppl": torch.exp(torch.tensor(avg_rag)).item(),
        "oracle_loss": avg_oracle,
        "oracle_ppl": torch.exp(torch.tensor(avg_oracle)).item(),
        "rag_improvement_pct": (avg_baseline - avg_rag) / avg_baseline * 100,
        "oracle_improvement_pct": (avg_baseline - avg_oracle) / avg_baseline * 100,
        "num_windows": len(baseline_losses),
        "chunk_size": chunk_size,
        "top_k": args.top_k,
    }
    out_path = Path("experiments/results/real/rag_shakespeare_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
