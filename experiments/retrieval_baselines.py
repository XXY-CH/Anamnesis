#!/usr/bin/env python3
"""P0: BM25 / exact-match / random baselines for chunk retrieval.

Tests whether 1M EM=1.000 is trivially achievable by lexical matching.
If BM25 also gets EM=1.000, the retrieval task is too easy and we must
acknowledge this as a limitation.
"""
from __future__ import annotations
import sys, random, json
from pathlib import Path
from collections import Counter
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Replicate the synthetic needle task setup
VOCAB_SIZE = 67  # char-level Shakespeare
QUERY_TOKEN = 1  # special query token
ANSWER_TOKEN = 2  # special answer token
CHUNK_SIZE = 512

def generate_haystack(n_chunks: int, chunk_size: int, needle_len: int = 8,
                      seed: int = 42) -> tuple[list[list[int]], int, list[int]]:
    """Generate synthetic haystack with one needle embedded."""
    rng = random.Random(seed)
    needle = [ANSWER_TOKEN] + [rng.randint(3, VOCAB_SIZE - 1) for _ in range(needle_len)]

    chunks = []
    needle_chunk_idx = rng.randint(0, n_chunks - 1)

    for i in range(n_chunks):
        if i == needle_chunk_idx:
            pos = rng.randint(0, chunk_size - len(needle) - 1)
            chunk = [rng.randint(3, VOCAB_SIZE - 1) for _ in range(chunk_size)]
            for j, t in enumerate(needle):
                chunk[pos + j] = t
            chunks.append(chunk)
        else:
            chunks.append([rng.randint(3, VOCAB_SIZE - 1) for _ in range(chunk_size)])

    return chunks, needle_chunk_idx, needle


def exact_match_baseline(chunks, needle):
    """Find chunk containing the exact needle token sequence."""
    for i, chunk in enumerate(chunks):
        for j in range(len(chunk) - len(needle) + 1):
            if chunk[j:j + len(needle)] == needle:
                return i
    return -1


def bm25_baseline(chunks, needle, k1=1.5, b=0.75):
    """BM25 scoring: find chunk most similar to needle query."""
    n_chunks = len(chunks)
    df = Counter()
    chunk_tf = []

    for chunk in chunks:
        tf = Counter(chunk)
        chunk_tf.append(tf)
        for token in set(chunk):
            df[token] += 1

    scores = []
    for i in range(n_chunks):
        score = 0.0
        chunk_len = len(chunks[i])
        avg_len = CHUNK_SIZE
        for token in needle:
            if token in chunk_tf[i]:
                tf = chunk_tf[i][token]
                idf = math.log((n_chunks - df[token] + 0.5) / (df[token] + 0.5) + 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * chunk_len / avg_len))
                score += idf * tf_norm
        scores.append(score)

    return scores.index(max(scores))


def needle_token_baseline(chunks, answer_token=ANSWER_TOKEN):
    """Simply find chunk containing the ANSWER token."""
    for i, chunk in enumerate(chunks):
        if answer_token in chunk:
            return i
    return -1


def run_experiment(n_chunks_list, needle_len=8):
    """Run all baselines at different context lengths."""
    results = []

    for n_chunks in n_chunks_list:
        total_tokens = n_chunks * CHUNK_SIZE
        print(f"\n=== {n_chunks} chunks ({total_tokens:,} tokens) ===")

        for seed in range(3):
            chunks, true_idx, needle = generate_haystack(n_chunks, CHUNK_SIZE, needle_len, seed=seed)

            rng = random.Random(seed)
            pred = rng.randint(0, n_chunks - 1)
            random_em = 1.0 if pred == true_idx else 0.0

            em_pred = exact_match_baseline(chunks, needle)
            exact_em = 1.0 if em_pred == true_idx else 0.0

            bm25_pred = bm25_baseline(chunks, needle)
            bm25_em = 1.0 if bm25_pred == true_idx else 0.0

            nt_pred = needle_token_baseline(chunks)
            needle_em = 1.0 if nt_pred == true_idx else 0.0

            r = {
                "n_chunks": n_chunks,
                "total_tokens": total_tokens,
                "seed": seed,
                "random_em": random_em,
                "exact_match_em": exact_em,
                "bm25_em": bm25_em,
                "needle_token_em": needle_em,
            }
            results.append(r)
            print(f"  seed={seed}: random={random_em:.3f} exact={exact_em:.3f} "
                  f"bm25={bm25_em:.3f} needle_token={needle_em:.3f}")

    print("\n=== AGGREGATE ===")
    for nc in n_chunks_list:
        subset = [r for r in results if r["n_chunks"] == nc]
        print(f"  {nc} chunks ({nc * CHUNK_SIZE:,} tok): "
              f"random={sum(r['random_em'] for r in subset)/len(subset):.3f} "
              f"exact={sum(r['exact_match_em'] for r in subset)/len(subset):.3f} "
              f"bm25={sum(r['bm25_em'] for r in subset)/len(subset):.3f} "
              f"needle_tok={sum(r['needle_token_em'] for r in subset)/len(subset):.3f}")

    out = Path(ROOT) / "experiments" / "results" / "retrieval_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    return results


if __name__ == "__main__":
    n_chunks_list = [16, 64, 256, 512, 1024, 2048]
    run_experiment(n_chunks_list)
