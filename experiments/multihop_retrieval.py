"""Multi-hop reasoning retrieval task.

Tests whether the retrieval pipeline can support multi-step reasoning
beyond exact needle recall. Places multiple premises scattered in long
context and requires combining them for the answer.

Task design:
  - Place Premise A ("The secret code starts with ALPHA") in chunk i
  - Place Premise B ("The second part is BETA") in chunk j (j ≠ i)
  - Query: "What is the full code?" → Answer: "ALPHA BETA"
  - Model must retrieve BOTH chunks and combine information
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_multihop_data(
    vocab_size: int = 100,
    seq_len: int = 2048,
    n_premises: int = 2,
    chunk_size: int = 512,
    seed: int = 42,
):
    """Generate a multi-hop reasoning sample.

    Returns:
        input_ids: [1, seq_len] token sequence
        answer_tokens: list of answer token ids
        premise_positions: list of (start, end) for each premise
    """
    rng = random.Random(seed)
    n_chunks = seq_len // chunk_size

    # Reserve special tokens
    QUERY_TOKEN = vocab_size - 3  # "What follows?"
    ANSWER_TOKEN = vocab_size - 2  # marker before answer
    SEP_TOKEN = vocab_size - 1  # separates premises

    # Generate premise content
    premise_tokens = []
    for p in range(n_premises):
        # Each premise: random tokens with an embedded fact
        fact_start = rng.randint(4, chunk_size // 2)
        fact_tokens = [rng.randint(0, vocab_size // 2) for _ in range(rng.randint(3, 6))]
        chunk = [rng.randint(0, vocab_size // 4) for _ in range(chunk_size)]
        # Embed fact
        for i, ft in enumerate(fact_tokens):
            if fact_start + i < chunk_size:
                chunk[fact_start + i] = ft
        premise_tokens.append((chunk, fact_tokens))

    # Place premises in different chunks
    chunk_indices = rng.sample(range(n_chunks), n_premises + 1)
    premise_chunks = chunk_indices[:n_premises]
    query_chunk = chunk_indices[n_premises]

    # Build full sequence
    input_ids = [rng.randint(0, vocab_size // 4) for _ in range(seq_len)]

    # Place premises
    for p, (chunk_tokens, _) in enumerate(premise_tokens):
        start = premise_chunks[p] * chunk_size
        input_ids[start:start + chunk_size] = chunk_tokens

    # Place query in last selected chunk
    query_start = query_chunk * chunk_size
    input_ids[query_start] = QUERY_TOKEN
    input_ids[query_start + 1] = SEP_TOKEN
    # Answer: concatenation of all fact tokens
    answer_tokens = []
    for _, fact_tokens in premise_tokens:
        answer_tokens.extend(fact_tokens)
    input_ids[query_start + 2] = ANSWER_TOKEN
    for i, at in enumerate(answer_tokens):
        if query_start + 3 + i < (query_chunk + 1) * chunk_size:
            input_ids[query_start + 3 + i] = at

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    return input_ids, answer_tokens, premise_chunks, query_chunk


def evaluate_multihop(
    model_dir: str,
    d_model: int = 64,
    n_heads: int = 8,
    n_layers: int = 8,
    vocab_size: int = 100,
    seq_len: int = 2048,
    n_premises: int = 2,
    n_eval: int = 50,
    device: str = "mps",
    seed: int = 42,
):
    """Evaluate multi-hop retrieval accuracy."""
    from src.models.anamnesis import AnamnesisConfig, AnamnesisModel

    config = AnamnesisConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_model * 4, max_seq_len=seq_len,
        layerwise_gamma=True, engram_layers=(2,),
        engram_num_slots=8192, engram_use_conv=True,
    )
    model = AnamnesisModel(config).to(device)

    if Path(model_dir).exists():
        ckpt = torch.load(model_dir, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(n_eval):
            input_ids, answer_tokens, _, _ = generate_multihop_data(
                vocab_size, seq_len, n_premises, seed=seed + i
            )
            input_ids = input_ids.to(device)
            logits = model(input_ids)  # [1, seq_len, vocab]

            # Find answer position (QUERY_TOKEN + 2)
            query_pos = (input_ids[0] == vocab_size - 3).nonzero(as_tuple=True)[0]
            if len(query_pos) == 0:
                continue
            answer_start = query_pos[0].item() + 3

            # Check if answer tokens are predicted correctly
            all_correct = True
            for j, ans_tok in enumerate(answer_tokens):
                pos = answer_start + j
                if pos >= seq_len:
                    all_correct = False
                    break
                pred = logits[0, pos].argmax().item()
                if pred != ans_tok:
                    all_correct = False
                    break

            if all_correct:
                correct += 1
            total += 1

    em = correct / total if total > 0 else 0.0
    print(f"Multi-hop ({n_premises} premises, seq_len={seq_len}): EM={em:.3f} ({correct}/{total})")
    return em


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default="")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-premises", type=int, default=2)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Quick test with random data
    input_ids, ans, premise_chunks, query_chunk = generate_multihop_data(
        args.vocab_size, args.seq_len, args.n_premises, seed=args.seed
    )
    print(f"Generated multi-hop sample:")
    print(f"  seq_len={args.seq_len}, n_premises={args.n_premises}")
    print(f"  Premise chunks: {premise_chunks}")
    print(f"  Query chunk: {query_chunk}")
    print(f"  Answer tokens: {ans}")

    if args.model_dir:
        evaluate_multihop(
            args.model_dir, args.d_model, args.n_heads, args.n_layers,
            args.vocab_size, args.seq_len, args.n_premises, args.n_eval,
            args.device, args.seed,
        )
