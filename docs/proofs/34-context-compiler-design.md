# Proof 34: Context Compiler — External Memory for Small Reasoners

## Problem Statement

A small RetNet with d=256, L=8 layers processes sequences in O(1) recurrent memory,
but the retention state S_t decays exponentially: information from position s is
weighted by γ^(t-s). For sequences longer than ~1/|log γ|, old information is lost.

We need an external memory module that:
1. Processes long context into a fixed-size, queryable representation
2. Does NOT degrade the Small Reasoner's LM capability when attached
3. Can be queried efficiently during inference

## Information-Theoretic Setup

**Definition**: Let X = (x_1, ..., x_N) be a sequence of N tokens. The **retention
information** at position t is:

$$I_{\text{ret}}(t) = \{S_t^{(\ell)} : \ell = 1, \ldots, L\}$$

where $S_t^{(\ell)}$ is the recurrent state at layer $\ell$.

**Definition**: The **decayed information** at position t about position s is:

$$D(t, s) = \prod_{r=s+1}^{t} \gamma_r$$

For fixed γ, $D(t,s) = \gamma^{t-s}$. For γ=0.97, after 100 positions: D ≈ 0.05.
After 200 positions: D ≈ 0.002. Effectively zero.

**Claim**: For N >> 1/|log γ|, the retention state loses most information about
early positions. The Context Compiler must recover this.

## Context Compiler Architecture

### Principle: Separate Processing from Reasoning

The Context Compiler is a **pre-processing module** that runs ONCE on the input
before the Small Reasoner starts generating. It is NOT part of the LM loss computation.

```
Input (64K tokens)
    │
    ▼
┌──────────────────┐
│ Context Compiler │  (can be slow — preprocessing)
│ - Chunk into 512 │
│ - Run RetNet     │
│ - Extract keys   │
│ - Store in hash  │
└────────┬─────────┘
         │
         ▼
Compiled Memory (fixed size: M entries)
    │
    ▼
┌──────────────────┐
│  Small Reasoner  │  (fast — O(1) recurrent inference)
│  - RetNet d=256  │
│  - Queries memory│  ← on-demand, not always
│  - Generates     │
└──────────────────┘
```

### Mathematical Formulation

**Compiled Memory** M is a set of (key, value) pairs:

$$M = \{(k_i, v_i)\}_{i=1}^{K}$$

where K is the memory budget (e.g., K=256 entries).

**Context Compiler** function: C: R^{N×d} → R^{K×2d}

The compiler processes the full sequence and selects K key-value pairs to store.

**Query mechanism**: At position t, the Small Reasoner generates a query q_t
and retrieves from memory:

$$r_t = \sum_{i=1}^{K} \text{softmax}(q_t^T k_i) \cdot v_i$$

This is standard attention over K entries — O(K) cost, independent of N.

### What to Store: The Selection Problem

The core question: which K entries to store from N positions?

**Option 1: Uniform sampling** — Store every N/K-th position.
Simple but wasteful — most tokens are not information-dense.

**Option 2: Importance sampling** — Score each position by information content,
store top-K.

**Option 3: Milestone-based** — Use the existing milestone mechanism to identify
structurally important positions.

**Option 4: Learned selection** — Train a small network to predict which positions
are worth storing.

We analyze each:

### Analysis of Selection Strategies

**Theorem (Uniform Sampling Bound)**: For uniform sampling with K entries from N
positions, the expected maximum gap between consecutive stored positions is:

$$E[\text{max gap}] = \frac{N}{K} \cdot H_K / K \approx \frac{N \ln K}{K^2}$$

For N=64000, K=256: max gap ≈ 25000/65536 ≈ 280 tokens. With retention decay
γ=0.97, information from 280 positions ago is weighted by 0.97^280 ≈ 0.0002.
**Most information in the gap is lost.**

**Theorem (Milestone Bound)**: If milestones mark every "important" token (defined
as tokens whose correct prediction requires information from >D positions ago),
then storing milestone-adjacent tokens ensures:

$$P[\text{recall failure}] \leq 1 - \frac{K_{\text{milestones}}}{K_{\text{total important}}}$$

This is the oracle upper bound — with perfect milestone detection, we only fail
when there are more important tokens than memory slots.

**Theorem (Optimal Selection)**: The optimal selection minimizes the expected loss
increase from information loss:

$$\min_{S \subset [N], |S|=K} E\left[\sum_{t \notin S} -\log P(x_t | x_{<t}, \{x_s : s \in S\})\right]$$

This is NP-hard in general (subset selection), but can be approximated greedily.

### Proposed Design: Three-Stage Compiler

**Stage 1: Chunk processing** (O(N) time)
- Process input in chunks of C=512 tokens using RetNet's chunkwise mode
- Each chunk produces a recurrent state and a set of candidate entries

**Stage 2: Selection** (O(N/K) time)
- Score each chunk's candidates using an importance scorer
- Keep top-K entries across all chunks

**Stage 3: Structuring** (O(K) time)
- Organize selected entries into a structured memory:
  - **Key**: position-encoded token embedding (enables positional lookup)
  - **Value**: hidden state at that position (richer than raw embedding)
  - **Metadata**: chunk index, local position, importance score

### Inference Interface

During inference, the Small Reasoner:
1. Runs normally using its recurrent state
2. At each step, optionally queries the compiled memory:
   - Generate query q_t from current hidden state
   - Compute attention over K memory entries
   - Add gated residual to hidden state

**Critical**: The query is gated — the model learns WHEN to query memory.
For standard LM on short sequences, the gate stays closed (no memory access).
For recall tasks on long sequences, the gate opens when external info is needed.

This satisfies Proof 33's criterion: the memory provides information NOT in the
retention state (it bypasses the decay chain), and it's only activated when needed.

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Compiler (preprocessing) | O(N) | O(K) |
| Memory query (per step) | O(K·d) | O(K·d) |
| Small Reasoner (per step) | O(L·d²) | O(L·d²) |

For N=64000, K=256, d=256, L=8:
- Compiler: O(64K × 256² × 8) ≈ 0.3 TFLOPS (seconds on GPU)
- Per-step query: O(256 × 256) = 65K multiply-adds (negligible)
- Per-step reasoner: O(8 × 256²) ≈ 0.5M multiply-adds

**Total per token during inference: O(L·d² + K·d)** — still O(1) in sequence length.

## Next Steps

1. Implement the Context Compiler as a separate module in `src/memory/`
2. Test on synthetic needle-in-haystack at 4K+ with external memory
3. Compare: model with integrated Engram vs model with external Context Compiler
4. Measure: does external memory help recall WITHOUT hurting LM quality?
