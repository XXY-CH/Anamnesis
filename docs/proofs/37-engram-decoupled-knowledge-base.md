# Proof 37: Engram as Decoupled Static Knowledge Base

## Problem Statement

The model's LM performance degrades when Engram is trained jointly (ppl 4.4 → 9.5
on Shakespeare). Yet Engram's O(1) hash retrieval is essential for scaling to
million-token context. We need to prove that Engram can be decoupled from the
model: trained separately, stored on disk, and accessed on-demand without
affecting LM quality or consuming GPU memory.

## Architecture

```
┌─────────────────┐          ┌──────────────────┐
│  Small Reasoner │          │  Engram (disk)   │
│  θ_model (GPU)  │  query   │  H[k] → v        │
│  d=64, L=8      │ ───────► │  C=8192 slots    │
│                 │ ◄─────── │  d=64 per slot   │
└─────────────────┘  value   └──────────────────┘
```

## Analysis

### 1. Gradient Decoupling

**Theorem**: If the Engram hash table H is frozen, then ∂L_LM/∂θ_model is
independent of H.

**Proof**: The LM loss L = -Σ log P(x_t | x_{<t}; θ). When H is frozen,
readout(h_t, H) is a fixed function of h_t. No gradient term involves ∂H. ∎

**Corollary**: The LM can be trained to convergence WITHOUT the Engram.
The Engram is trained separately on the frozen LM's representations (exactly
the pipeline approach proven in Proof 36).

### 2. Memory Analysis

| Scale | Slots | d | GPU Memory (on disk) |
|-------|-------|---|----------------------|
| Small | 8K | 64 | 2 MB (0 persistent) |
| Medium | 1M | 128 | 512 MB (0 persistent) |
| Large | 100M | 256 | 100 GB (0 persistent) |

Disk-stored Engram: O(1) per query, ~0.1ms NVMe latency. During training,
cache accessed slots in LRU buffer (~1000 slots = 0.25 MB).

### 3. Separation of Memorization and Reasoning

Model parameters encode **computation** (reasoning). Engram slots encode
**data** (facts). These are orthogonal:

- More model capacity → better reasoning, NOT better memorization
  (proven: Transformer solves needle@512, RetNet doesn't — same params)
- More engram capacity → better memorization, NOT better reasoning
  (proven: TCB enables recall but hurts ppl from 4.4 to 9.5)

**Key insight**: A small model with large external memory achieves:
- Same reasoning as small model (fast O(1) recurrent inference)
- Same memorization as large model (unlimited disk storage)

## Implications

1. Engram should be kept as external module, not removed entirely
2. Two-phase training: LM first (no Engram), then Engram on frozen LM
3. Engram training strategy to be determined separately

## Limitations

1. Hash collisions at high utilization (mitigate: multi-head hashing)
2. Cold start: Engram empty at inference start (needs pre-population)
3. Within-chunk position identification still open (see Proof 36)
