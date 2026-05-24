# RetNet + Attention Residual + Engram: Autonomous Research Project

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
> "Give an AI agent a real training setup and let it experiment autonomously.
> Modify, train, evaluate, keep or discard, repeat."

## Research Overview

**Small Reasoner + Million-Context Memory Compiler**

The core idea: a small dense model cannot *directly understand* million-token
context any more than a human can hold a million words in working memory.
Instead, we build a **Context Compiler** that preprocesses long context into
structured, typed, verifiable memory states, and a **Small Reasoner** that
retrieves from those states selectively during inference.

### Pipeline: capture → keep → align → margin → decide

```
Long Context
     │
     ▼
┌──────────────┐
│    CAPTURE    │  Context Compiler: chunk → extract entities/definitions/
│   (compiler)  │  constraints → canonical keys → mark critical tokens
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     KEEP      │  Typed Memory: write to the right slot type
│  (typed mem)  │  Engram / Snapshot / TokenCopyBuffer / RetNet state
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    ALIGN      │  Oracle-to-Learned: oracle annotation → prove upper bound
│   (training)  │  → train gate to approximate oracle allocation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    MARGIN     │  Margin Ledger: every memory read must produce margin
│  (ledger)     │  evidence (logit gain). If margin < threshold → discard.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   DECIDE      │  Small Reasoner: RetNet backbone + AttnRes depth reuse
│  (reasoner)   │  O(1) recurrent inference, queries typed memory on demand
└──────────────┘
```

### Typed Memory Architecture

| Memory Type | What it stores | Latency | Capacity | Device |
|-------------|---------------|---------|----------|--------|
| **RetNet State** | Streaming recurrent state | O(1) | Fixed d²×L | GPU |
| **TokenCopyBuffer** | Raw token embeddings for exact recall | O(1) | K slots | GPU |
| **Snapshot** | Reasoning intermediates at milestones | O(1) | K snapshots | GPU |
| **Engram (hot)** | Frequently accessed static knowledge | O(1) hash | M slots | GPU |
| **Engram (cold)** | Full static knowledge base | O(1) hash + IO | Millions of slots | CPU/NVMe |
| **AttnRes** | Cross-layer depth residual | O(1) | Last N layers | GPU |

### Three Progressive Experiments

| Stage | Context | Content | Success Criterion |
|-------|---------|---------|-------------------|
| **Stage 1: 64K** | 64K synthetic context | 1-8 key facts scattered in noise | Verify capture/readout/margin chain works |
| **Stage 2: 256K** | 256K document + citations | Multi-paragraph with cross-references | Verify multi-hop retrieval and citation accuracy |
| **Stage 3: 1M** | 1M tokens, multi-hop reasoning | Nested definitions, chain-of-proof | Verify scalable memory compiler |

### Oracle-to-Learned Approach

For each mechanism, we follow this disciplined path:

1. **Oracle annotation**: manually mark which positions are "critical" in synthetic data
2. **Prove upper bound**: show that with oracle knowledge, the mechanism achieves perfect performance
3. **Train gate to approximate**: replace oracle with a learned gate, train to match oracle allocation
4. **Verify margin**: confirm the learned gate produces sufficient logit margin over baseline

### Architecture Constraints

- Dense architecture — no MoE, no sparse attention, no brute-force scaling
- RetNet provides O(1) recurrent inference baseline
- Every improvement must be structural — smarter mechanisms, not bigger models
- All mechanisms must have formal proof + empirical validation on synthetic tasks

### Research Phases

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1: Mechanism Validation** | Verify each component independently on synthetic tasks | **COMPLETE** |
| **Phase 2: O(1) Recurrent Inference** | Constant-memory inference via recurrent mode | **COMPLETE** |
| **Phase 3: Context Compiler (64K)** | Build capture/readout/margin pipeline for 64K context | **CURRENT** |
| Phase 4: Multi-hop Memory (256K) | Document-level retrieval with cross-references | Planned |
| Phase 5: Million-Context (1M) | Full memory compiler at 1M token scale | Planned |
| Phase 6: Real Tasks | Transfer to real language modeling (TinyStories etc.) | Planned |

### Phase 1 Results (COMPLETE)

| Capability | Task | seq_len | eval_em | Steps |
|-----------|------|---------|---------|-------|
| Long-context recall | needle | 1024 | 1.000 | 400 |
| Static fact memory | alien_static | 64 | 1.000 | 400 |
| Recursive reasoning | XOR | 1024 | 1.000 | 600 |
| Single-step reasoning | xor_final | 128 | 1.000 | 200 |

Key finding: RetNet alone fails needle (eval_em=0.000). TokenCopyBuffer
provides the direct copy path that makes exact recall possible.

### Phase 2 Results (COMPLETE)

Recurrent mode matches parallel mode exactly:

| seq_len | Parallel eval_em | Recurrent eval_em | Max diff |
|---------|-----------------|-------------------|----------|
| 128 | 1.000 | 1.000 | 0.000 |
| 512 | 1.000 | 1.000 | 0.015 |
| 1024 | 1.000 | 1.000 | 0.000 |

Memory is O(d²×L + d×K) — constant regardless of sequence length.

### Additional Data Points

- **Sinusoidal PE**: converges slower than learned PE (0.906 vs 1.000 at step 600, seq_len=1024)
- **Positional keys**: critical for TokenCopyBuffer at seq_len > 256 (0.797→1.000)
- **abs() on residual_scale**: prevents sign reversal from AdamW weight decay

### Phase 3 Progress: Input-Dependent Mechanisms + Length Scaling

**Input-dependent gamma** (like Mamba's selective SSM): γ(x_t) = σ(W_γ x_t + b_γ)

| Seq Len | Baseline eval_em | + input-dep γ | Steps |
|---------|-----------------|---------------|-------|
| 128 | 0.394 | **0.519** | 200 |
| 1024 | 0.588 | **0.600** | 400 |
| 2048 | 0.575 | **0.650** | 400 |
| 4096 | — | **0.500** | 800 |

Consistent improvement at all lengths. Kept as optional enhancement (`--input-dependent-gamma`).

**Discarded mechanisms** (help short-range, hurt long-range due to O(1/D) gradient):
- Output gate: 0.544@128 but 0.512@1024 → discarded
- Value gate (LSTM input gate): unstable at 4096 → discarded

**Key findings:**
- Proof 31: O(1/D) gradient vanishing is fundamental for any recurrent-chain mechanism
- Proof 32: Retention state noise is bounded (geometric series), independent of seq_len
- NaN bug fixed: gated decay mask exp() overflow at seq_len > 2048
- 4096 works with 800 steps (training dynamics, not architecture limit)
- 8192 needs O(s²) memory workaround (chunkwise training)

### Phase 3.5: Real-Data Validation (TinyStories)

Character-level LM on TinyStories (10M train chars, 500K valid, vocab=94, seq_len=512, 5000 steps):

| Config | Params | val_ppl | tok/s |
|--------|--------|---------|-------|
| d128 bare RetNet (512) | 1.67M | 2.99 | 19.9K |
| d128 bare RetNet (1024) | 1.67M | 3.14 | 4.9K |
| **d256 bare RetNet (512)** | **6.6M** | **2.50** | 5.8K |
| d128 full (Engram + milestones + TCB) | 8.08M | 3.14 | 15.8K |
| Engram-only (no milestones/TCB) | 8.02M | 4.41@1K | 20.4K |

### Phase 3.6: Controlled Baselines + Delta Rule (Discarded)

**Needle task with TCB+milestones** (all use ours variant, seq_len=512, 400 steps):

| Variant | eval_em@400 | eval_em=1.0 at step |
|---------|-------------|---------------------|
| ours (baseline) | 1.000 | 140 |
| retnet (bare, no TCB) | 0.000 | — |
| transformer (bare, no TCB) | 0.000 | — |

Key: TCB is essential for needle. Bare RetNet and Transformer can't solve it.

**Delta rule** (S = (γ-β)S + β·k⊗v) — from Gated DeltaNet (ICLR 2025):

| Config | Seq Len | eval_em=1.0 at step | eval_loss@400 |
|--------|---------|---------------------|---------------|
| Baseline | 512 | 140 | — |
| Delta rule | 512 | 160 | — |
| Baseline (batch=1) | 2048 | 180 | 0.091 |
| Delta rule (batch=1) | 2048 | 160 | 0.119 |

**Verdict: DISCARDED.** Delta rule is neutral-to-slightly-worse. More complex, less stable training (oscillating loss), no accuracy advantage. Reason: delta rule helps when retention state IS the memory, but our TCB already handles exact recall. Delta rule solves a problem we've already solved. Proof 35 documents the analysis.

### Phase 3.7: Context Compiler — Oracle Proof-of-Concept

**Oracle position lookup**: train at 512, inject password token logits at answer positions.

| Eval Length | Model Only | Oracle Injection | Train Length |
|-------------|-----------|------------------|-------------|
| 512 | 1.000 | 1.000 | 512 |
| 1024 | crash | **1.000** | 512 |
| 2048 | crash | **1.000** | 512 |
| 4096 | crash | **1.000** | 512 |

**This proves**: the model's last-chunk processing is sufficient for correct prediction IF the right information is injected. The bottleneck is POSITION SELECTION, not retrieval quality.

**ImportanceScorer fails**: trained with BCE loss on password positions, the MLP scorer ranks positions 1-3 at ranks 29, 36, 42 out of 1024. Hidden states at password positions aren't distinctive — password importance comes from the RELATIONSHIP to the query, not intrinsic properties.

**Key insight**: content-based importance scoring (on hidden states) cannot identify password positions because the hidden state at position 3 (just START+3 tokens) looks like any other short prefix. The model needs a TASK-AWARE selection mechanism, not a content-aware one.

**Key finding: Bare RetNet wins on all metrics.** Engram adds 6.3M hash table params that hurt
performance on standard LM. The memory mechanisms (Engram, TCB, milestones) are designed for exact
recall, not statistical next-token prediction. This validates the pipeline design: the Small Reasoner
should be a clean RetNet, and the Memory Compiler should be a separate, task-activated module.

**Mathematical basis:** For LM, every position contributes to loss equally. The RetNet state already
captures sequential dependencies. Adding Engram's gated residual introduces capacity waste — the model
must learn to suppress it. For recall tasks, the needle's information must survive decay → TCB bypasses
decay chain → exact recall. Different tasks need different mechanisms.

## Autonomous Research Loop

### Objective

**Primary metric**: `eval_exact_match` on synthetic tasks — higher is better.
**Secondary metric**: `eval_loss` — lower is better.

These are unambiguous. The agent never has to guess whether an experiment succeeded.

### The Loop (NEVER STOP)

```
FOREVER:
  1. HYPOTHESIZE — form a concrete, falsifiable hypothesis about the architecture
  2. IMPLEMENT — modify src/ or experiments/ to test the hypothesis
  3. TRAIN — run experiments/train_synthetic.py with fixed step budget
  4. EVALUATE — compare eval_exact_match and eval_loss against baseline
  5. DECIDE:
     - IMPROVED → keep changes, commit with results, proceed
     - NEUTRAL + SIMPLER → keep (simplification win), commit
     - NEUTRAL + MORE COMPLEX → discard, git checkout changed files
     - WORSE → discard, git checkout changed files
  6. DISCOVER — analyze failures, identify bugs or architectural weaknesses
  7. FIX — patch bugs, adjust hyperparameters, or redesign components
  8. PROVE — when a mechanism survives empirical testing, write or update
     a formal proof in docs/proofs/ that validates the design
  9. COMMIT & PUSH — git push every meaningful step for traceability
  10. REFLECT — update this file or docs/ with findings, then loop
```

### Experiment Protocol

1. **Fixed step budget**: default 200 steps per experiment run.
   This makes all experiments directly comparable regardless of what changed.
2. **Baseline**: before any change, run the current code to establish the baseline.
3. **Single change per experiment**: modify one thing at a time so causality is clear.
4. **Multiple tasks**: test across needle, xor, xor_final, alien, alien_static.
   A real improvement should hold across tasks, not just one.
5. **Seed discipline**: always use --seed 42 for reproducibility.
   Use --eval-seed 10042 for evaluation consistency.

### Simplicity Criterion

All else being equal, simpler is better:

- A small improvement that adds ugly complexity is **not** worth it.
- Removing something and getting equal or better results is a **simplification win** — always keep.
- A ~0 improvement with much simpler code? **Keep.**
- Adding 20 lines of hacky code for 0.001 eval_loss improvement? **Not worth it.**

### Crash & Failure Protocol

- **OOM / CUDA error**: reduce batch_size or seq_len, retry once. If still fails, discard.
- **NaN / Inf loss**: check gradient norms, reduce learning rate, check for log(0). Fix if trivial, discard if fundamental.
- **Test failure**: fix the bug, re-run. If the test was wrong, update the test.
- **Timeout**: if a single experiment exceeds 10 minutes, kill and treat as failure.

### Git Discipline

Every meaningful step gets a commit:

```
feat: add milestone snapshot readout with RMSNorm gating
fix: engram gate NaN on empty hash table lookup
refactor: extract retention decay init into separate method
test: add contract test for engram hash capacity bounds
proof: formalize gradient dominance of snapshot readout
experiment: needle-512 eval_em improved 0.72→0.89 with snapshots
docs: record decision to defer MoE to phase 2
```

**Push after each commit.** Non-essential files (results, logs, figures, checkpoints)
are gitignored and stay local.

## Project Structure

```
Resources/
├── src/                      # Core implementation (MODIFIABLE)
│   ├── models/retnet_engram.py  # Main model — primary edit target
│   ├── models/recurrent_state.py # Fixed-size O(1) recurrent state
│   ├── layers/               # Retention, AttnRes, Engram, Milestone layers
│   ├── training/             # Training pipelines
│   └── utils/                # Data processing, metrics
├── experiments/              # Experiment runner (MODIFIABLE)
│   ├── train_synthetic.py    # Primary training script
│   ├── configs/              # YAML experiment configurations
│   ├── results/              # [gitignored] Serialized results
│   └── logs/                 # [gitignored] Training logs
├── tests/                    # Test suite — run before every commit
├── docs/
│   ├── proofs/               # Formal mathematical proofs (TRACKED)
│   ├── architecture/         # Architecture design documents
│   └── methodology/          # Research methodology
├── analysis/                 # Post-hoc analysis
│   └── notebooks/            # [gitignored outputs] Jupyter exploration
└── references/               # BibTeX, dataset descriptions
```

## Coding Standards

- **Python 3.10+** with type hints on all public signatures
- **PyTorch** as primary framework
- **Immutability**: never mutate tensors in-place; use functional operations
- **Reproducibility**: seed everything; log all hyperparameters
- **Max file length**: 400 lines — extract modules early
- **Testing**: pytest — run `pytest tests/` before every commit

## Key Hyperparameters (defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| d_model | 64 | small for fast iteration |
| n_heads | 4 | |
| n_layers | 8 | |
| batch_size | 16 | |
| seq_len | 128 | scale up to 512 for pressure tests |
| learning_rate | 3e-4 | |
| steps | 200 | fixed budget per experiment |
| engram_slots | 8192 | hash table size |
| attnres_every | 4 | AttnRes layer frequency |
| branch_init_scale | 1e-4 | residual branch init |

## Key References

- Sun et al. (2023) — Retentive Network: A Successor to Transformer
- Vaswani et al. (2017) — Attention Is All You Need
- He et al. (2016) — Deep Residual Learning
- Tononi & Koch (2015) — Consciousness and Engram
- Katharopoulos et al. (2020) — Transformers are RNNs

## Tools & Libraries

- `torch`, `torch.nn` — Core framework
- `einops` — Tensor operations
- `pytest` — Testing
- `black`, `ruff` — Code formatting
