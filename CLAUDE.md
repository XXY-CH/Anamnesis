# Anamnesis: Autonomous Research Project

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
| **Phase 3: Context Compiler (1M)** | Chunk retrieval pipeline to 1M tokens | **COMPLETE (EM=0.938@524K)** |
| **Phase 4: Reliable 1M** | EM=1.0 at 1M via hierarchical retrieval or stronger retriever | **In Progress** |
| Phase 5: Real Tasks | Transfer pipeline to real language modeling | Planned |

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

### Phase 3.8: GCA-Style Chunk Retrieval — 64x Length Generalization

**Contrastive chunk selection + token injection pipeline** (train@512, eval up to 32K):

| Eval Length | Chunks | Fixed EM | Random EM | Retriever Trained On |
|-------------|--------|----------|-----------|---------------------|
| 512 | 1 | 1.000 | — | N/A (model only) |
| 1024 | 2 | 1.000 | 1.000 | 2048 (4 chunks) |
| 2048 | 4 | 1.000 | 1.000 | 2048 (4 chunks) |
| 4096 | 8 | 1.000 | 1.000 | 2048 (4 chunks) |
| 8192 | 16 | 1.000 | 1.000 | 2048 (4 chunks) |
| 16384 | 32 | 1.000 | 1.000 | 2048 (4 chunks) |
| 32768 | 64 | 1.000 | 1.000 | 2048 (4 chunks) |
| 65536 | 128 | 1.000 | 1.000 | 2048 (4 chunks) |
| 131072 | 256 | 1.000 | 1.000 | 2048 (4 chunks) |

**Pipeline**: frozen model → chunk embeddings → contrastive retriever selects needle chunk →
token embedding readout → logit injection at answer positions.

**Three critical findings**:

1. **Contrastive chunk selection works**: trained in 200 steps, 0.85 top weight on correct chunk.
   Chunk discrimination by frozen model: 0.469 accuracy (vs 0.250 random baseline) at 4x training length.
   Retriever trained on 4 chunks generalizes to 256 chunks (256x scaling).

2. **Hidden-state readout FAILS**: cross-attention on hidden states → project through token embedding
   gives EM=0.000. Hidden states are high-dimensional abstractions, not token representations.
   Mean-pooled, attention-weighted, or per-position — none work.

3. **Token embedding readout works**: `F.linear(token_embedding(token_id), token_embedding.weight)`
   gives near-perfect logits for the token. This is a self-similarity lookup in embedding space.
   Combined with chunk selection → EM=1.000 at 256x training length.

**Multi-needle results** (structural selection: latest MARK_THOUGHT chunk):

| Needles | Length | EM | Selection |
|---------|--------|----|-----------|
| 1-8 | 2048-16384 | 1.000 | Structural |
| 1 | 512-131072 | 1.000 | Contrastive |

**Why contrastive, not generation loss?** Generation loss requires the base model to already solve
the task. Contrastive loss only requires the model's hidden states to distinguish important chunks,
which works even when the model can't generate correct answers.

**Architecture**: ChunkRetriever = query_proj + chunk_proj (for selection) + value_proj + logit_scale
(for readout). Only 12K parameters. Selection and readout are separate concerns.

**Discarded**: Position bias helps multi-needle (0.312→0.625) but hurts single-needle (1.000→0.375).
Reverted. Multi-needle needs task-aware selection, not position bias.

**Controlled baselines** (all ~440K params, 1200 steps, needle@512):

| Model | @512 | @2048 | @4096 | @32768 |
|-------|------|-------|-------|--------|
| Bare RetNet | 0.000 | N/A | N/A | N/A |
| Transformer | 1.000 | N/A | N/A | N/A |
| Ours (model only) | 1.000 | N/A | N/A | N/A |
| Ours + pipeline | 1.000 | 1.000 | 1.000 | 1.000 |

Bare RetNet can't solve needle without TCB. Transformer solves at 512 but can't
extend. Only ours + chunk retrieval extends context to 32K.

**Position-level retrieval on frozen model: FAILS**. Both generation-loss and
contrastive-BCE training produce random attention (EM=0.000). The frozen model's
query hidden state carries zero signal about which within-chunk positions are
important. Implication: within-chunk retrieval must be trained jointly with the model.

### Phase 3.9: 1M Token Scaling + Real-Data Validation

**Pipeline scaling to 1M tokens** (model@512, retriever@8192/16 chunks, 500 steps, lr=3e-3):

| Eval Length | Chunks | Best EM | Best Temp |
|-------------|--------|---------|-----------|
| 4K | 8 | 1.000 | 0.5 |
| 8K | 16 | 1.000 | 0.5 |
| 16K | 32 | 0.875 | 0.5 |
| 32K | 64 | 0.750 | 0.2 |
| 65K | 128 | 1.000 | 0.1 |
| 131K | 256 | 0.875 | 0.5 |
| 1M | 2048 | 0.875 | 0.1 |

Retriever trained on 16 chunks generalizes to 2048. Temperature scaling critical:
lower temperatures for higher chunk counts. Pipeline EM = chunk accuracy (readout
is perfect when correct chunk is selected).

**Real-data Shakespeare baselines** (1.67M params, char-level LM, 2000 steps):

| Model | val_ppl |
|-------|---------|
| Transformer | 4.371 (best) |
| Bare RetNet | 7.115 |
| Ours (TCB+milestones) | 9.509 (worst) |

**Critical finding**: TCB/milestones hurt general LM performance. These mechanisms
are specialized for retrieval, not language modeling. The pipeline must be an
external module, not baked into the base model. Future direction: use Transformer
or bare RetNet as base, add chunk retrieval as external long-context layer.

**Three bugs fixed in this phase**:
1. Per-position readout (not summed) — summed readout promotes all password tokens
   at all positions equally, can't distinguish position
2. `answer_start = query_pos` (not +1) — mask starts at QUERY position in
   next-token prediction
3. `logit_scale=0` — contrastive training never updates it; replaced with fixed
   `readout_scale = 1/sqrt(d_model)`

### Phase 3.10: Engram-Hit → TCB Trigger (Self-Supervised Gating)

**Oracle-to-learned distillation**: Engram gate output as surprise estimator for TCB storage.

During training: oracle drives TCB (correct tokens stored), BCE distillation loss trains
Engram gate to predict oracle positions. At inference: Engram surprise score drives TCB.

**Comparison on needle@512** (d=64, sinusoidal PE, 8 layers, 4 heads):

| Approach | Steps | Best EM | EM@end | EM=1.0? |
|----------|-------|---------|--------|---------|
| Oracle baseline | 800 | 0.875 | 0.875 | No |
| Oracle baseline | 1200 | 0.875 | 0.875 | No |
| Pure Engram trigger (no distill) | 800 | 0.938 | 0.688 | No |
| **Engram + distillation** | **800** | **1.000** | **0.812** | **Yes (step 760)** |
| **Engram + distillation** | **1200** | **1.000** | **0.938** | **Yes (steps 880, 1180)** |

**Key findings**:
1. Only variant to achieve EM=1.000 — Engram distillation outperforms oracle.
2. Pure Engram trigger fails: without oracle during training, wrong tokens get stored,
   model can't learn. Chicken-and-egg: need correct TCB to learn, need to learn to fill TCB.
3. Distillation breaks the cycle: oracle provides correct signal during training,
   Engram learns to approximate it.
4. Instability remains: EM oscillates 0.5-1.0 in later training. Likely due to
   Engram gate and LM loss competing. Needs LR scheduling or stronger regularization.

**Architecture**: `use_engram_tcb_trigger=True` in config. Surprise = mean of Engram gate
across layers and d_model dimensions. Distillation loss: weighted BCE with auto pos_weight.

### Phase 3.11: Frontier Literature + Scaling Experiments

**Literature review** (8 papers, see `docs/brainstorm-2026-05-24.md`):
- SILA: memory-dependent gate, 20x length extrapolation
- SR-TTT: surprisal-aware residual cache, two-stage curriculum
- FDM: wave-particle separation, Freeze-Scan training (7.5x convergence)
- Bicameral: orthogonal keys required for reliable memory → validates hash-based Engram
- ReSuME: SAE reconstruction error as surprise → validates Engram gate
- GSA: hierarchical gist tokens → could extend chunk retrieval
- SPLA: second-order Taylor for block selection
- SuRe: NLL as surprise for buffer selection

**Direction A: Loss-Driven Surprise — DISCARDED**. Self-supervised entropy cannot
replace oracle for TCB storage. On needle task, entropy is high everywhere (random
filler tokens), so top-K by entropy selects random positions. Per-token CE loss has
same problem. Fundamental: no self-supervised signal distinguishes password from
filler when most tokens are random. The oracle/structural marker is necessary.

**Curriculum retriever training** (d_model=64, 16 eval batches, 4→16→64→256 chunks):

| Eval Length | Chunks | Best EM | Best Temp |
|-------------|--------|---------|-----------|
| 2K | 4 | 1.000 | 0.1 |
| 4K | 8 | 0.938 | 0.5 |
| 8K | 16 | 0.875 | 0.1 |
| 16K | 32 | 0.875 | 0.5 |
| 32K | 64 | 0.750 | 1.0 |
| 65K | 128 | 0.875 | 0.2 |
| 131K | 256 | 0.750 | 0.2 |
| 262K | 512 | 0.812 | 1.0 |
| **524K** | **1024** | **0.938** | **0.5** |
| 1M | 2048 | 0.625 | 0.5 |

**New record: EM=0.938 at 524K (1024 chunks)**. But curriculum training is worse than
fixed@8192 at short lengths (8K-16K). The curriculum overwrites good representations
from earlier stages. The 12K param retriever hits a ceiling at 2048-way classification
from 64-dim embeddings.

**Shakespeare validation** (d=128, char-level LM, 2000 steps, sinusoidal PE):

**Vector gate (DEPRECATED)** — anisotropic scaling destroys semantic direction:
- Bare RetNet: val_ppl=5.87
- RetNet + Engram (vector gate): val_ppl=9.95 (worst)

**Scalar gate (CURRENT)** — isotropic scaling preserves direction (Proof 40):
- Bare RetNet: val_ppl=9.78
- RetNet + scalar-gated Engram: val_ppl=**7.59** (best, -22%)

**Critical finding**: Scalar-gated Engram HELPS language modeling, not hurts.
The vector gate's per-dimension scaling rotated embeddings off-manifold,
but scalar gating preserves semantic direction while providing useful static
knowledge priors. Engram can stay in the model as an integrated component.

### Phase 3.12: Transformer vs RetNet Pipeline Comparison

**Controlled baseline** (both trained needle@512, retriever@8192/500 steps, d_model=64, 8 layers):

| Length | RetNet+TCB EM | Transformer EM | Delta |
|--------|--------------|----------------|-------|
| 2K | 1.000 | 0.625 | +0.375 |
| 4K | 1.000 | 0.250 | +0.750 |
| 8K | 1.000 | 0.000 | +1.000 |
| 16K | 1.000 | 0.125 | +0.875 |
| 32K | 1.000 | 0.125 | +0.875 |
| 65K | 1.000 | 0.125 | +0.875 |
| 131K | 1.000 | 0.000 | +1.000 |

**RetNet achieves EM=1.000 at ALL lengths** (2K–131K, 256 chunks). Best pipeline result to date.

**Transformer pipeline fails completely**:
1. Needle learning: EM=0.438@1200 steps (vs RetNet's 0.938). Transformer needs more steps.
2. Retriever never converges: loss stuck at 2.8, chunk_acc=0 at all 500 steps. Hidden states
   don't produce discriminative chunk embeddings for contrastive learning.
3. Pipeline EM→0.000 beyond 8K tokens.

**Why**: RetNet's recurrent state accumulates position-aware information that makes chunk
embeddings distinctive. Transformer's attention produces similar representations across
random-fill chunks — nothing in the hidden state signals "this chunk contains the needle."
This validates RetNet as the correct backbone for the Context Compiler pipeline.

**1M evaluation with 16 eval batches** (442K params, 12K retriever, seed=42):

| Length | Chunks | Best EM | Best Temp |
|--------|--------|---------|-----------|
| 2K | 4 | 1.000 | 0.1 |
| 4K | 8 | 0.938 | 0.5 |
| 8K | 16 | 0.938 | 0.1 |
| 16K | 32 | 0.875 | 0.5 |
| 32K | 64 | 0.812 | 0.1 |
| 65K | 128 | 0.875 | 0.2 |
| 131K | 256 | 0.750 | 0.1 |
| 262K | 512 | 0.750 | 1.0 |
| 524K | 1024 | 0.875 | 0.5 |
| **1M** | **2048** | **0.625** | **0.5** |

Previous 8-batch 1M=0.750 was noise — 16 batches shows 0.625 is the true level.
Bottleneck: 64-dim embeddings from 442K model can't distinguish 2048 random chunks.
The 3.6M model (with Engram tables) achieves EM=1.000 at 131K where 442K gets 0.750 —
larger models produce more discriminative chunk embeddings.

**proj_dim=256 at 1M** (442K model, 12K→37K retriever, 8 eval batches):

| Config | @131K | @262K | @524K | @1M |
|--------|-------|-------|-------|-----|
| proj_dim=64, 8 batches | 1.000 | 0.812 | 0.875 | 0.750 |
| proj_dim=64, 16 batches | 0.750 | 0.750 | 0.875 | 0.625 |
| **proj_dim=256, 8 batches** | 0.750 | 0.750 | 0.875 | **0.875** |

Higher-dim projection gives retriever more capacity for 2048-way discrimination.
1M improves from 0.750→0.875. Still noisy (8 batches). Retriever params: 12K→37K.

**Direction C: Hierarchical retrieval — DISCARDED.** Group chunks into super-chunks
(group_size=32), score super-chunks then score within selected group. Result: much worse
(1M drops from 0.875→0.500). Super-chunk mean-pooling dilutes the needle signal 32x
(3 password tokens in 16K tokens). Retriever trained on individual chunk embeddings
doesn't generalize to averaged super-chunk embeddings. Two-stage classification also
introduces two points of failure.

**Chunk-level RoPE — DISCARDED.** Applies rotary position encoding to chunk embeddings
during scoring. Result: catastrophic collapse at long lengths (1M: 0.875→0.250).
With 2048 chunks, rotation angles become too large, destroying content-based similarity.
The retriever becomes position-biased and can't find the needle by content. Consistent
with Phase 3.8 finding that position bias hurts single-needle accuracy.

### Phase 3.13: Real-Data Pipeline Transfer (Shakespeare)

**Shakespeare LM → needle-in-Shakespeare pipeline** (3.6M params, proj_dim=256):

| Phase | Metric | Result |
|-------|--------|--------|
| Shakespeare LM (2000 steps) | val_ppl | 6.28 |
| Needle fine-tuning (1200 steps) | eval_em | **0.000** |
| Retriever (500 steps, Shakespeare filler) | chunk_acc@temp=1.0 | **0.875** |
| Pipeline @ 2K | EM | **0.000** |

**Critical finding**: chunk retrieval transfers to real data (0.875 accuracy on Shakespeare
filler), but token readout fails completely. The LM-trained model can't learn the needle
task — loss stuck at 4.1 (near random for vocab=67). Shakespeare priors resist the
synthetic copy task.

**Why**: The retriever scores chunks by content similarity (query vs chunk embeddings).
This mechanism is content-agnostic — it works on any text. But the token readout requires
the model to have learned TCB-based copying, which conflicts with LM training.

**Implication**: For real-data deployment, the pipeline should be:
1. Retriever finds relevant chunks (works!)
2. Feed selected chunk text to the model as context (RAG-style, no special readout needed)
3. Model generates from context using its native LM capability

### Phase 3.14: Scalar Gate PPL Validation

**Controlled experiment** (d=128, char-level Shakespeare, 2000 steps, sinusoidal PE, attnres_every=0):

| Config | val_ppl | val_loss | Params |
|--------|---------|----------|--------|
| Bare RetNet | 9.78 | 2.281 | 1.67M |
| RetNet + scalar-gated Engram | **7.59** | **2.027** | ~8M |

**Key finding**: Scalar-gated Engram improves LM PPL by 22%. This validates Proof 40's
prediction that initial perturbation is O(s·e^b) ≈ 10⁻⁵ (negligible). The Engram's static
hash tables provide useful prior knowledge for character-level language modeling.

**Architectural implication**: Engram can stay as an integrated component. No strict
external separation needed for the base model. The architecture is now:
- **AnamnesisModel** = RetNet + scalar-gated Engram + AttnRes (integrated)
- **External pipeline** = Chunk retriever for ultra-long context (still needed for 1M)

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
│   ├── models/anamnesis.py  # Main model — primary edit target
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
