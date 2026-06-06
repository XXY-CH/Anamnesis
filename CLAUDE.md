# Anamnesis: Autonomous Research Project

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
> "Give an AI agent a real training setup and let it experiment autonomously.
> Modify, train, evaluate, keep or discard, repeat."

## Final Architecture

**Anamnesis = RetNet (8h + layerwise gamma) + optional scalar-gated Engram + external chunk retrieval**

Three separable components, each independently validated:

| Component | What | When to enable | Cost |
|-----------|------|---------------|------|
| **Layerwise gamma** | Depth-dependent retention decay schedule | Always | 0 extra params |
| **Scalar-gated Engram** | Hash-based N-gram lookup tables | Char-level/small-vocab; BPE only at lr=1e-3 | ~6M hash params |
| **Chunk retrieval** | Contrastive chunk selection + token readout | Ultra-long context (≤1M tokens) | 12K retriever params |

**Design rules:**
- Engram helps BPE at lr=1e-3 in lucky seeds, but high variance makes it unreliable (Phase 5.34)
- Engram can hurt at large model widths on repetitive data (Phase 5.32)
- Layerwise gamma is the universal benefit — works on all datasets, all sizes

## Key Results

### Language Modeling (char-level, lr=1e-3, 5K steps)

| Model | d | Shakespeare | TinyStories | WikiText-2 | Mean Δ vs TF |
|-------|---|-------------|-------------|------------|-------------|
| **Anamnesis** | **128** | **3.12±0.11** | **2.82±0.03** | **4.59±0.33** | **-26.7%** |
| Transformer | 128 | 3.94±0.07 | 5.22 | 5.29±0.83 | — |
| **Anamnesis** | **256** | **2.42±0.02** | 2.43 | 4.48 | **-14.0%** |
| Transformer | 256 | 3.38±0.07 | **2.25** | 4.75 | — |

Anamnesis wins 5/6 dataset×scale comparisons. The single loss (TinyStories d=256) is caused by
Engram: bare RetNet 8h+layerwise achieves PPL=2.20, beating Transformer (2.25).

### Fair Ablation (Shakespeare char-level, lr=1e-3, 5K steps, seed=42)

| Component | d=128 PPL | d=256 PPL | d=128 share | d=256 share |
|-----------|----------|----------|------------|------------|
| Baseline (4h) | 7.99 | 5.57 | — | — |
| + Layerwise γ + 8h | 4.11 | 4.31 | 60% | 59% |
| + Engram | **3.02** | **3.44** | 40% | 41% |
| Transformer | 3.94 | 3.38 | — | — |

Contribution flips with model size: Engram dominant at d=128, layerwise dominant at d=256.

### 1M Token Retrieval

EM=1.000 at 1M tokens (2048 chunks) with Engram-enhanced embeddings.
Transformer pipeline fails beyond 8K (hidden states non-discriminative).

### BPE Results (WikiText-2, d=128, 5K steps)

| Model | lr=3e-4 | lr=1e-3 (s42) | lr=1e-3 (s100) | Mean ± std |
|-------|---------|---------------|----------------|-----------|
| **Transformer** | 122.67 | 72.37 | **67.67** | **70.02 ± 3.3** |
| Anamnesis (Engram 8K) | 183.84 | **67.49** | 83.63 | 75.56 ± 11.4 |
| Bare RetNet 8h+lw | 171.23 | 89.32 | — | — |

**BPE is seed-dependent**: Anamnesis wins at s42 (67.49 vs 72.37) but loses at s100 (83.63 vs 67.67).
Engram introduces high variance on BPE (std=11.4). Bare RetNet (std=2.7) and Transformer (std=3.3) are stable.
**Variance source: Engram gate initialization** — low BPE SNR makes gate learning seed-dependent (Proof 48).
Anamnesis reliably wins on char-level (5/6 comparisons). BPE: disable Engram for stable results.

### 10K Steps Scaling (Shakespeare char-level, lr=1e-3, seed=42)

| Model | 5K PPL | 10K PPL | Improvement | Δ at 10K |
|-------|--------|---------|-------------|----------|
| **Anamnesis** | 3.12 | **1.88** | **-39.7%** | **-35.8%** |
| Transformer | 3.94 | 2.93 | -25.6% | — |

Anamnesis improves 55% faster. Advantage grows: -20.8% (5K) → -35.8% (10K).
Confirms Engram learning acceleration (Proof 31) compounds over training.

10K ablation: Bare RetNet 10K PPL=2.62 (from 4.11, -36.3%).
RetNet+layerwise flips from -4.3% vs TF at 5K to +10.6% at 10K — needs more training to shine.
Engram contribution grows: -24.1% (5K) → -28.2% (10K).

TinyStories 10K: Anamnesis 2.38, Transformer 2.07 — **advantage REVERSES** (+14.9%).
5K advantage (-46%) was due to Transformer undertraining. Transformer improves -60.3% (5K→10K) vs Anamnesis -15.6%.
On diverse data (Shakespeare), Anamnesis advantage grows. On repetitive data (TinyStories), Transformer catches up.
"Advantage scales with repetitiveness" is a 5K artifact — Transformer needs more training on repetitive patterns.

WikiText-2 10K: Anamnesis 3.94, Transformer 4.00 (-1.5%, barely wins).
Complete 10K pattern: Shakespeare GROWS (-35.8%), WikiText-2 SHRINKS (-1.5%), TinyStories REVERSES (+14.9%).
Advantage ∝ data diversity. Engram = learning accelerator, most valuable early in training on diverse data.

Shakespeare 20K: Anamnesis 1.51, Transformer 2.62 (-42.4%).
Advantage keeps growing: -20.8% (5K) → -35.8% (10K) → -42.4% (20K).
Anamnesis improves 1.9x faster even 10K→20K. TF 20K (2.62) = bare RetNet 10K.

Complete scaling ablation (Shakespeare char, seed=42):
RetNet vs TF: +4.3% (5K) → -10.6% (10K) → -20.9% (20K) — RetNet is the scaling champion.
Engram contribution: -24.1% (5K) → -28.2% (10K) → -27.1% (20K) — stable ~27% accelerator.
Growing advantage is driven by RetNet+layerwise, NOT Engram.

### Context-Length Crossover

Mamba wins at short context (seq_len=128, PPL=3.77), Anamnesis wins at long context (seq_len=512, PPL=3.12).

## Research Overview

**Small Reasoner + Million-Context Memory Compiler**

### Pipeline: capture → keep → align → margin → decide

```
Long Context → CAPTURE (compiler) → KEEP (typed mem) → ALIGN (training)
     → MARGIN (ledger) → DECIDE (small reasoner: RetNet + O(1) recurrent inference)
```

### Typed Memory Architecture

| Memory Type | What it stores | Latency | Capacity | Device |
|-------------|---------------|---------|----------|--------|
| **RetNet State** | Streaming recurrent state | O(1) | Fixed d²×L | GPU |
| **TokenCopyBuffer** | Raw token embeddings for exact recall | O(1) | K slots | GPU |
| **Engram (hot)** | N-gram hash lookup, scalar-gated | O(1) hash | M slots | GPU |
| **Engram (cold)** | Full static knowledge base | O(1) hash + IO | Millions | CPU/NVMe |

### Research Phases

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1: Mechanism Validation | Verify components on synthetic tasks | **COMPLETE** |
| Phase 2: O(1) Recurrent Inference | Constant-memory inference | **COMPLETE** |
| Phase 3: Context Compiler (1M) | Chunk retrieval pipeline | **COMPLETE** |
| Phase 4: Reliable 1M | EM=1.0 at 1M via Engram-enhanced embeddings | **COMPLETE** |
| Phase 5: Real Tasks | Transfer to real language modeling | **COMPLETE** |
| Phase 6: Paper & Submission | Finalize paper for publication | **IN PROGRESS** |

## Phase 1-4 Summary (COMPLETE)

**Synthetic task results:** Needle@1024 EM=1.000 (TCB essential), XOR@1024 loss=0.000.
Recurrent mode matches parallel (max diff 0.015). Memory O(d²×L + d×K).

**Pipeline (train@512 → eval 1M):** Contrastive chunk selection, token embedding readout.
EM=1.000 at 1M with Engram-enhanced embeddings (proj_dim=256).

**Discarded:** Delta rule, vector gate, input-dependent gamma, AttnRes, hierarchical retrieval, chunk-level RoPE.

## Phase 5 Summary (COMPLETE)

**Top 10 findings:**

1. **Layerwise gamma is RetNet-fundamental** — 15-22% PPL reduction, 0 extra params
2. **8 heads synergize with layerwise** — 25% together (not additive 21%)
3. **Engram accelerates learning** — Gap grows 5.5%→36% during training (Proof 31)
4. **lr=1e-3 is optimal** — 2.5x faster convergence; fair LR comparison essential
5. **Training stability** — Variance 1.5-3.7x lower than Transformer
6. **Engram is conditional on LR** — Helps char-level (-16% to -46%); BPE: high variance, seed-dependent (Phase 5.34)
7. **RetNet scaling anomaly** — d=256 worse than d=128 on diverse data; Engram compensates
8. **Advantage scales with repetitiveness** — TinyStories -46% > Shakespeare -21% > WikiText-2 -18%
9. **Multi-needle fails** — Top-2 recall 20% (near random); needs iterative retrieval
10. **Conv1D +13.7%** — Only 320 params, local positional context after hash lookup

## Autonomous Research Loop

### The Loop (NEVER STOP)

1. HYPOTHESIZE → 2. IMPLEMENT → 3. TRAIN → 4. EVALUATE → 5. DECIDE → 6. DISCOVER → 7. FIX → 8. PROVE → 9. COMMIT → 10. REFLECT

### Protocol

- **Single change per experiment**
- **Seed 42** for reproducibility
- **Matched T_max and lr** for fair comparison
- Simpler + equal/better → always keep

### Git Discipline

`feat|fix|refactor|docs|test|perf|experiment|proof: <description>`
Push after each commit. Results/logs/figures are gitignored.

## Project Structure

```
Resources/
├── src/models/               # anamnesis.py, retnet_engram.py, minimal_mamba.py
├── src/layers/               # Retention, AttnRes, Engram, Milestone
├── experiments/              # train_real.py (main), train_synthetic.py, multihop_eval.py
├── experiments/results/real/  # [gitignored] All experiment results
├── docs/paper.tex            # Main paper
├── docs/proofs/              # Formal proofs (1-48)
└── references/               # BibTeX
```

## Key Hyperparameters (optimized defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| d_model | 128 | validated up to 256 |
| n_heads | 8 | synergistic with layerwise gamma |
| n_layers | 8 | |
| learning_rate | 1e-3 | 2.5x faster than 3e-4 |
| steps | 5000 | T_max=5000 cosine schedule |
| engram_slots | 4096 | U-shaped optimum |
| layerwise_gamma_spread | 1.0 | U-shaped optimum |
| position_encoding | sinusoidal | |
| use_engram | conditional | Char-level always OK; BPE high variance, not recommended |

## Key Proofs

| # | Title | Key result |
|---|-------|-----------|
| 31 | O(1/D) gradient vanishing | Fundamental for recurrent chains |
| 40 | Scalar gate preserves direction | Isotropic > anisotropic scaling |
| 47 | Engram SNR vs collision | SNR ∝ √(KS/M) |
| 48 | Engram BPE limit (revised) | LR-dependent: helps at 1e-3, hurts at 3e-4 |

## Key References

- Sun et al. (2023) — Retentive Network
- Vaswani et al. (2017) — Attention Is All You Need
- Gu & Dao (2023) — Mamba
- Katharopoulos et al. (2020) — Transformers are RNNs

## Tools

- `conda run -n base python` — Training (homebrew python3 lacks torch)
- `pytest tests/` — Run before every commit
- `torch`, `einops` — Core dependencies
