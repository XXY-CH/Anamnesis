# Brainstorm: Research Directions Grounded in Frontier Literature (2026-05-24)

## Literature Survey

### Critical Papers

1. **SILA (ICLR 2026)** - Selective Ignoring Linear Attention
   - Decouples memory store/recall, memory-dependent gate
   - Weighted loss for selective memory writing → 20× length extrapolation
   - **Relevance**: Validates our Engram trigger approach. Their memory-dependent gate ≈ our Engram surprise.

2. **SR-TTT** - Surprisal-Aware Residual TTT
   - Uses reconstruction loss as self-supervised signal for incompressible tokens
   - Routes surprising tokens to external sparse memory (Residual Cache)
   - Two-stage curriculum: freeze backbone → train cache
   - **Relevance**: EXACTLY our oracle-to-learned pattern. Validates two-stage training.

3. **FDM (Fan Duality Model)** - Wave-Particle Separation
   - Explicit wave (norm-preserving recurrence) + particle (selective cache)
   - Freeze-Scan training: freeze scan, optimize cache → 7.5× convergence
   - Identifies "gradient sink problem": recurrent scan dominates gradients, starves cache
   - **Relevance**: Validates Direction 2 (Dual-Stream). Explains our training instability.

4. **Bicameral Architecture / Knowledge Objects**
   - Orthogonality Constraint: reliable memory needs orthogonal keys
   - Semantic embeddings CAN'T be orthogonal → hash-based discrete storage
   - Neural memory collapses 97%→0% under semantic interference
   - **Relevance**: Validates our Engram hash tables. Proves why discrete hashing is necessary.

5. **ReSuME (ICLR 2026 Workshop)** - Representational Surprise via SAEs
   - SAE reconstruction error = representational surprise
   - Surprise-gated memory writing, covariance-aware normalization
   - **Relevance**: Validates Direction 1 (IB-based surprise). Our Engram gate is a simpler version.

6. **GSA (Gist Sparse Attention)** - Hierarchical compression
   - Gist tokens as information-dense pivots for selective unfolding
   - Coarse-to-fine: meta-gist → gist → raw tokens
   - Log-linear complexity
   - **Relevance**: Our chunk embeddings ≈ gist tokens. Could extend to hierarchical chunks.

7. **SPLA** - Sparse Plus Linear Attention
   - Partitions context into exact (sparse) + approximate (residual linear attention)
   - Second-order Taylor for block selection (no heuristics)
   - **Relevance**: Our chunk selection could use Taylor-based metric instead of contrastive.

8. **SuRe** - Surprise-based Replay for Continual Learning
   - NLL as surprise signal for buffer selection
   - Mathematical proof: surprise selection reduces Distribution Fidelity Locally
   - **Relevance**: Surprise = loss. Could replace oracle with per-token loss as importance signal.

## Research Directions (Ranked by Impact + Feasibility)

### Direction A: Loss-Driven Surprise Gating (replaces oracle entirely)
**Inspired by**: SR-TTT, SuRe, ReSuME
**Hypothesis**: Per-token LM loss is a self-supervised importance signal. High-loss tokens are "surprising" and should be stored in TCB.
**Implementation**: During training, compute per-token cross-entropy. Tokens with loss > threshold get stored in TCB. No oracle needed.
**Risk**: Loss signal may be noisy at early training. Cold-start problem persists.
**Why it might work**: SR-TTT showed reconstruction loss works as surprise signal. SuRe proved NLL-based selection is optimal.
**Mathematical basis**: IB objective min I(X;Z) - β·I(Z;Y) where Y is the next-token prediction task.

### Direction B: Wave-Particle Separation with Freeze-Scan (addresses instability)
**Inspired by**: FDM, SILA
**Hypothesis**: Separate the retention state into wave (prediction, normal decay) and particle (identity, slow/no decay + external cache). Freeze-Scan training eliminates gradient sink.
**Implementation**: Two-phase training. Phase 1: train base RetNet normally. Phase 2: freeze RetNet, train particle/cache component only.
**Risk**: Architectural change is larger. May not work on real data.
**Why it might work**: FDM showed 7.5× convergence improvement with Freeze-Scan. Our training instability matches FDM's described gradient sink.
**Mathematical basis**: Wave-particle duality. Wave = norm-preserving unitary (Givens rotations in FDM). Particle = selective addressing into fixed-size cache.

### Direction C: Hierarchical Chunk Memory (extends pipeline to real data)
**Inspired by**: GSA, RetroLM
**Hypothesis**: Multi-resolution chunk memory (chunk → section → document) with coarse-to-fine retrieval enables efficient long-context on real data.
**Implementation**: Train chunk embeddings, build hierarchical index, retrieve at coarse level then refine.
**Risk**: Complex engineering. May not generalize from synthetic to real data.
**Why it might work**: GSA showed 11+ point improvement on RAG benchmarks with selective unfolding.
**Mathematical basis**: Information bottleneck at each level: coarse level compresses more, fine level preserves detail.

### Direction D: Orthogonal Hash Memory with Typed Slots (strengthens Engram)
**Inspired by**: Bicameral Architecture, CraniMem
**Hypothesis**: Typed hash slots (entity/relation/value) with orthogonal addressing prevent semantic interference.
**Implementation**: Modify Engram hash to use typed slots. Add schema enforcement.
**Risk**: Requires structured data. May not work on unstructured text.
**Why it might work**: Bicameral paper proved neural memory collapses without orthogonal keys. Hash-based storage is provably interference-free.

## Recommended Execution Order

1. **Direction A** (quickest validation): Replace oracle with loss-driven surprise. Test on needle@512.
2. **Direction B** (addresses instability): Freeze-Scan training for Engram trigger. Test convergence.
3. **Direction C** (real-data transfer): Apply pipeline to TinyStories/Shakespeare.
4. **Direction D** (long-term): Typed hash slots for structured memory.

## Controlled Baselines (completed)

| Variant | Best EM@512 |
|---------|-------------|
| Transformer | 0.000 |
| Bare RetNet | 0.000 |
| Ours (oracle TCB) | 0.875 |
| Ours (Engram trigger + distill) | 1.000 |
