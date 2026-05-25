# Proof Status Registry

> Living document tracking the validation status of all formal proofs.
> Updated: 2026-05-25

## Status Definitions

| Status | Meaning |
|--------|---------|
| Verified | Empirically validated, math holds |
| Revised | Conclusion partially invalidated by new evidence |
| Deprecated | Conclusion overturned by experiment or architecture change |
| Pending | Drafted but not yet empirically validated |

## Registry

### Foundational Proofs (1-15)

| # | Title | Status | Notes |
|---|-------|--------|-------|
| 01 | Gradient interference in multi-branch | Verified | Core AttnRes theory |
| 02 | Conditional non-decay gradient | Verified | |
| 03 | Engram hash capacity | Verified | Hash collision bounds |
| 04 | Engram residual perturbation | Revised | Vector gate assumption, see Proof 40 |
| 05 | Budgeted gate stability | Verified | |
| 06 | Total gradient dominance | Verified | |
| 07 | Budget-constrained gate | Verified | |
| 08 | Softmax readout mass | Verified | |
| 09 | Utility estimation robustness | Verified | |
| 10 | Engram update stability | Revised | Vector gate assumption, see Proof 40 |
| 11 | Global conditional feasibility | Verified | |
| 12 | Anchor success probability | Verified | |
| 13 | Normalization Jacobian stability | Verified | |
| 14 | Adaptive memory price stability | Verified | |
| 15 | Proof closure (Phase 1 summary) | Verified | |

### Architecture Proofs (16-30)

| # | Title | Status | Notes |
|---|-------|--------|-------|
| 16 | AttnRes-LinearMoE realignment audit | Verified | |
| 17 | Depth AttnRes non-dilution | Verified | |
| 18 | Dense baseline conditional theorem | Verified | |
| 19 | Residual injection composition guard | Revised | Vector gate assumption, see Proof 40 |
| 20 | Milestone-triggered retention gate | Verified | |
| 21 | Distance-penalized AttnRes | Verified | |
| 22 | Milestone snapshot readout | Verified | |
| 23 | Reasoning state reuse theorem | Verified | |
| 24 | Canonicalized Engram retrieval | Verified | |
| 25 | Engram Hoeffding concentration bound | Verified | |
| 26a | MoE routing Lyapunov stability | Verified | |
| 26b | Tight Engram concentration bound | Verified | |
| 27a | Gated retention ISS stability | Verified | |
| 27b | Snapshot gradient flow dominance | Verified | |
| 28 | Residual scale non-negativity corollary | Verified | |
| 29 | Token copy buffer expressiveness | Verified | TCB essential for synthetic needle |
| 30 | Addressed reasoning controller | Verified | |

### Phase 3 Proofs (31-42)

| # | Title | Status | Notes |
|---|-------|--------|-------|
| 31 | Input-dependent gamma analysis | Verified | Validated at all lengths |
| 32 | TCB retrieval quality bound | Verified | |
| 33 | Engram LM capacity waste | **Deprecated** | Vector gate caused PPL crash; paper uses scalar gate. See Proof 40. |
| 34 | Context Compiler design | **Deprecated** | Token readout fails on real data. Replaced by RAG pipeline. See Proof 41. |
| 35 | Delta rule analysis | Verified | Delta rule correctly discarded |
| 36 | Chunk retrieval analysis | Verified | Scaling to 1M validated (EM=0.875) |
| 37 | Engram as decoupled knowledge base | Pending | Never completed; direction changed |
| 38 | Learned gating replaces MARK_THOUGHT | Revised | Valid for synthetic; limited for RAG pipeline |
| 39 | Engram TCB trigger analysis | Revised | Works with oracle distillation; self-supervised fails |
| 40 | Scalar gating lower bound | Pending | Vector to Scalar gate fix |
| 41 | RAG separation principle | Pending | Retrieval vs generation manifold separation |
| 42 | RoPE SNR collapse | Pending | Position encoding destroys content at scale |

## Key Deprecation Explanations

**Proof 33 (Deprecated)**: Concluded Engram hurts LM based on val_ppl 9.95 vs 5.87 (Shakespeare).
Root cause identified: code used **per-dimension vector gate** (anisotropic scaling destroys
semantic direction). The DeepSeek Engram paper uses **scalar dot-product gate** (isotropic
scaling preserves direction). With scalar gate + zero init, PPL cannot increase (Proof 40).

**Proof 34 (Deprecated)**: Token embedding readout pipeline fails on real data (EM=0.000 on
Shakespeare). Replaced by RAG-style pipeline: chunk retrieval (works, 0.875 accuracy) +
native autoregressive generation. See Proof 41 for theoretical justification.
