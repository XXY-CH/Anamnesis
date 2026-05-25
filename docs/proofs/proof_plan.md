# Proof Plan: Dense RetNet + Block AttnRes + Engram

Created: 2026-05-03
Updated: 2026-05-04 after clarifying MoE as deferred, not rejected

## Working Thesis

The architecture should not yet be claimed as a globally `O(1)` long-context
system, an impossible-triangle solution, or a general Transformer/MoE
replacement. The near-term defensible research question is:

```text
Under strict small-state sequence constraints, how much deterministic long-range
recall and static factual lookup can be recovered by adding bounded, orthogonal
auxiliary memory paths?
```

After auditing Kimi Attention Residuals and the synthetic results, the
defensible Dense baseline accounting is:

```text
time axis:              O(1) RetNet recurrent backbone per token
depth axis:             O(N_depth) Block Attention Residual sources
knowledge axis:         O(K) Engram hash retrieval
dense channel axis:     standard Dense MLP/FFN channel mixing, phase-1 control
capacity axis:          MoE is deferred, phase-2 extension
time-preservation axis: milestone-triggered gated retention, optional/conditional
exact token axis:       O(B_time) Milestone Snapshot Readout, optional/conditional
```

where `K` is the number of Engram hash heads, `N_depth` is the number of
AttnRes block summaries, and `B_time` is the bounded number of token-time
snapshots. These must be reported as resources, not hidden behind an `O(1)`
headline. The key research question is therefore split: RetNet handles sequence
streaming, Block AttnRes handles depth-wise residual dilution, Engram handles
static lookup memory, and Dense MLP/FFN layers handle ordinary channel mixing in
phase 1 without sparse routing.

MoE has been deferred out of the active proof target for engineering isolation.
This is not a theoretical rejection. The Linear-MoE paper remains in the corpus
as evidence that a phase-2 RetNet/linear-RNN + MoE extension is plausible, but
MoE routing, expert capacity, and load-balancing are intentionally excluded from
the current theorem and implementation baseline.

## Immediate Proof Order

1. A-depth: prove Kimi Block AttnRes gives direct depth-wise access to earlier
   layer/block sources under attention-mass and dominance conditions.
2. C: prove Engram hash collision, noise variance, update safety, and namespace
   capacity for paper-faithful hashed lookup.
3. Dense channel axis: state the ordinary Dense MLP/FFN assumption and verify
   it does not add routing/budget proof obligations.
4. A-time/B appendix: keep Budgeted Sparse Attention Anchors and budgeted gates as an
   optional exact-token memory extension, not as the definition of Kimi AttnRes.
5. A-time/gate extension: if milestone-triggered pass-through becomes active,
   prove bounded cumulative leakage and milestone trigger precision/recall.
6. A-time/exact-readout extension: if high-entropy exact recall is required,
   prove the bounded snapshot readout condition and evaluate exact-match against
   RetNet-only recurrence.
7. AttnRes extension: if distance-penalized AttnRes becomes active, prove the
   content-margin condition under the distance penalty and test old-assumption
   recall.
8. ARC controller: prove the bounded controller contract for selective write,
   sparse read, and gated fusion, with address-margin and fusion-margin
   conditions.
9. Replacement ladder: state necessary conditions for any future general LLM
   replacement claim, including resource-recall lower bounds and functional
   coverage.
10. Tightening pass: replace loose collision and gate arguments with high
   probability Engram concentration and ISS-style gated-retention stability.

This order reflects the corrected axes. Depth dilution and token-time forgetting
are different problems.

## Main Conditional Theorems To Prove

### G-depth: Block AttnRes Depth-Signal Theorem

Let layer/block source `i` be useful for a deeper layer `l`. If Block AttnRes
assigns that source attention mass at least `epsilon_depth`, normalization is
non-degenerate, and the value path dominates cancellation terms, then there is a
direct depth-wise gradient/information path whose lower bound does not decay
exponentially with `l-i`.

### G-knowledge: Engram Safety Theorem

If the active key count, hash heads, slot count, residual scale, update norm, and
downstream margin satisfy the C-side inequalities, then Engram retrieval/update
noise remains a bounded perturbation.

### G-dense: Dense Channel-Mixing Compatibility Claim

The active model uses ordinary Dense MLP/FFN channel mixing after RetNet sequence
mixing. This is a baseline architectural assumption inherited from standard
Transformer/RetNet-style blocks, not a new sparsity theorem. Deferring MoE
removes expert routing, expert load balancing, expert all-to-all communication,
and top-k routing stability from the phase-1 proof obligations only.

### G-time Optional: Milestone Snapshot Readout Theorem

High-entropy needle diagnostics show that milestone gates can preserve a
protected recurrent state without making the original token value queryable.
Therefore exact token recall requires a value path:

```text
h_i -> snapshot_cache -> snapshot_readout/logit_bias -> y_T
```

If a critical token is captured by the snapshot mask, the snapshot budget is not
exceeded, and the readout assigns it attention/logit mass at least
`epsilon_time`, then exact recall has a path that does not rely on recovering the
token from the saturated RetNet state. The cost is `O(B_time)`, not `O(1)`.

### G-time Optional: Milestone-Triggered Retention Gate

If the model emits explicit reasoning milestones, the time-axis preservation
condition is not `gamma=1 forever`. It is bounded cumulative leakage:

```text
sum_{t=i+1}^{T} (1 - g_t) <= eta
```

combined with high milestone recall and bounded false positives.

### G-reasoning Future: Reasoning State Reuse

To claim general reasoning enhancement, snapshots must capture reusable
intermediate reasoning states rather than raw answer tokens. The required
condition is capture plus readout mass plus downstream reasoning margin.

### G-knowledge Future: Canonicalized Engram Retrieval

To claim knowledge retrieval enhancement, Engram needs a canonicalized key hit
rate, bounded hash-collision noise, and a measurable fact margin. Module-drop
evaluation must show Engram is causally used.

### G-replacement Future: Resource-Conditional Baseline Dominance

To move toward a general LLM replacement claim, the system must beat matched
Transformer/Mamba/RetNet baselines on a benchmark family under explicit resource
accounting:

```text
S_R + O(B_time d) + O(N_depth d) + O(KM d) + O(C)
```

It must also respect the lower bound that arbitrary exact recall of `m`
independent high-entropy facts requires memory scaling with `m`.

### G-controller Future: Addressed Reasoning Controller

To unify the memory paths, a small controller must choose when to write, what to
write, which bounded memory to read, and how to fuse the retrieved value. The
controller is acceptable only if its state and memory caps are fixed by policy,
so it changes constant factors rather than the sequence-length order. Its core
proof obligations are address margin, fusion margin, anti-collapse stability,
and telemetry for the chain:

```text
capture -> keep -> address -> fuse -> decision
```

## Proof Artifacts

| Artifact | Purpose |
|---|---|
| `docs/proofs/01-gradient-interference.md` | Historical A-time proof skeleton for optional Sparse Anchor Residual, not Kimi AttnRes. |
| `docs/proofs/02-conditional-nondecay-gradient-proof.md` | Historical A-time proof with lemmas, assumptions, cache-size dependence, and stochastic-gate corollary; appendix only unless sparse anchors are reintroduced. |
| `docs/proofs/03-engram-hash-capacity-proof.md` | First formal C1 proof for multi-head Engram collision probability and signed collision-noise variance. |
| `docs/proofs/04-engram-residual-perturbation-proof.md` | Bridge from C1 hash-noise variance to margin-safe residual injection. |
| `docs/proofs/05-budgeted-gate-stability-proof.md` | Historical B-time proof for optional anchor/cache gates, not required by the current Dense RetNet + Block AttnRes + Engram baseline. |
| `docs/proofs/06-total-gradient-dominance-proof.md` | Historical A-time rigor pass: path existence does not imply total gradient unless value path dominates cancellation terms. |
| `docs/proofs/07-budget-constrained-gate-proof.md` | Historical B-time hard-budget proof: oracle gate selects top positive utilities up to budget. |
| `docs/proofs/08-softmax-readout-mass-proof.md` | Historical A-time proof: attention logit margin is sufficient for `p_{T,i} >= epsilon`; depth analogue is in proof 17. |
| `docs/proofs/09-utility-estimation-robustness-proof.md` | Historical B-time proof: gate stability under bounded utility-estimation error. |
| `docs/proofs/10-engram-update-stability-proof.md` | C3 proof: bounded Engram slot/session-cache updates preserve non-target decisions under collision, residual-scale, and margin constraints. |
| `docs/proofs/11-global-conditional-feasibility-theorem.md` | Historical optional-anchor composition theorem; superseded for the active baseline by the Dense proof plan plus proof 17. |
| `docs/proofs/12-anchor-success-probability-factorization.md` | Historical A-time proof: decomposes `p_A` into gate recall, readout success, and dominance success. |
| `docs/proofs/13-normalization-jacobian-stability.md` | Shared support proof: normalization caveats apply to both optional anchors and depth-wise AttnRes. |
| `docs/proofs/14-adaptive-memory-price-stability.md` | Historical B-time proof: adaptive Lagrangian memory price for optional anchor/cache gates. |
| `docs/proofs/15-proof-closure.md` | Closure document for the earlier optional Sparse Anchor Residual theorem; not the active Dense-first phase-1 baseline closure. |
| `docs/proofs/16-attnres-linearmoe-realignment-audit.md` | Realignment audit: Kimi AttnRes is depth-axis; Linear-MoE is retained as phase-2 extension evidence. |
| `docs/proofs/17-depth-attnres-non-dilution-proof.md` | A-depth proof: Kimi-style AttnRes gives a direct non-diluted depth-wise value path under source attention mass and alignment assumptions. |
| `docs/proofs/18-dense-baseline-conditional-theorem.md` | Active Dense-first phase-1 theorem: composes RetNet bounded recurrence, Block AttnRes depth readout, Engram bounded perturbation, and Dense MLP/FFN assumptions while deferring MoE and token-time anchors. |
| `docs/proofs/19-residual-injection-composition-guard.md` | Active composition guard: prevents Engram and Block AttnRes residual injections from colliding through placement, LayerScale/ReZero scales, default-closed gates, and branch-norm logging. |
| `docs/proofs/20-milestone-triggered-retention-gate.md` | Optional A-time gate theorem: milestone-triggered retention preserves protected state only under bounded cumulative leakage, trigger recall, and anti-spam constraints. |
| `docs/proofs/21-distance-penalized-attnres.md` | Optional AttnRes extension: adds ALiBi-style source-age/depth-distance penalty and records the old-assumption recoverability condition. |
| `docs/proofs/22-milestone-snapshot-readout.md` | Optional exact token-time theorem: bounded milestone snapshots provide a queryable value path when RetNet gates preserve amplitude but not recoverable high-entropy identity. |
| `docs/proofs/23-reasoning-state-reuse-theorem.md` | Future reasoning theorem: snapshots become general reasoning aids only when they capture reusable intermediate states and preserve downstream decision margin. |
| `docs/proofs/24-canonicalized-engram-retrieval-theorem.md` | Future knowledge theorem: Engram retrieval enhancement requires canonical key hit rate, bounded collision noise, and causal module-drop evidence. |
| `docs/proofs/25-general-llm-replacement-necessary-conditions.md` | Replacement ladder: necessary conditions, resource accounting, and information lower bound before any general LLM replacement claim. |
| `docs/proofs/25-engram-hoeffding-concentration-bound.md` | Conditional concentration note: projected signed Engram collision noise has an exponential tail only under idealized hash assumptions. |
| `docs/proofs/26-moe-routing-lyapunov-stability.md` | Phase-2 risk note: small residual injection gives a local router-perturbation bound, not a global MoE stability proof. |
| `docs/proofs/26-tight-engram-concentration-bound.md` | Tighter Engram theorem: Bernstein/net high-probability collision-noise bound under signed bounded-vector assumptions. |
| `docs/proofs/27-gated-retention-iss-stability.md` | Tighter gate theorem: input-to-state stability conditions that prevent milestone pass-through from saturating recurrent state. |
| `docs/proofs/27-snapshot-gradient-flow-dominance.md` | Conditional optimization note: snapshot-to-logit may have a gradient advantage when capture and margin conditions hold. |
| `docs/proofs/28-residual-scale-nonnegativity-corollary.md` | Engineering corollary: `abs()` prevents scalar sign reversal but does not prove branch alignment or margin. |
| `docs/proofs/29-token-copy-buffer-expressiveness.md` | Conditional exact-copy logit theorem: TokenCopyBuffer succeeds only under capture, attention, embedding-margin, and base-logit-margin conditions. |
| `docs/proofs/30-addressed-reasoning-controller.md` | Target-design controller theorem: ARC is valid only under fixed resource caps, address-margin, fusion-margin, and anti-collapse conditions. |
| `docs/proofs/proof_rigor_audit_2026-05-11.md` | Corrective audit that downgrades over-strong proof claims and records reviewer rules for future theorem drafts. |
| future `MoE reintegration proof` | Phase-2 capacity theorem: sparse experts may be reintroduced after Dense memory paths are stable; must prove routing robustness under Engram/AttnRes/Snapshot perturbations. |
| `docs/proofs/three_part_original_paper_audit_2026-05-03.md` | Original-paper audit for RetNet, Engram, and sparse attention residual; identifies which claims are directly supported and which are project extensions. |
| `docs/proofs/engram_gemini_alignment_audit_2026-05-03.md` | Audit of Gemini Engram claims against the downloaded Engram paper and current project direction. |
| `docs/proofs/gemini_full_alignment_and_recommended_papers.md` | Full consistency check for Gemini discussion plus recommended paper list. |
| `docs/proofs/pdf_assumption_audit_2026-05-03.md` | PDF-backed audit of A1/C1 assumptions, including implementation mismatch and citation caveats. |
| `references/papers/MANIFEST.md` | Exact paper corpus and source links. |
| `docs/progress/2026-05-03-proof-kickoff.md` | Work log and next steps. |

## Implementation Gap To Track

The current codebase now implements the small aligned research scaffold:

- `src/layers/retention.py`: retention backbone with optional milestone gates.
- `src/layers/attention_residual.py`: Kimi-style depth-axis Block Attention Residual readout over layer/block sources.
- `src/layers/engram.py`: deterministic multi-head N-gram hash lookup Engram.
- `src/layers/milestone_snapshot.py`: bounded token-time milestone snapshot readout for exact recall diagnostics.
- `src/models/anamnesis.py`: Dense RetNet + optional Engram + optional Block AttnRes + optional snapshot readout.
- `experiments/train_synthetic.py`: synthetic diagnostic runner for `ours`, ablations, RetNet, and Transformer baselines.

The larger research program still needs:

- namespace/salt design for multimodal hash keys;
- real recurrent/chunkwise inference path validation beyond parallel training mode;
- richer instrumentation for branch norms, readout mass, capture recall, and module-drop deltas;
- ARC-style controller diagnostics for write/read/fusion decisions, starting
  with TokenCopyBuffer correct-slot attention;
- non-copy reasoning diagnostics that can expose snapshot shortcut overfitting.

Therefore the current mathematical work is now partly executable, but each
positive claim still needs module-specific ablation evidence.

## Current Conditional Chain

```text
Active A-depth: if a prior layer/block source receives sufficient Block AttnRes
    mass and the value path dominates cancellation terms, the depth-wise path
    avoids products over all intermediate layer Jacobians.

A-time appendix: if critical anchors are selected and read, sparse residual
    paths bypass RetNet exponential decay. This is not part of Kimi AttnRes and
    is not active unless token-time anchors are reintroduced.

A2: total-gradient non-decay additionally requires the residual value path to
    dominate key, gate, and other cancellation terms.

A3: the readout-mass condition follows from a softmax logit margin; fixed
    epsilon with growing cache size requires margin growing like log m.

C1/C2: if active Engram keys and hash capacity satisfy collision/noise bounds,
       Engram retrieval is a bounded residual perturbation.

C3: online Engram edits and session-cache branches are safe only under bounded
    update norm, bounded active update count per namespace, sufficient hash
    capacity, residual-scale control, and downstream margin. This records the
    Gemini update/cache idea as a conditional theorem rather than an assumed
    property of the Engram paper.

B1 appendix: if anchor utilities have a margin gap, a stable sparse-gate threshold exists;
    without that gap, stable lambda cannot be guaranteed.

B2 appendix: under a hard cache budget and oracle additive utilities, the optimal gate
    selects the top positive utilities up to budget.

B3 appendix: with estimated utilities, threshold/top-B choices remain stable only when
    estimation error is smaller than the relevant utility margin.

G1 appendix: composing A-time/B/C gives a joint success probability at least
    p_A - |Y|delta_C - delta_B for a finite protected probe set Y. The global
    theorem is useful only if this lower bound is positive.

A4 appendix: p_A can be decomposed as r q d, where r is critical-anchor gate recall,
    q is readout-mass success conditional on selection, and d is residual-path
    dominance success conditional on selection plus readout. Therefore G1's
    measurable condition is r q d > |Y|delta_C + delta_B.

A5: normalization does not invalidate A1 only when the useful backpropagated
    direction is outside LayerNorm/RMSNorm null or near-null subspaces and the
    activation scale keeps the Jacobian bounded.

B4 appendix: an adaptive memory price reaches a stable budget interval only when the
    utility boundary gap is positive and the price step is smaller than that
    gap after accounting for utility-estimation error.

Closure: within the simplified model, the proof chain is closed as a conditional
    theorem for optional token-time anchors only. The measurable success
    condition is r q d > |Y|delta_C + delta_B.

Original-paper audit: RetNet and Engram are literature-supported components.
    Root `papers/attention-residual/Attention Residuals.pdf` directly supports
    depth-wise AttnRes, but our token-time Sparse Anchor Residual remains a
    project extension, so A-side non-decay proofs must be split into depth-axis
    AttnRes and optional time-axis anchors.

MoE deferral: Linear-MoE remains phase-2 extension evidence, but the phase-1
    baseline is Dense-first for engineering isolation.

Active Dense-first closure: proof 18 replaces the older optional-anchor closure
    as the current phase-1 global theorem. It proves only conditional feasibility
    for Dense RetNet + Block AttnRes + Engram and explicitly defers MoE and token-time
    sparse anchors.

Residual composition guard: proof 19 turns the same-layer Engram/AttnRes
    interference risk into a concrete implementation contract. Fresh user
    context is routed through RetNet and Block AttnRes, not written into global
    Engram.

Milestone-gate correction: proof 20 adds the missing A-time gate theorem. The
    key condition is bounded cumulative leakage, not permanent pass-through.

Distance-penalty correction: proof 21 treats ALiBi-style AttnRes penalty as a
    project extension. It creates recency bias only among available summaries;
    it does not recover information that RetNet/gates failed to preserve.
```
