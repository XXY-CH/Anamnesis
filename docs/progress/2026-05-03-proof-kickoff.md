# 2026-05-03 Proof Kickoff

## User Goal

Acquire the papers implied by the GPT-5.5-Pro discussion and start preparing a
mathematical proof path for the RetNet + Engram + dynamic attention residual
architecture. All reasoning, plans, and proof process must be preserved in the
workspace.

## Repository State Observed

- Existing project skeleton includes `src/layers/retention.py`,
  `src/layers/engram.py`, `src/layers/attention_residual.py`, and
  `src/models/anamnesis.py`.
- Existing notes are high-level and do not yet contain a formal proof trail.
- The directory is not currently a git repository, so durable trace is file-based
  rather than commit-based.

## Papers Retrieved

See `references/papers/MANIFEST.md` for the full table.

Fetched locally:

- RetNet: `references/papers/core/retnet_2307.08621.pdf`
- VQ-VAE: `references/papers/core/vq_vae_1711.00937.pdf`
- Attention Is All You Need: `references/papers/related/attention_is_all_you_need_1706.03762.pdf`
- Linear Attention: `references/papers/related/linear_attention_2006.16236.pdf`
- DeepNet: `references/papers/related/deepnet_2203.00555.pdf`
- Memorizing Transformers: `references/papers/related/memorizing_transformers_2203.08913.pdf`
- Feature Hashing: `references/papers/related/feature_hashing_0902.2206.pdf`
- Count-Min Sketch: `references/papers/related/count_min_sketch_2005.pdf`
- Complementary Learning Systems: `references/papers/related/complementary_learning_systems_1995.pdf`
- Orange Pi 4 Pro A733 manual: `references/papers/hardware/orangepi_4_pro_user_manual_v1.4.pdf`

Attempted but not mirrored:

- Tonegawa et al. 2015 Engram review; the MIT mirror returned HTTP 404.

## Current Mathematical Position

The strongest first theorem is conditional:

```text
If a critical token is written to sparse KV anchors and later receives attention
mass >= epsilon, then the gradient path through the residual cache does not carry
the RetNet gamma^(T-i) decay factor.
```

This changes the central burden of proof from "RetNet stores everything" to:

```text
Can the gate learn high recall for future-useful anchors under budget B?
```

## Files Created

- `references/papers/MANIFEST.md`
- `docs/proofs/proof_plan.md`
- `docs/proofs/01-gradient-interference.md`
- `docs/proofs/02-conditional-nondecay-gradient-proof.md`
- `docs/proofs/03-engram-hash-capacity-proof.md`
- `docs/proofs/pdf_assumption_audit_2026-05-03.md`
- `docs/progress/2026-05-03-proof-kickoff.md`

## Proof Step Added

Started formal proof A1:

- Lemma 1 proves the RetNet-only path decays under the stronger finite-time
  assumption `||Gamma||_op <= gamma < 1`.
- Lemma 2 isolates the sparse residual value path and shows it has no
  `Gamma^(T-i)` factor.
- Theorem A1 gives a directional lower bound under explicit alignment,
  selection, and readout-mass assumptions.
- Cache-size dependence is now written as a separate constraint: unstructured
  attention gives roughly `p_{T,i} ~= 1/m`, so sparse residuals move the problem
  from distance to cache quality and budget.

Started formal proof C1:

- Lemma C1 bounds one-head collision probability by `(N-1)/M`.
- Lemma C2 bounds all-head consistent false collision by `(N-1)/M^K`.
- Lemma C3 records that signed hashing gives zero-mean collision noise.
- Lemma C4 bounds expected squared noise by `((N-1) R^2)/(K M)`.
- Theorem C1 separates exact false-match capacity from aggregate noise capacity.

PDF assumption audit added:

- RetNet paper supports the contractive decay assumption via fixed per-head
  `gamma < 1`, but current code's `gamma.abs()` is not enough to guarantee that
  after training.
- Feature Hashing supports signed zero-mean hashing; Count-Min supports the
  multi-hash sketch pattern but not the signed vector estimator.
- VQ-VAE supports discrete latent codes but not semantic alignment.

Bias/error correction pass:

- Updated `src/layers/retention.py` to use a fixed RetNet-style per-head
  `gamma < 1` buffer instead of trainable `abs(gamma)`.
- Added `tests/test_retention.py` to lock the contractive decay invariant and
  recurrent state shape.
- Tightened C1 independence assumptions: signed collision-noise variance uses
  independent Rademacher signs, not Count-Min's nonnegative counter estimator.

Proof continuation:

- Added `docs/proofs/04-engram-residual-perturbation-proof.md` to connect C1's
  hash-noise variance to downstream output perturbation and margin-safe Engram
  residual injection.
- Added `docs/proofs/05-budgeted-gate-stability-proof.md` to formalize the
  sparse gate's oracle threshold rule and show that stable `lambda` requires a
  utility margin gap.
- Added `docs/proofs/06-total-gradient-dominance-proof.md` to prevent the
  overclaim that a non-decaying path implies a non-decaying total gradient.
- Added `docs/proofs/07-budget-constrained-gate-proof.md` to prove the
  hard-budget oracle gate selects top positive utilities up to budget.
- Added `docs/proofs/08-softmax-readout-mass-proof.md` to derive the readout
  mass condition `p_{T,i} >= epsilon` from an explicit softmax logit margin.
- Added `docs/proofs/09-utility-estimation-robustness-proof.md` to show gate
  stability requires utility-estimation error below the selection margin.
- Downloaded Engram paper `references/papers/core/engram_conditional_memory_2601.07372.pdf`.
- Added `docs/proofs/engram_gemini_alignment_audit_2026-05-03.md`: Gemini's
  Engram architecture/offload/reasoning claims are mostly aligned, but RL-only
  online Engram updating and non-parametric slot editing are not proven by the
  paper and must remain future hypotheses.
- Downloaded additional Engram-adjacent papers: Hash Embeddings, Product-Key
  Memory, Infini-gram, ROME, MEMIT, parametric/non-parametric memory, SentencePiece,
  and RMSNorm.
- Added `docs/proofs/gemini_full_alignment_and_recommended_papers.md` to answer
  the broader Gemini consistency question and list the recommended reading stack.
- Recorded the Gemini update/cache boundary as a formal constraint: online
  Engram edits and temporary session caches are not assumed safe by default.
  Added `docs/proofs/10-engram-update-stability-proof.md` to prove a sufficient
  margin-safe condition under bounded update norm, namespace separation,
  collision capacity, residual-scale control, and downstream margin.
- Tightened the C3 conservative no-signed-cancellation bound from linear `S` to
  quadratic `S^2`, because multiple updates can align without zero-mean
  cancellation.
- Added `docs/proofs/11-global-conditional-feasibility-theorem.md` to compose
  A/B/C into the then-current optional-anchor theorem: joint success probability is at least
  `p_A - |Y|delta_C - delta_B`, and the theorem is meaningful only when this
  lower bound is positive.
- Added `docs/proofs/12-anchor-success-probability-factorization.md` to unpack
  `p_A` into measurable factors: critical-anchor gate recall `r`, readout
  success `q`, and residual dominance success `d`. The global feasibility
  condition becomes `r q d > |Y|delta_C + delta_B`.
- Added `docs/proofs/13-normalization-jacobian-stability.md` to close the
  normalization assumption in A1/A2/G1. It records the RMSNorm Jacobian bound
  and the LayerNorm nullspace caveat: residual gradients can still vanish if
  the useful direction lies in a normalization null or near-null subspace.
- Added `docs/proofs/14-adaptive-memory-price-stability.md` to formalize the
  Lagrangian memory-price controller. Under fixed utilities, strict top-budget
  gap `w`, and step size `eta n < w`, the price enters the stable budget
  interval in finite time. With estimated utilities, require
  `eta n < w - 2 eta_Delta`.
- Added `docs/proofs/15-proof-closure.md` to close the current proof chain as a
  conditional feasibility theorem. The final measurable condition is
  `r q d > |Y|delta_C + delta_B`; anything stronger would require empirical
  validation or additional assumptions.
- Added `docs/proofs/three_part_original_paper_audit_2026-05-03.md` after
  rechecking the three designed components against the original downloaded PDF
  texts. RetNet backbone and Engram conditional memory are directly supported
  with caveats; token-time Sparse Anchor Residual anchors are a novel optional
  project component
  assembled from attention/residual/external-KV primitives and must be proven
  and tested separately.
- User pointed out the root `papers/` folder. Added that corpus to
  `references/papers/MANIFEST.md`, extracted key local PDFs, and corrected the
  audit: `papers/attention-residual/Attention Residuals.pdf` directly supports
  depth-wise attention residual aggregation, but not our token-time sparse CoT
  anchor cache. The current proof should call our module "Sparse Anchor
  Residual" when precision matters.
- Re-read Kimi `Attention Residuals` after Gemini's reassessment and downloaded
  Linear-MoE `references/papers/related/linear_moe_2503.05447.pdf` from arXiv.
  Added `docs/proofs/16-attnres-linearmoe-realignment-audit.md`. At that moment
  the corrected architecture was multi-axis with MoE as a candidate extension;
  this has since been staged into a Dense-first phase-1 baseline: RetNet for
  time, Dense MLP/FFN for channel mixing, Block AttnRes for depth, Engram for
  hashed external memory, and optional Budgeted Sparse Attention Anchors for
  exact token-time facts. MoE remains a phase-2 capacity-axis extension. Current
  `src/layers/attention_residual.py` and `src/layers/engram.py` are not yet
  paper-faithful implementations.
- Rewrote `docs/proofs/proof_plan.md` to split the proof targets by axis and
  added `docs/proofs/17-depth-attnres-non-dilution-proof.md` for Kimi-style
  depth-axis Block AttnRes. The earlier token-time sparse-anchor theorem remains
  optional rather than the main AttnRes proof.
- User deferred MoE from the active target after noting realistic engineering
  risks. Updated the proof plan to the Dense-first phase-1 baseline:
  `RetNet + Block AttnRes + Engram + Dense MLP/FFN`.
- Reclassified Linear-MoE as phase-2 extension evidence. It remains in the paper
  corpus as a plausible capacity-axis route, but is not part of the phase-1
  theorem, implementation target, or vulnerability model.
- Corrected the earlier A/B proof chain: proofs 01/02/05-15 are now explicitly
  historical/appendix material for optional token-time Sparse Anchor Residuals,
  not the Kimi AttnRes proof.
- Added `docs/proofs/18-dense-baseline-conditional-theorem.md` as the active
  phase-1 global theorem after MoE deferral. It composes RetNet bounded recurrence,
  Block AttnRes depth readout, Engram bounded perturbation, and Dense MLP/FFN
  assumptions, while excluding MoE and token-time sparse anchors.
- Acknowledged the user's file move from
  `docs/proofs/16-system-vulnerabilities-and-mitigations.md` to
  `docs/architecture_vulnerabilities.md` and updated that document for the
  Dense-first phase-1 baseline with MoE deferred.
- Audited Gemini's latest Engram/context-routing feedback against the local
  Engram text. Deterministic hash retrieval, tokenizer compression, early/sparse
  Engram placement, 100B host offload, and factual-vs-reading ablation are
  supported by the extracted paper text.
- Updated `docs/architecture_vulnerabilities.md`: Engram semantic drift is now
  handled by functional boundary and context routing, not by turning Engram into
  soft semantic retrieval in the active baseline.
- Added `docs/proofs/19-residual-injection-composition-guard.md` for the
  same-layer Engram + Block AttnRes residual collision risk. It requires
  interleaved placement, independent near-zero LayerScale/ReZero scales,
  default-closed gates, parallel-normalized branches, and branch-norm logging.
- Updated proof 18 so the active Dense theorem depends on proof 19's residual
  composition guard.
- On 2026-05-04, audited Gemini's proposed milestone-gated RetNet and
  distance-penalized AttnRes extensions. Corrected overclaims in
  `docs/architecture_summary_and_deviations.md`: milestone gates and ALiBi-style
  AttnRes are project extensions, not paper-backed guarantees.
- Added `docs/proofs/20-milestone-triggered-retention-gate.md`. It proves the
  conditional A-time requirement as bounded cumulative leakage, plus milestone
  recall and anti-spam constraints.
- Added `docs/proofs/21-distance-penalized-attnres.md`. It records the
  recoverability condition under a distance/source-age penalty and warns that
  the penalty only prioritizes available summaries; it cannot recover state that
  was never preserved.
- Updated `docs/architecture_vulnerabilities.md` with milestone spam, logic
  cliff, recency-penalty miscalibration, and dead-module syndrome.
- Switched from proof-only work to code practice. Replaced the old EMA-style
  `EngramGate` and generic residual wrappers with a small aligned training
  architecture:
  `RetentionLayer` with optional milestone gates, `HashedNgramEngram`,
  `BlockAttentionResidual`, `AnamnesisModel`, and a synthetic LM training
  step.
- Added tests for deterministic Engram lookup, depth-axis AttnRes readout,
  milestone gate windows, model forward/backward, and one toy optimizer step.
  Verification: `pytest -q` passed with 7 tests.
- Added `experiments/train_synthetic.py`, a flexible synthetic diagnostic
  runner. It supports `ours`, `retnet`, and `transformer` variants under the
  same random seed/data/optimizer, with `needle`, `xor`, and `alien` tasks,
  masked losses, CSV output, and optional loss-curve PNGs.
- Added `src/models/transformer_baseline.py` as the standard Transformer
  comparison model for synthetic diagnostics.
- Ran 512-token needle diagnostics. Low-entropy password tokens produced a false
  positive: RetNet and ours both learned the small answer prior. Updated
  `experiments/train_synthetic.py` to use high-entropy password tokens and report
  exact-match.
- Added `--retention-gamma`, `--needle-password-len`, and `--alien-num-pairs` to
  `experiments/train_synthetic.py` for harsher diagnostics. A stress run with
  `--retention-gamma 0.95`, `--milestone-ttl 1024`, and
  `--needle-password-len 1` showed RetNet and Transformer fail as expected, but
  ours also failed to stabilize exact recall. This exposes a missing mechanism:
  current milestone gating slows decay but does not provide an explicit
  queryable snapshot/work-memory readout.
- Added `src/layers/milestone_snapshot.py` and integrated optional
  `MilestoneSnapshotReadout` into `src/models/anamnesis.py`. The snapshot
  branch stores bounded token-time hidden vectors adjacent to milestone markers
  and can optionally add a final snapshot-derived logit bias.
- Ran the high-entropy 512-token needle stress test again with
  `--retention-gamma 0.95`, `--milestone-ttl 1024`,
  `--needle-password-len 1`, `--use-snapshots`, and
  `--use-snapshot-logit-bias`. Results in
  `experiments/results/synthetic_needle_512_len1_gamma095_snapshot_logits/`:
  ours reached best exact match `0.75` at step 180, while vanilla RetNet and
  Transformer remained at `0.0`. This is the first positive separation for the
  explicit time-axis snapshot mechanism, but it is not yet stable because ours
  ended at exact match `0.375` at step 200.
- Added held-out evaluation support to `experiments/train_synthetic.py` via
  `--eval-batches` / `--eval-seed`, plus ablation variants:
  `ours_nosnapshot`, `ours_snapshot`, and `ours_snapshot_logits`.
- Ran a 512-token high-entropy needle ablation with held-out eval:
  `experiments/results/synthetic_needle_512_snapshot_ablation_eval/`.
  Best held-out exact match:
  `ours_nosnapshot=0.000`, `ours_snapshot=0.031`,
  `ours_snapshot_logits=0.344`, `retnet=0.000`. This confirms the first
  generalizing signal comes from the explicit snapshot-to-logit value path, not
  from the RetNet milestone gate or residual-only snapshot branch.
- Added `docs/proofs/22-milestone-snapshot-readout.md`. The theory is now
  explicitly adjusted: milestone gates can bound decay of a protected state
  component, while exact high-entropy recall requires a separate bounded
  snapshot readout with capture probability, readout mass, and logit-margin
  assumptions.
- Ran non-copy ablations and recorded them in
  `docs/progress/2026-05-04-synthetic-snapshot-ablation.md`. Current evidence
  boundary: snapshot-to-logit helps high-entropy needle exact recall, does not
  clearly improve XOR token accuracy, and does not show robust advantage on the
  current alien dictionary task.
- Added `xor_final` and `alien_static` diagnostics. `xor_final` confirms
  snapshot readout can dominate a single-answer readout path; `alien_static`
  confirms the current static mapping task is too easy to isolate Engram because
  RetNet also reaches held-out exact match `1.0`.
- Added module-drop held-out evaluation to `experiments/train_synthetic.py` via
  `--eval-drop-modules snapshot,engram,attnres`. This emits fields like
  `eval_no_snapshot_exact_match` and `eval_no_engram_loss`, enabling causal
  ablation during the same run.
- Added future theorem tracks:
  `docs/proofs/23-reasoning-state-reuse-theorem.md` for general reasoning
  enhancement and `docs/proofs/24-canonicalized-engram-retrieval-theorem.md`
  for knowledge retrieval enhancement.
- Hardened `alien_static` with configurable key/value space and train/test key
  split. In the hard setting, Engram gives a weak seen-fact causal gain
  (`0.992` EM vs `0.969` with Engram disabled), while test-key EM remains `0.0`.
- Recorded third-party review response in
  `docs/progress/2026-05-04-external-review-response.md`. Accepted the core
  downgrade: current results support bounded signal paths and weak seen-fact
  memory, not global O(1) long-context guarantees or mature LLM-scale claims.
- Added the next proof-tightening layer:
  `25-general-llm-replacement-necessary-conditions.md`,
  `26-tight-engram-concentration-bound.md`, and
  `27-gated-retention-iss-stability.md`. These define the ladder toward a
  possible resource-conditional replacement claim, replace loose Engram tail
  reasoning with high-probability concentration assumptions, and add ISS-style
  anti-saturation conditions for gated RetNet state.
- Corrected MoE positioning: MoE is deferred for engineering isolation, not
  rejected. Dense FFN is the phase-1 control baseline; a phase-2 MoE
  reintegration proof must handle routing robustness under Engram/AttnRes and
  snapshot perturbations.

## Next Step

Stabilize and audit the snapshot readout:

- run fixed held-out evaluation after each logged train step, not only training
  batch exact match;
- ablate residual-only snapshot readout versus logit-bias snapshot readout;
- run the same setup on XOR/tree reasoning and alien dictionary tasks to detect
  exact-copy shortcut overfitting;
- measure snapshot attention mass, snapshot scale, branch norms, and module-drop
  deltas.
