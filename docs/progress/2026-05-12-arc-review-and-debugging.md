# ARC Review And Debugging Pass

Date: 2026-05-12

Source reviewed: `/Users/xiexingyu/Downloads/ARC 条件小状态记忆.docx`

## 5.5pro Result Summary

The document proposes ARC, Addressed Reasoning Controller, as a unifying module
for:

```text
when to write
what to write
when to read
which memory slots to read
how to fuse retrieved values into reasoning
```

Its proposed ARC parts are:

| Part | Useful idea | Required downgrade |
|---|---|---|
| Memory Manager | Bounded slots with metadata and eviction. | Must expose fixed caps and eviction metrics. |
| Write Gate | Uses surprisal, utility, recency, and gate score. | Needs anti-collapse proof and false-positive telemetry. |
| Read Policy | Sparse top-k read over bounded memory. | Needs address-margin proof; cannot attend over all history. |
| Reasoning Fusion | Gated residual fusion of snapshot, depth, and Engram outputs. | Needs downstream logit/decision margin and module-drop evidence. |

## Decision

ARC is accepted as a target control surface, not as an already validated module.
The proof stack now treats ARC as a bounded controller with these obligations:

```text
fixed resource caps
address margin
fusion margin
anti-collapse stability
capture -> keep -> address -> fuse -> decision telemetry
```

## Proof Updates

- Added `docs/proofs/30-addressed-reasoning-controller.md`.
- Updated `docs/proofs/proof_plan.md` so ARC becomes an explicit future proof
  obligation after exact-token readout and before replacement claims.
- Updated `docs/proofs/formal_hybrid_architecture_proofs.tex` to include ARC in
  the title, abstract, budget contract, contribution list, and architecture
  discussion.
- Updated `docs/architecture/current_model_architecture_2026-05-11.md` with ARC
  as a target controller, explicitly marked not implemented.

## Debugging Step

Root-cause hypothesis:

```text
The immediate long-context failure is likely not only "memory was not stored".
It may be "memory was stored but the query-side readout selected the wrong slot
or lacked enough downstream logit margin."
```

Minimal instrumentation added:

```text
token_copy_weights
token_copy_valid
token_copy_pos_ids
token_copy_valid_count
```

These diagnostics are emitted by `AnamnesisModel(..., return_diagnostics=True)`
when TokenCopyBuffer is active. This makes correct-slot attention measurable in
the next length-scaling experiment.

Added probe:

```text
experiments/probe_copy_alignment.py
```

This trains a small Needle model and reports exact match, valid source-slot hit
rate, correct-slot attention mass, answer-position attention entropy, and valid
slot count across evaluation lengths.

## Verification

Commands run:

```text
git diff --check
python -m pytest tests/test_aligned_architecture.py tests/test_budget_contracts.py tests/test_retention.py
python experiments/probe_copy_alignment.py --device cpu --train-seq-len 16 --train-steps 1 --eval-lengths 16,32 --eval-batches 1 --batch-size 2 --d-model 16 --n-layers 2
```

Result:

```text
17 passed
copy-alignment smoke emitted token-copy metrics for 16 and 32 token contexts:
  16 tokens: EM 0.000, slot_hit 1.000, slot_mass 0.201, entropy 1.237
  32 tokens: EM 0.000, slot_hit 1.000, slot_mass 0.163, entropy 1.201
```

The smoke run was intentionally undertrained. Its value is diagnostic plumbing:
the expected source slots are present (`slot_hit=1.000`), while correct-slot mass
is low. Longer runs can now distinguish "missing memory" from "weak addressing"
and "weak fusion margin."

## Next Debugging Experiment

Run a short train-1024, eval-2048/4096/8192 sweep with copy diagnostics enabled.
For the Needle task, compute:

```text
correct-slot attention mass
copy attention entropy
token_copy_valid_count
copy-logit margin if available
eval exact match
```

If correct-slot attention collapses while valid slots are present, the next
implementation target is a relative-role or pointer-style ARC readout. If
correct-slot attention is high but exact match is low, the next target is fusion
or logit-margin control.
