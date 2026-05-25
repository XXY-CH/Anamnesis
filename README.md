<div align="center">

# Engram Retention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20041183.svg)](https://doi.org/10.5281/zenodo.20041183)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](LICENSE)
[![CI](https://github.com/XXY-CH/Anamnesis/actions/workflows/ci.yml/badge.svg)](https://github.com/XXY-CH/Anamnesis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](requirements.txt)

**A proof-aligned PyTorch research scaffold for budgeted long-context memory.**

RetNet recurrence + Block Attention Residuals + hashed Engram lookup +
milestone snapshots.

</div>

## What This Is

Engram Retention is an early-stage research codebase for studying whether small,
auditable auxiliary-memory paths can improve sparse long-context recall without
requiring a full KV cache over every previous token.

The project combines implementation, tests, synthetic diagnostics, proof notes,
and citation metadata in one repository. The goal is not just to propose an
architecture, but to keep each claim close to code that can be run, ablated, and
falsified.

## What This Is Not

This repository does **not** claim a universal replacement for Transformer
attention or general-purpose LLM memory. The active claim is conditional and
bounded:

- RetNet-style recurrence handles compact streaming sequence mixing.
- Block Attention Residuals reuse depth-wise intermediate states.
- Hashed Engram lookup injects deterministic N-gram memory as a residual branch.
- Milestone gates and snapshots preserve a small selected set of token-time
  facts under explicit budgets.

## Architecture At A Glance

```text
input token ids
    |
    v
token embedding + position embedding
    |
    v
Dense RetNet-Engram layers
    |
    |-- RetentionLayer
    |     parallel/recurrent sequence mixing with fixed per-head decay
    |
    |-- Dense FFN
    |     phase-1 channel-mixing baseline; MoE is deferred
    |
    |-- HashedNgramEngram
    |     deterministic N-gram lookup with gated residual injection
    |
    |-- BlockAttentionResidual
    |     depth-axis readout over earlier block/layer states
    |
    |-- MilestoneRetentionGate
    |     optional selected-token decay protection
    |
    |-- MilestoneSnapshotReadout
    |     bounded snapshot cache plus readout branch
    |
    v
RMSNorm + output projection
    |
    v
next-token logits + diagnostic metrics
```

## Repository Map

| Path | Role |
|---|---|
| [src/](src/README.md) | PyTorch implementation surface. |
| [src/layers/](src/layers/README.md) | Retention, Engram, AttnRes, milestone gate, and snapshot primitives. |
| [src/models/](src/models/README.md) | Full RetNet-Engram model and Transformer baseline. |
| [src/training/](src/training/README.md) | Lightweight toy training helpers. |
| [experiments/](experiments/README.md) | Synthetic diagnostic runner and experiment configs. |
| [analysis/](analysis/README.md) | Plotting scripts for curated figures. |
| [tests/](tests/README.md) | Unit and integration smoke tests. |
| [docs/](docs/README.md) | Methodology, architecture notes, proof trail, and progress records. |
| [docs/proofs/](docs/proofs/README.md) | Theorem drafts, assumption audits, and proof closure notes. |
| [papers/](papers/README.md) | Human-written reading notes over the local literature corpus. |
| [references/](references/README.md) | BibTeX and paper manifest; mirrored PDFs are not committed. |

PDF papers, local virtual environments, generated experiment results, temporary
extraction text, and agent runtime state are intentionally excluded from Git.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a lighter environment:

```bash
python -m pip install -r requirements.txt
```

## Verify

```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

Expected baseline: all tests pass. The current public test suite covers
retention decay, recurrent state shape, Engram determinism, Block AttnRes
readout, milestone snapshots, full model forward/backward, and toy training
updates.

## Run A Synthetic Diagnostic

Small smoke run:

```bash
python experiments/train_synthetic.py \
  --task needle \
  --variants ours,retnet,transformer \
  --steps 5 \
  --batch-size 4 \
  --seq-len 32 \
  --log-interval 5 \
  --out-dir experiments/results/smoke
```

Snapshot readout ablation:

```bash
python experiments/train_synthetic.py \
  --task needle \
  --variants ours_snapshot_logits,retnet,transformer \
  --use-milestones \
  --use-snapshots \
  --use-snapshot-logit-bias \
  --steps 200 \
  --eval-batches 4 \
  --out-dir experiments/results/needle_snapshot_eval
```

Generated result directories are ignored by default. Commit only curated
summaries, paper-ready figures, or documented conclusions.

## Research Trail

Recommended reading order:

1. [docs/proofs/proof_plan.md](docs/proofs/proof_plan.md)
2. [docs/proofs/pdf_assumption_audit_2026-05-03.md](docs/proofs/pdf_assumption_audit_2026-05-03.md)
3. [docs/proofs/25-general-llm-replacement-necessary-conditions.md](docs/proofs/25-general-llm-replacement-necessary-conditions.md)
4. [docs/progress/2026-05-04-external-review-response.md](docs/progress/2026-05-04-external-review-response.md)
5. [docs/proofs/15-proof-closure.md](docs/proofs/15-proof-closure.md)

The proof trail deliberately separates:

- RetNet recurrence from exact token-time recall.
- Depth-axis Block AttnRes from token-time memory.
- Engram lookup from final-layer logit bias.
- Dense phase-1 evidence from future MoE capacity extensions.

## Citation

If this repository helps your work, cite the Zenodo DOI:

```bibtex
@software{xie_2026_engram_retention,
  author = {Xie, Xingyu},
  title = {Engram Retention},
  year = {2026},
  doi = {10.5281/zenodo.20041183},
  url = {https://github.com/XXY-CH/Anamnesis},
  license = {CC-BY-4.0}
}
```

The same metadata is available in [CITATION.cff](CITATION.cff).

## License

Creative Commons Attribution 4.0 International (`CC-BY-4.0`). See
[LICENSE](LICENSE).
