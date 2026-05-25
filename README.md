<div align="center">

# Anamnesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20041183.svg)](https://doi.org/10.5281/zenodo.20041183)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](LICENSE)
[![CI](https://github.com/XXY-CH/Anamnesis/actions/workflows/ci.yml/badge.svg)](https://github.com/XXY-CH/Anamnesis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](requirements.txt)

**A proof-aligned PyTorch research scaffold for budgeted long-context memory.**

RetNet recurrence + Kimi-style Block Attention Residuals + DeepSeek-style gated Engram lookup + RAG separation.

</div>

---

## What This Is

**Anamnesis** (from the Greek philosophy of *recalling innate knowledge*) is an early-stage research codebase studying whether small, auditable auxiliary-memory paths can improve sparse long-context recall without requiring a full KV cache over every previous token.

The project tightly integrates implementation, test suites, synthetic diagnostics, proof notes, and citation metadata in one repository. The goal is to keep each architectural claim closely bound to code that can be run, ablated, and empirically falsified.

---

## Key Pillars of the Architecture

*   **RetNet Recurrence**: Handles streaming, horizontally-causal sequence mixing with linear $O(1)$ inference cost, supporting both fixed decay rates and dynamic input-dependent decay.
*   **Block Attention Residuals (Kimi Style)**: Simplifies the vertical residual stream by replacing uniform addition with zero-parameter softmax attention over preceding layer/block outputs.
*   **Hashed N-gram Engram (DeepSeek Style)**: Retrieves static, context-independent priors via Deterministic Multi-Head hashing, fuzed with dynamic hidden states via an **isotropic scalar gate** (Eq 4) and a **causal depthwise Conv1D** (Eq 5) to preserve semantic direction.
*   **RAG Separation Pipeline**: Decouples discrete similarity retrieval from continuous language generation, prepending relevant text chunks directly in the prompt to preserve the contextual Jacobian chain and avoid logit rank deficiency.
*   **Milestone Gates and Copy Buffers**: Protects critical position decays and implements precise, budget-bounded episodic fact recall under explicit resource constraints.

---

## Architecture At A Glance

Our unified dual-brain architecture (Anamnesis) coordinates the sequence, depth, and memory axes:

```mermaid
graph TB
    %% Styling definitions
    classDef inputStyle fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef embStyle fill:#313244,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef blockStyle fill:#181825,stroke:#f38ba8,stroke-width:3px,color:#cdd6f4;
    classDef layerStyle fill:#313244,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef memStyle fill:#11111b,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4;
    classDef outputStyle fill:#1e1e2e,stroke:#fab387,stroke-width:2px,color:#cdd6f4;

    %% Nodes and relationships
    Input["Input Tokens [batch, seq_len]"]:::inputStyle
    Input --> Embed["Token Embedding + PE (sinusoidal / learned)"]:::embStyle
    Embed --> Block1["Anamnesis Block (DenseRetNetEngramLayer x N)"]:::blockStyle

    subgraph Block ["Inside the Anamnesis Layer Block"]
        direction TB
        X["Hidden States (x)"]
        
        %% Retention Path
        X --> Norm1["RMSNorm"]
        Norm1 --> Ret["RetentionLayer<br/>(Sequence Axis: Causal Decay & Dynamic Gamma)"]:::layerStyle
        
        %% FFN Path
        X --> Norm2["RMSNorm"]
        Norm2 --> FFN["Dense FFN<br/>(Channel Axis: SiLU Activation)"]:::layerStyle
        
        %% Sum 1
        Ret --> Sum1["+"]
        FFN --> Sum1
        X --> Sum1
        
        %% Engram Path (Selected Layers, e.g., Layer 2)
        Sum1 -->|"ffn_norm(x)"| EngramNorm["RMSNorm"]
        EngramNorm --> Engram["HashedNgramEngram<br/>(Memory Axis: Deterministic Multi-Head Lookup)"]:::memStyle
        
        subgraph EngramSub ["Engram Fusion (DeepSeek Eq 4 & 5)"]
            direction TB
            Key["Projected Key: W_K e_t"] --> KeyNorm["RMSNorm"]
            Val["Projected Value: W_V e_t"]
            Score["Dot-product Score: (RMSNorm(h) * RMSNorm(key)) / sqrt(d)"]
            Score --> Gate["Isotropic Scalar Gate:<br/>sigmoid(score + b) (b = -3.0)"]:::layerStyle
            Gate --> Mult["Gated Value:<br/>gate * Val"]
            Mult --> Conv["Causal Depthwise Conv1D<br/>(kernel=4, dilation=3, groups=d_model)"]:::layerStyle
            Conv --> SiLU["SiLU Activation"]
            SiLU --> Y["Engram Output (Y)"]
        end
        Engram --> EngramSub
        
        Sum1 --> AddEngram["+"]
        Y -->|Residual Scale s| AddEngram
        
        %% AttnRes Path (Every 4 Layers, e.g., Layer 3, 7)
        AddEngram --> AttnRes["BlockAttentionResidual<br/>(Depth Axis: Kimi Softmax Attn)"]:::memStyle
        
        subgraph AttnResSub ["AttnRes Readout"]
            direction TB
            Stacked["Stacked Previous Block Outputs"] --> AttnNorm["RMSNorm"]
            Query["Pseudo-Query w_l"]
            Query --> Softmax["Softmax Weighted Blending"]
            AttnNorm --> Softmax
            Softmax --> Readout["Depth Readout (zero parameters, no value projections)"]
        end
        AttnRes --> AttnResSub
        
        AddEngram --> AddAttnRes["+"]
        Readout -->|Residual Scale s| AddAttnRes
        
        %% Depth Cache
        AddAttnRes --> DepthSource["depth_sources.append(x)"]
    end
    
    Block1 --> Block
    AddAttnRes --> FinalNorm["Final RMSNorm"]:::outputStyle
    
    %% Episodic Readouts
    FinalNorm --> Head["Output Head (Linear Logits)"]:::outputStyle
    FinalNorm -->|"optional"| TCB["Token Copy Buffer<br/>(Episodic Memory cosine lookup)"]:::memStyle
    FinalNorm -->|"optional"| Snapshot["Milestone Snapshot Readout<br/>(State snapshot cache)"]:::memStyle
    
    Head --> SumLogits["+"]
    TCB -->|copy_logits| SumLogits
    Snapshot -->|snapshot_logits| SumLogits
    
    SumLogits --> Output["Vocabulary Logits"]:::outputStyle
```

---

## Repository Map

| Path | Role |
|---|---|
| [src/](src/README.md) | PyTorch implementation surface. |
| [src/layers/](src/layers/README.md) | Retention, Engram, AttnRes, milestone gate, and snapshot primitives. |
| [src/models/](src/models/README.md) | Full RetNet-Engram model and Transformer baseline. |
| [src/training/](src/training/README.md) | Lightweight toy training helpers. |
| [experiments/](experiments/README.md) | Synthetic diagnostic runner and experiment configs. |
| [analysis/](analysis/README.md) | Plotting scripts for curated figures. |
| [tests/](tests/README.md) | Unit, integration, and paper-faithfulness tests. |
| [docs/](docs/README.md) | Methodology, architecture notes, proof trail, and progress records. |
| [docs/proofs/](docs/proofs/README.md) | Theorem drafts, assumption audits, and proof status registry. |
| [papers/](papers/README.md) | Human-written reading notes over the local literature corpus. |
| [references/](references/README.md) | BibTeX and paper manifest; mirrored PDFs are not committed. |

---

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

---

## Verify

```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

All 25 unit and paper-faithfulness tests should pass. The test suite covers retention decay, recurrent state shape, Engram determinism, Block AttnRes readout, milestone snapshots, paper-faithfulness configurations, full model forward/backward, and toy training updates.

---

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

---

## Research Trail

Recommended reading order in `docs/proofs/`:

1. [00-proof-status-registry.md](docs/proofs/00-proof-status-registry.md) (All 42 proofs tracked here)
2. [proof_plan.md](docs/proofs/proof_plan.md)
3. [pdf_assumption_audit_2026-05-03.md](docs/proofs/pdf_assumption_audit_2026-05-03.md)
4. [40-scalar-gating-lower-bound.md](docs/proofs/40-scalar-gating-lower-bound.md)
5. [41-rag-separation-principle.md](docs/proofs/41-rag-separation-principle.md)
6. [42-rope-snr-collapse.md](docs/proofs/42-rope-snr-collapse.md)
7. [15-proof-closure.md](docs/proofs/15-proof-closure.md)

---

## Citation

If this repository helps your work, cite the Zenodo DOI:

```bibtex
@software{xie_2026_anamnesis,
  author = {Xie, Xingyu},
  title = {Anamnesis},
  year = {2026},
  doi = {10.5281/zenodo.20041183},
  url = {https://github.com/XXY-CH/Anamnesis},
  license = {CC-BY-4.0}
}
```

The same metadata is available in [CITATION.cff](CITATION.cff).

---

## License

Creative Commons Attribution 4.0 International (`CC-BY-4.0`). See [LICENSE](LICENSE).
