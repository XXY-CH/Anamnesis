# Memory and Retrieval Pipeline

This directory implements the external memory compiler and retrieval pipeline used by Anamnesis to scale context lengths up to 1M tokens.

## Contents

- [context_compiler.py](file:///Users/xiexingyu/Documents/项目/Resources/src/memory/context_compiler.py) - Preprocesses long text sequences into structured (key, value, position) memory representations (Proof 34).
- [chunk_retriever.py](file:///Users/xiexingyu/Documents/项目/Resources/src/memory/chunk_retriever.py) - A learned Grouped Cross-Attention (GCA) inspired retriever that scores context chunks and performs high-fidelity token logits readout.

## Pipeline Workflow

1. **Chunking**: The long-context document is split into non-overlapping chunks.
2. **Encoding & Storage**: Each chunk is encoded and its relevant features stored as external memory keys and values.
3. **Chunk Selection**: A lightweight `ChunkRetriever` (GCA-based) utilizes contrastive query similarity to retrieve the target chunk containing the reasoning fact.
4. **Token Readout**: Self-similarity projection in embedding space extracts high-fidelity logits for token injection at prediction time.
