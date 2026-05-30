# Layers

This folder contains reusable neural network components.

## Files

- [retention.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/retention.py) - RetNet-style parallel, recurrent, and chunkwise retention layer with multi-scale decay.
- [engram.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/engram.py) - Deterministic hashed N-gram Engram residual branch for semantic-conserving long-context memory.
- [attention_residual.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/attention_residual.py) - Block Attention Residual depth-reuse branch operating directly on raw preceding block states.
- [milestone_gate.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/milestone_gate.py) - Optional milestone-conditioned retention gate for context persistence.
- [milestone_snapshot.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/milestone_snapshot.py) - Bounded snapshot collection and readout from critical milestone steps.
- [token_copy_buffer.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/token_copy_buffer.py) - Token copy buffer supporting high-fidelity exact token retrieval and logit injection.
- [learned_token_gate.py](file:///Users/xiexingyu/Documents/项目/Resources/src/layers/learned_token_gate.py) - Learned token-level gating module approximating optimal oracle memory triggers.

Layer modules should stay small, typed, and independently testable.

