# Proof 38: Learned Gating to Replace MARK_THOUGHT

## Problem Statement

The TokenCopyBuffer (TCB) currently relies on MARK_THOUGHT tokens to identify
which positions to store. This is an oracle annotation that doesn't exist in
real data. We need a learned mechanism that identifies important positions
without explicit markers.

## Why MARK_THOUGHT Exists

At position t during the forward pass, the model must decide: should this
token be stored in the TCB for later retrieval? The model doesn't know which
tokens will be queried. MARK_THOUGHT provides this at training time (supervised).
Without it, the model needs to learn from the downstream loss (self-supervised).

## Analysis

### Approach: Gumbel Top-K with Learned Scoring

**Scoring**: $s_t = \sigma(W_g \cdot h_t + b_g) \in (0, 1)$

**Gumbel selection** (temperature τ):
$$p_t = \frac{\exp((\log s_t + g_t) / \tau)}{\sum_{t'} \exp((\log s_{t'} + g_{t'}) / \tau)}$$

As τ → 0: hard top-K. At τ > 0: soft, differentiable approximation.

**Temperature schedule**:
- Early (0-40%): τ=2.0 (soft, all tokens contribute)
- Mid (40-80%): τ=0.5 (peaky, high-score tokens dominate)
- Late (80-100%): τ=0.1 (near-hard, effectively top-K)

### Gradient Flow

$$\frac{\partial L}{\partial W_g} = \frac{\partial L}{\partial \text{logits}} \cdot \frac{\partial \text{logits}}{\partial \text{readout}} \cdot \frac{\partial \text{readout}}{\partial \text{stored}} \cdot \frac{\partial \text{stored}}{\partial s_t} \cdot \frac{\partial s_t}{\partial W_g}$$

Gumbel approximation ensures ∂stored/∂s_t ≠ 0 for ALL positions (soft
selection), avoiding dead gradient problem of hard thresholding.

### Convergence Argument

The LM loss at the answer position provides clear signal:
- Password tokens in TCB → answer correct → reinforce storage score
- Password tokens NOT in TCB → answer wrong → increase their score
- Filler tokens in TCB → no help → decrease their score

This is reinforcement from the training data. The gate learns: "which tokens,
if stored, reduce the loss?"

### Alternative Approaches Considered

| Approach | Params | Needs Oracle | Pros | Cons |
|----------|--------|-------------|------|------|
| Surprise-based | 0 | No | Simple | Filler also surprising |
| γ inverse | 0 | No | Reuses γ | Low γ ≠ important |
| Attention-based | 0 | No | Uses model attn | No attn in recurrent mode |
| **Gumbel top-K** | **d+1** | **No** | **Learned, differentiable** | **Needs annealing** |

### Implementation Sketch

```python
class LearnedTokenGate(nn.Module):
    def __init__(self, d_model, max_store=8):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1, bias=True)
        self.max_store = max_store
        self.temperature = 2.0

    def forward(self, hidden):
        scores = self.score_proj(hidden).squeeze(-1)  # [B, T]
        scores = torch.sigmoid(scores)

        if not self.training:
            # Hard top-K at inference
            _, indices = scores.topk(self.max_store, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            return mask, scores

        # Soft Gumbel selection during training
        noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        soft = torch.softmax((scores.log() + noise) / self.temperature, dim=-1)
        return soft, scores
```

## Key Results

The learned gate replaces MARK_THOUGHT with a 65-parameter network (d=64 + bias).
Gradient signal flows from the answer loss through TCB readout to the gate.
Temperature annealing ensures convergence: soft start, hard finish.

## Next Steps

1. Implement `LearnedTokenGate` in `src/layers/`
2. Replace `_content_before_milestone_mask()` with learned gate
3. Test on needle@512 (should match MARK_THOUGHT baseline)
4. Test on real data (no oracle annotations needed)
