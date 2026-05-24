# Proof 35: Delta Rule Update — Gated State Overwrite

## Problem Statement

Standard RetNet accumulates state additively: S_t = γ·S_{t-1} + k_t ⊗ v_t.
For long sequences, old state entries decay but are never overwritten. This limits
the model's ability to "forget" irrelevant information and replace it with new content.

The Gated DeltaNet (ICLR 2025) proposes the delta rule:

$$S_t = \gamma \cdot S_{t-1} + \beta_t \cdot (k_t \otimes v_t - S_{t-1})$$

Rearranging:

$$S_t = (\gamma - \beta_t) \cdot S_{t-1} + \beta_t \cdot k_t \otimes v_t$$

where β_t = σ(W_β x_t + b_β) ∈ (0, 1) is input-dependent.

## Analysis

### Case 1: β_t = 0 (no write)
S_t = γ·S_{t-1}. The state simply decays — no new information is written.
Equivalent to a "skip" token in Mamba's selective mechanism.

### Case 2: β_t = 1 (full overwrite)
S_t = (γ - 1)·S_{t-1} + k_t ⊗ v_t.
Since γ < 1 (decay), γ - 1 < 0, so old state is *negatively* weighted.
For γ ≈ 1 (slow decay): S_t ≈ 0·S_{t-1} + k_t ⊗ v_t = k_t ⊗ v_t.
Complete state replacement.

### Case 3: β_t ≈ γ (balanced)
S_t ≈ 0·S_{t-1} + γ·k_t ⊗ v_t = γ·k_t ⊗ v_t.
Old state fully erased, new entry scaled by γ.

### Theorem (Effective Decay Range)

For the delta rule with β ∈ (0, 1) and γ ∈ (0, 1):

**Effective decay**: α_t = γ - β_t ∈ (γ - 1, γ) ⊂ (-1, 1)

When β_t > γ, the effective decay α_t becomes **negative**, causing the state
to oscillate rather than monotonically decay. This is potentially unstable.

**Constraint for stability**: β_t ≤ γ for all t ensures α_t ≥ 0 (monotonic decay).

### Gradient Analysis

For the delta rule state update, the gradient of the loss L with respect to β_t:

$$\frac{\partial L}{\partial \beta_t} = \frac{\partial L}{\partial S_t} \cdot \frac{\partial S_t}{\partial \beta_t}$$

$$\frac{\partial S_t}{\partial \beta_t} = -S_{t-1} + k_t \otimes v_t = (k_t \otimes v_t - S_{t-1})$$

This gradient measures the **mismatch** between the new key-value pair and the
current state. When the new content is very different from existing state, the
gradient pushes β_t toward 1 (full overwrite). When similar, it pushes toward 0.

This is a *local* gradient — it doesn't suffer from the O(1/D) vanishing problem
of Proof 31 because β only appears in one time step.

### Theorem (Bounded State Norm)

For the delta rule with ||k_t|| ≤ K_max, ||v_t|| ≤ V_max, and β_t ≤ γ:

$$||S_t|| \leq \sum_{s=1}^{t} \gamma^{t-s} \cdot \beta_s \cdot K_{max} \cdot V_{max}$$

$$\leq K_{max} \cdot V_{max} \cdot \sum_{s=1}^{t} \gamma^{t-s}$$

$$\leq \frac{K_{max} \cdot V_{max}}{1 - \gamma}$$

Same upper bound as standard RetNet. The state is well-controlled.

### Complexity

The delta rule adds one linear projection (d_model → n_heads) per retention layer,
plus one sigmoid activation. Total extra parameters: n_layers × d_model × n_heads.

For d=64, n_heads=4, n_layers=8: 8 × 64 × 4 = 2,048 extra parameters.

## Key Differences from Standard RetNet

| Property | Standard RetNet | Delta Rule |
|----------|----------------|------------|
| State update | S = γS + k⊗v | S = (γ-β)S + β·k⊗v |
| Write strength | Always 1 | Input-dependent β ∈ (0,1) |
| Forget mechanism | Only via γ decay | Explicit via β overwrite |
| State replacement | Never (only decay) | Possible (β → 1) |
| Extra params | 0 | d × n_heads × n_layers |
| Stability | Unconditionally stable | Requires β ≤ γ for monotonicity |

## Hypothesis

The delta rule should help on long-sequence recall tasks because:
1. It can overwrite irrelevant state entries instead of just decaying them
2. The model learns when to strongly write (high β for important tokens)
3. Less state "pollution" from irrelevant filler tokens

Risk: If β > γ, effective decay becomes negative → potential instability.
Mitigation: Initialize beta_proj.bias to 0 (sigmoid(0) = 0.5), and monitor.
