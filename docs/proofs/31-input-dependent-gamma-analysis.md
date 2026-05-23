# Input-Dependent Gamma: Expressiveness, Gradient Analysis, and Architecture Role

Created: 2026-05-23

Status: formal analysis of making RetNet decay rate γ input-dependent
(analogous to Mamba's selective SSM parameters). Establishes the SNR bound,
the vanishing-gradient bottleneck, and the correct architectural role.

## 0. Model

Standard RetNet recurrent state update:

```
S_t = γ · S_{t-1} + k_t ⊗ v_t
```

Input-dependent variant:

```
S_t = γ(x_t) · S_{t-1} + k_t ⊗ v_t
```

where γ(x_t) = σ(W_γ x_t + b_γ) ∈ (0, 1) with σ the sigmoid function.

The parallel-mode attention weight from position i to position j (i ≥ j) is:

```
A_{i,j} = Π_{r=j+1}^{i} γ(x_r)         (input-dependent)
A_{i,j} = γ^{i-j}                        (fixed)
```

### Notation

- `D = q - p`: distance from needle (position p) to query (position q)
- `γ_h`: fixed decay rate for head h
- `γ(x_t)_h`: input-dependent decay for head h at position t
- `SNR`: signal-to-noise ratio at the query position

## 1. Theorem: Strict Generalization

**Proposition 1.** Input-dependent γ strictly generalizes fixed γ.

*Proof.* If γ(x_t) = γ₀ for all t, we recover fixed-γ RetNet exactly.
The converse fails: input-dependent γ can implement token-level gating
that no fixed γ can. For a sequence of T tokens where token at position p
must be preserved and all others forgotten, input-dependent γ sets
γ(x_r) = 1 for r ∈ (p, q] and γ(x_r) = 0 otherwise. No single fixed γ
achieves both perfect retention of p and perfect forgetting of others. □

## 2. Theorem: SNR Comparison for Needle Task

**Setting.** Needle at position p with key-value pair (k_p, v_p).
Query at position q = p + D. Filler tokens at positions {0,...,p-1, p+1,...,q-1}.

The retention output at q is:

```
o_q = Σ_{j=0}^{q} A_{q,j} · (q_q^T k_j) · v_j
```

The signal component is A_{q,p} · (q_q^T k_p) · v_p.
The noise component is Σ_{j≠p} A_{q,j} · (q_q^T k_j) · v_j.

### Fixed γ SNR

```
A_{q,p} = γ^D

SNR_fixed ∝ γ^D / Σ_{j≠p} γ^{q-j}
```

For D = 1024, γ = 0.97: γ^D = 0.97^1024 ≈ exp(-31.1) ≈ 3.2×10⁻¹⁴.

The signal has vanished. Even with milestone gate boosting γ = 0.999:
0.999^1024 ≈ exp(-1.02) ≈ 0.36 — significant decay.

### Input-Dependent γ SNR (Optimal Policy)

If the model learns γ(x_r) = γ_high for r ∈ (p, q] and γ(x_r) = γ_low for r ∉ (p, q]:

```
A_{q,p} = γ_high^D
```

For the signal to survive at D = 1024, we need γ_high^D ≥ ε (tolerable decay).
This requires γ_high ≥ exp(log(ε) / D).

For ε = 0.9, D = 1024: γ_high ≥ exp(-0.000105) ≈ 0.999895.
This is σ(9.14).

**Claim.** The logit value needed for near-perfect retention at distance D is:

```
logit_needed ≈ log(D / δ)
```

where δ = 1 - γ_high is the tolerable per-step leakage.

*Proof sketch.* γ_high = 1 - δ. For γ_high^D ≈ exp(-δD) ≥ ε:
δ ≤ -log(ε)/D. Then logit = log(γ_high/(1-γ_high)) ≈ log(1/δ) ≈ log(D/log(1/ε)). □

For D = 1024, ε = 0.9: logit_needed ≈ log(1024/0.105) ≈ 9.2.

The bias is initialized at logit(γ_fixed) ≈ 3.5–6.2 depending on head.
The model must learn to push logits from ~6 to ~9.2. This is achievable
but requires sufficient gradient signal.

## 3. Theorem: Vanishing Gradient Through Cumulative Gamma

This is the key bottleneck.

**Proposition 3 (Gradient Chain).** Let L be the loss at the query position.
The gradient of L with respect to the logit s_r = W_γ x_r + b_γ at position
r between the needle (p) and query (q) is:

```
∂L/∂s_r = ∂L/∂o_q · (∂o_q/∂γ(x_r)) · γ(x_r)(1 - γ(x_r))
```

The critical term is ∂o_q/∂γ(x_r), which involves:

```
∂/∂γ_r [Π_{j=p+1}^{q} γ(x_j)] = (Π_{j≠r} γ(x_j)) = A_{q,p} / γ(x_r)
```

Therefore:

```
||∂L/∂s_r|| ≤ C · A_{q,p} · |γ(x_r)(1-γ(x_r))| / γ(x_r)
           = C · A_{q,p} · (1-γ(x_r))
```

When γ(x_r) ≈ 1 (the regime we want for preservation):
- A_{q,p} ≈ γ_high^D, which is small for large D
- (1-γ(x_r)) ≈ δ, which is also small

**The gradient magnitude scales as O(δ · exp(-δD)).**

This is maximized at δ = 1/D, giving gradient ≈ O(1/D · exp(-1)) = O(1/D).

**Corollary.** The gradient signal for learning input-dependent γ decays as O(1/D)
with the needle-query distance. At D = 1024, the gradient is ~8× weaker than
at D = 128.

This explains the empirical observation: input-dependent γ works at short
sequences but struggles at long ones.

## 4. Theorem: Composition with Milestone Gate + TokenCopyBuffer

The milestone gate (proof 20) and TCB (proof 29) were designed precisely to
bypass the cumulative gamma chain. Their composition with input-dependent γ
creates a three-level memory hierarchy:

```
Level 1: Input-dependent γ   — local selective filtering (O(100) tokens)
         Learns token-level forget/retain decisions.
         Gradient bottleneck: O(1/D).

Level 2: Milestone gate       — medium-range preservation (O(1000) tokens)
         Sets γ = γ_milestone > γ_fixed for a TTL window after milestones.
         Bypasses gradient chain via hard-coded gate schedule.

Level 3: TokenCopyBuffer      — long-range exact recall (O(10K+) tokens)
         Direct addressable copy path, no decay chain at all.
         Bypasses retention entirely.
```

**Proposition 4 (Complementarity).** Input-dependent γ is most effective as a
LOCAL selective filter within the milestone-protected window, NOT as the
primary long-range memory mechanism.

*Proof sketch.* Within a milestone window of length W:
- The milestone gate ensures γ ≥ γ_milestone, preventing catastrophic decay
- Input-dependent γ can selectively boost or reduce γ within [γ_milestone, 1)
- The gradient bottleneck is O(1/W) instead of O(1/D), manageable for W ≤ 256

The three mechanisms compose without interference because:
1. `_resolve_gate` takes max(milestone_gate, dynamic_gamma), so milestone
   provides a floor and dynamic_gamma can only boost above it
2. TCB operates on the embedding space, independent of retention state
3. Each mechanism addresses a different failure mode of fixed-γ retention □

## 5. Design Implication: Per-Layer Input-Dependent γ

**Proposition 5 (Layer Specialization).** Different layers should learn
different γ policies. Early layers act as high-pass filters (low γ, forget
noise). Late layers act as integrators (high γ, accumulate evidence).

*Argument.* Consider the gradient flow through layer l:

```
∂L/∂W_γ^l = ∂L/∂x_L · (Π_{l'=l}^{L-1} ∂x_{l'+1}/∂x_{l'}) · ∂x_l/∂γ_l · ∂γ_l/∂W_γ^l
```

Each layer receives the residual stream from all previous layers. Early layers
see raw token embeddings and should learn surface-level filtering. Late layers
see abstracted features and should learn evidence accumulation.

This is analogous to Mamba's finding that different SSM layers learn different
temporal dynamics.

## 6. Experimental Predictions

From the above analysis:

1. **Input-dependent γ alone** improves needle@128 (D ≈ 100) but plateaus
   at needle@1024+ (D ≈ 900+) due to gradient vanishing. **Confirmed.**

2. **Input-dependent γ + milestone + TCB** should outperform milestone + TCB
   alone, because input-dependent γ provides local selectivity within the
   protected window while milestone/TCB handle long-range.

3. **Per-layer γ visualization** should show early layers with lower average γ
   and later layers with higher average γ.

4. **Gradient norm analysis** should show γ gradients decreasing with distance D.

## 7. Corrected Parameterization

The sigmoid parameterization γ = σ(s) has a subtle issue: its gradient
σ(s)(1-σ(s)) vanishes when γ → 1 (since 1-σ(s) → 0). This compounds the
O(1/D) vanishing from Section 3.

Better: use a log-space parameterization that decouples the gradient from
the value:

```
γ(x) = exp(-exp(-s(x)))    where s(x) = W_γ x + b_γ

∂γ/∂s = exp(-exp(-s)) · exp(-s) = γ · exp(-s)
```

When s → +∞ (γ → 1): gradient = 1 · 0 = 0. Still vanishes.
When s = 0 (γ = exp(-1) ≈ 0.37): gradient = 0.37 · 1 = 0.37.
When s → -∞ (γ → 0): gradient → 0.

The fundamental issue is that any smooth bijection (0,1) → ℝ has vanishing
gradient near the boundaries. This is unavoidable for ANY parameterization.

**The real solution is not a better parameterization but the architectural
composition in Section 4**: use milestone/TCB for long-range and let
input-dependent γ handle only local selectivity.

## 8. Summary

| Property | Fixed γ | Input-dependent γ | Milestone + TCB |
|----------|---------|-------------------|-----------------|
| Long-range SNR | γ^D → 0 | γ(x)^D → 0 (same issue) | Protected by design |
| Gradient at D=1024 | N/A | O(1/D) ≈ 0.001 | Direct (no chain) |
| Local selectivity | None | Strong | None |
| Extra parameters | 0 | d_model × n_heads | TTL budget |
| Correct role | Default decay | Local token filter | Long-range memory |

**Conclusion.** Input-dependent γ is a useful but LOCAL mechanism. It should
be kept as a complementary component in the architecture, always paired with
milestone gate + TCB for long-range recall. Do not expect it to solve
long-range memory by itself.
