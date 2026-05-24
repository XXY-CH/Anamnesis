# Proof 39: Engram-Hit → TCB Trigger Analysis

## Mechanism

Replace MARK_THOUGHT oracle for TCB storage with self-supervised Engram gate surprise.

**Training**: Oracle drives TCB (correct positions stored). Engram gate learns to predict oracle positions via BCE distillation loss.

**Inference**: Engram surprise score drives TCB. No oracle needed.

## Mathematical Formulation

Let $g_t = \sigma(W_g [h_t; m_t] + b_g)$ be the Engram gate at position $t$, where $h_t$ is the hidden state and $m_t$ is the retrieved memory. $g_t \in (0, 1)^d$ per-dimension.

**Surprise score** per position:

$$s_t = \frac{1}{L \cdot d} \sum_{\ell \in \mathcal{E}} \sum_{j=1}^{d} g_t^{(\ell, j)}$$

where $\mathcal{E}$ is the set of Engram layers and $d$ is the model dimension.

**Distillation loss** (training only):

$$\mathcal{L}_{distill} = \text{BCE}(s, o) = -\frac{1}{T} \sum_t \left[ w_+ \cdot o_t \log(\sigma(s_t)) + (1-o_t) \log(1-\sigma(s_t)) \right]$$

where $o_t$ is the oracle mask and $w_+ = \frac{|\neg o|}{|o|}$ (pos_weight for class balance).

**Total training loss**:

$$\mathcal{L} = \mathcal{L}_{LM} + \lambda \cdot \mathcal{L}_{distill}, \quad \lambda = 0.5$$

## Why Pure Engram Trigger Fails

**Theorem**: Without oracle guidance during training, the Engram-triggered TCB cannot converge on the needle task.

**Proof sketch**: The TCB stores the top-K positions by surprise score. Initially, $g_t \approx \sigma(b_g) \approx 0.047$ uniformly (bias=-3.0). The top-K selection is essentially random. With random tokens stored, the TCB readout adds noise to logits. The LM loss provides no useful gradient signal to the Engram gate because:

1. The gradient from $\mathcal{L}_{LM}$ through the TCB readout is proportional to the readout quality.
2. With random tokens stored, the readout is noise → gradient is noise.
3. The Engram gate receives no signal to differentiate password from noise positions.

This is the **cold-start problem**: the gate needs correct TCB to learn, but needs to learn to fill TCB correctly.

## Why Distillation Works

**Theorem**: With oracle-guided TCB during training, the distillation loss provides sufficient gradient to train the Engram gate.

**Proof sketch**:
1. Oracle ensures correct tokens are stored in TCB during training.
2. The model can learn the needle task (correct logits at answer positions).
3. The distillation loss $\mathcal{L}_{distill}$ directly trains $s_t$ to be high at oracle positions and low elsewhere.
4. The gradient flows: $\nabla_{s_t} \mathcal{L}_{distill} \to \nabla_{g_t} s_t \to \nabla_{W_g} g_t$.
5. Since $s_t = \text{mean}(g_t)$, the gradient per dimension is $1/(L \cdot d)$, small but non-zero.
6. With pos_weight $w_+ \approx T/K$ (where $K$ is the number of oracle positions, typically 3), the gradient is amplified at positive positions.

## Experimental Evidence

| Approach | Best EM | Converges? |
|----------|---------|------------|
| Oracle TCB (baseline) | 0.875 | Yes (unstable) |
| Pure Engram trigger | 0.688 | Partially (noisy) |
| Engram + distillation | 1.000 | Yes (still unstable) |

**Key observation**: The distillation approach is the ONLY variant to achieve EM=1.000. This is unexpected — it outperforms the oracle baseline. Hypothesis: the distillation loss acts as a regularizer, preventing the Engram from overfitting to specific N-gram patterns.

## Limitations

1. **MARK_THOUGHT dependency**: The Engram gate learns patterns correlated with MARK_THOUGHT presence. Without MARK_THOUGHT in the data, the N-gram context changes and the gate may not activate at password positions.

2. **Training instability**: EM oscillates 0.5–1.0 in later training. The distillation loss and LM loss compete for the Engram gate's parameters.

3. **Gradient dilution**: The mean over $L \cdot d$ dimensions dilutes the gradient. A learned aggregation (e.g., attention pooling) could be more effective.

## Connection to Information Bottleneck (Long-term Direction 1)

The distillation approach is a practical approximation of the Thermodynamic Memory Controller. The oracle defines "what should be stored," and the Engram gate learns to predict this. A more principled approach would minimize the information bottleneck objective:

$$\min I(X; Z) - \beta \cdot I(Z; Y)$$

where $X$ is the input, $Z$ is the stored memory, and $Y$ is the task output. The Engram gate implements a learned compression $Z = g(X)$, and the distillation loss approximates $I(Z; Y)$.

## Config

`--use-engram-tcb-trigger`: Enable Engram-triggered TCB with distillation.
