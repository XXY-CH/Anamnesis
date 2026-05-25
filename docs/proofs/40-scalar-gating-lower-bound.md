# Proof 40: Context-Aware Scalar Gating Local Optimality Feasibility and Gradient Bounds

## Proposition

Under scalar dot-product gating with bias initialized to $b \ll 0$, the introduction of Engram conditional memory provides a local first-order feasibility guarantee: at initialization, the model's loss is equal to the baseline loss up to an exponential decay factor $\mathcal{O}(e^b)$, and gradients are strictly directed along directions of descent that improve upon the baseline. Furthermore, we derive the strict gradient vanishing bound to establish the optimal initialization window that avoids complete gate paralysis.

---

## 1. Vector Gate vs. Scalar Gate Representation

### Vector Gate (Anisotropic Scaling — Pollutes Semantic Manifolds)

$$\tilde{v}_t = \sigma(W_g [\text{RMSNorm}(h_t); \text{RMSNorm}(m_t)]) \odot W_V m_t$$

where $W_g \in \mathbb{R}^{d \times 2d}$ produces a per-dimension gate vector $\alpha_t \in (0,1)^d$.

**Mathematical Flaw**: Let $v_t = W_V m_t \in \mathbb{R}^d$ represent a retrieved semantic memory vector lying on a learned pre-trained representation manifold. Under vector gating, the output is scaled element-wise: $\tilde{v}_{t,i} = \alpha_{t,i} v_{t,i}$. This constitutes **anisotropic scaling**. 

Let $\theta$ be the angle between the original semantic direction $v_t$ and the gated output $\tilde{v}_t$:

$$\cos\theta = \frac{v_t^T \tilde{v}_t}{\|v_t\|_2 \|\tilde{v}_t\|_2} = \frac{\sum_{i=1}^d \alpha_{t,i} v_{t,i}^2}{\sqrt{\sum_{i=1}^d v_{t,i}^2} \cdot \sqrt{\sum_{i=1}^d \alpha_{t,i}^2 v_{t,i}^2}}$$

By the Cauchy-Schwarz inequality, $\cos\theta \leq 1$, with equality if and only if $\alpha_{t,i} = \alpha_{t,j}$ for all $i, j$ (scalar gating). For any non-zero variance in $\alpha_t$, $\cos\theta < 1$. Thus, vector gating rotates the semantic vector away from the learned manifold coordinates, injecting out-of-manifold noise into the residual stream and degrading the language model's perplexity (PPL).

### Scalar Dot-Product Gate (Isotropic Scaling — Preserves Semantic Direction)

$$\alpha_t = \sigma\left(\frac{\text{RMSNorm}(h_t)^T \cdot \text{RMSNorm}(W_K m_t)}{\sqrt{d}} + b\right)$$

$$\tilde{v}_t = \alpha_t \cdot (W_V m_t)$$

where $\alpha_t \in (0, 1)$ is a single scalar. The direction of the injected vector is identical to $W_V m_t$:

$$\frac{\tilde{v}_t}{\|\tilde{v}_t\|_2} = \frac{\alpha_t W_V m_t}{\alpha_t \|W_V m_t\|_2} = \frac{W_V m_t}{\|W_V m_t\|_2}$$

This is **isotropic scaling**. The semantic direction of the pre-trained embedding table is preserved, preventing manifold distortion.

---

## 2. Derivation of the Gradient Vanishing Bound

Let $\mathcal{L}$ be the next-token prediction loss, and let the output hidden state after Engram residual injection at layer $l$ be:

$$H_t^{(l)} = H_t^{(l-1)} + s \cdot \tilde{v}_t, \quad \tilde{v}_t = \alpha_t v_t, \quad \alpha_t = \sigma(S_t + b)$$

where $s$ is the trainable residual scale (initialized small, e.g., $10^{-4}$), and $S_t$ is the normalized dot-product score. 

The gradient of the loss with respect to the gating projection weights $W_K$ is computed via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W_K} = \sum_t \frac{\partial \mathcal{L}}{\partial H_t^{(l)}} \cdot \frac{\partial H_t^{(l)}}{\partial \alpha_t} \cdot \frac{\partial \alpha_t}{\partial S_t} \cdot \frac{\partial S_t}{\partial W_K}$$

Substituting the derivatives:
1. $\frac{\partial H_t^{(l)}}{\partial \alpha_t} = s \cdot v_t$
2. $\frac{\partial \alpha_t}{\partial S_t} = \alpha_t (1 - \alpha_t)$
3. $\frac{\partial S_t}{\partial W_K} = \frac{\partial}{\partial W_K} \left( \tilde{h}_t^T \cdot \text{RMSNorm}(W_K m_t) \right)$ where $\tilde{h}_t = \text{RMSNorm}(h_t)$.

Thus, the gradient expression is:

$$\frac{\partial \mathcal{L}}{\partial W_K} = s \cdot \sum_t \left[ \alpha_t(1 - \alpha_t) \cdot \left( \frac{\partial \mathcal{L}}{\partial H_t^{(l)}}^T v_t \right) \cdot \frac{\partial S_t}{\partial W_K} \right]$$

### The Gradient Vanishing Theorem

**Theorem**: If the bias is initialized to $b \ll 0$, then at initialization $\alpha_t \approx \sigma(b) = e^b / (1 + e^b) \approx e^b$. The norm of the gradient with respect to $W_K$ is bounded by:

$$\left\| \frac{\partial \mathcal{L}}{\partial W_K} \right\| \leq s \cdot e^b \cdot C$$

where $C > 0$ is a constant depending on the norms of $\frac{\partial \mathcal{L}}{\partial H_t^{(l)}}$, $v_t$, $h_t$, and $m_t$.

**Proof**:
Since $b \ll 0$, the activation score $S_t$ is close to 0 at initialization (due to random orthornormal projections and RMSNorm bounding the elements). Thus $S_t + b \approx b$.
The derivative of the sigmoid is:
$$\alpha_t(1 - \alpha_t) \approx \sigma(b)(1 - \sigma(b)) \approx e^b (1 - e^b) \leq e^b$$
Taking the norm of the gradient expression:
$$\left\| \frac{\partial \mathcal{L}}{\partial W_K} \right\| \leq s \cdot e^b \cdot \sum_t \left| \frac{\partial \mathcal{L}}{\partial H_t^{(l)}}^T v_t \right| \cdot \left\| \frac{\partial S_t}{\partial W_K} \right\|$$
Setting $C = \sum_t \left| \frac{\partial \mathcal{L}}{\partial H_t^{(l)}}^T v_t \right| \cdot \left\| \frac{\partial S_t}{\partial W_K} \right\|$ yields the bound.

### The Optimization Paradox
- If $b \to -\infty$, the initial loss perturbation vanishes: $\mathcal{L}_0 - \mathcal{L}_{\text{Baseline}} = \mathcal{O}(e^b) \to 0$. This ensures absolute safety at step 0.
- However, as $b \to -\infty$, the gradient norm $\left\| \frac{\partial \mathcal{L}}{\partial W_K} \right\| \to 0$ exponentially fast. The optimizer is **completely paralyzed**, and the gate will never learn to open.

### The Optimal Initialization Window
To balance PPL safety with gradient flow, we establish the optimal initialization window for the bias $b$:
$$b \in [-4.0, -2.5]$$
At $b = -3.0$:
- $\alpha_0 = \sigma(-3.0) \approx 0.0474$ (very small initial residual contribution, ensuring PPL perturbation is bounded by $0.0474 \cdot s \cdot \|W_V m_t\|_2 \approx 10^{-5}$, which is numerically negligible).
- $\alpha_0(1 - \alpha_0) \approx 0.0451$, which provides a small but non-vanishing gradient, allowing the Adam optimizer (which normalizes step sizes via running second moments) to smoothly wake up and optimize the gate.

---

## 3. Local First-Order Optimality Feasibility Condition

We correct the previous unrigorous assertion that $\mathcal{L} \leq \mathcal{L}_{\text{Baseline}}$ holds globally at all times during training. 

**Theorem (First-Order Local Feasibility)**:
Let the joint optimization trajectory be parameterized by $\Theta = \{\theta_{\text{base}}, \theta_{\text{gate}}\}$ under gradient descent with step size $\eta$. Let $\mathcal{L}(\Theta)$ be the loss.
At step $t = 0$, with gate bias $b \in [-4.0, -2.5]$ and $s = 10^{-4}$:

1. **Loss Preservation**:
   $$\mathcal{L}(\Theta_0) = \mathcal{L}_{\text{Baseline}}(\theta_{\text{base}, 0}) + \mathcal{O}(s \cdot e^b)$$
2. **First-Step Descent Guarantee**:
   If there exists a projection direction in $W_V m_t$ that is aligned with the baseline loss gradient:
   $$\exists t \quad \text{s.t.} \quad \left( \frac{\partial \mathcal{L}}{\partial H_t^{(l)}}^T W_V m_t \right) < 0$$
   then a gradient step on $\theta_{\text{gate}}$ strictly reduces the joint loss compared to a gradient step on the baseline alone:
   $$\mathcal{L}(\Theta_0 - \eta \nabla_\Theta \mathcal{L}) < \mathcal{L}_{\text{Baseline}}(\theta_{\text{base}, 0} - \eta \nabla_{\theta_{\text{base}}} \mathcal{L}_{\text{Baseline}}) \quad \text{for } \eta \to 0^+$$

**Proof**:
By Taylor expansion around $\Theta_0$:
$$\mathcal{L}(\Theta_0 - \eta g) = \mathcal{L}(\Theta_0) - \eta \|g\|_2^2 + o(\eta)$$
Since $\frac{\partial \mathcal{L}}{\partial W_V}$ and $\frac{\partial \mathcal{L}}{\partial W_K}$ are non-zero and point in the direction of steepest descent, the introduction of the extra parameters $\theta_{\text{gate}}$ expands the optimization degrees of freedom. Since the starting point is virtually identical to the baseline ($\mathcal{O}(s \cdot e^b)$), and the gradient step on the extra parameters has a negative inner product with the loss, the expanded parameter space ensures a strictly greater decrease in loss locally for an infinitesimally small step $\eta \to 0^+$.

At convergence ($t \to \infty$), due to the non-convexity of the joint loss surface, the trajectory is not guaranteed to find a lower global minimum than the baseline. However, the local feasibility guarantees that at initialization, the introduction of the gate is **strictly non-harmful and possesses a positive gradient descent trajectory**.

---

## Implementation Verification Invariant

To satisfy this proof, the implementation in `src/layers/engram.py` must enforce:
1. $k_t = W_K e_t$ and $v_t = W_V e_t$ directly from raw $e_t$ to preserve raw parameters.
2. $\text{RMSNorm}$ applied to Key $k_t$ inside the Sigmoid gating function to enforce the standard deviation bound of 1.
3. The initialization of `gate_bias` $b$ must be within $[-4.0, -2.5]$ (default $-3.0$) to avoid the gradient vanishing paradox.
