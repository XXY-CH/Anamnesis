# Proof 33: Engram Mechanisms Are Capacity-Wasting for Statistical LM

## Proposition

For standard next-token language modeling, adding Engram (hashed N-gram associative
memory) as a residual branch within the RetNet backbone increases parameter count
without reducing loss. The mechanism is *capacity-wasting*: it adds a computation
path whose information content is already captured by the retention state.

## Empirical Evidence

| Config | Params | val_ppl (TinyStories, 5K steps) |
|--------|--------|--------------------------------|
| Bare RetNet | 1.67M | **2.99** |
| + Engram (6.3M hash tables) | 8.08M | 3.14 |

Bare RetNet achieves lower perplexity with 5× fewer parameters.

## Definitions

**RetNet state** at position $t$: $S_t = \gamma S_{t-1} + k_t \otimes v_t$

This is a running weighted average of all past key-value pairs. For any N-gram
$(x_{t-n+1}, \ldots, x_t)$ that appears in the context, the retention state
encodes the relationship between $x_t$ and all preceding tokens, including
the N-gram context.

**Engram lookup**: For N-gram $(x_{t-n+1}, \ldots, x_t)$, compute
$h = \text{hash}(x_{t-n+1}, \ldots, x_t) \mod N_{\text{slots}}$ and retrieve
$E[h]$, a learned embedding. Apply gated residual:

$$r_{\text{engram}} = \sigma(W_g [h_t; E[h]]) \cdot W_v E[h]$$

where $h_t$ is the current hidden state and $\sigma$ is sigmoid.

## Argument

### 1. Information Redundancy

The Engram provides information of the form: "given the N-gram $(x_{t-n+1}, \ldots, x_t)$,
here is an associated embedding." The RetNet state $S_t$ already encodes
the same information because:

$$q_t^T S_{t-1} = \sum_{s<t} q_t^T k_s v_s \prod_{r=s+1}^{t} \gamma_r$$

For the most recent $n$ tokens (where decay $\gamma^s \approx 1$), this sum
directly computes the association between the current query and recent N-gram
contexts. The RetNet *already performs* a soft, attention-weighted version of
what the Engram does via hard hash lookup.

### 2. Hash Collision Noise

The Engram uses $N_{\text{slots}}$ hash buckets. For N-gram order $n$ with
vocabulary size $V$, the number of possible N-grams is $V^n$. The collision
probability for uniform hashing is:

$$P[\text{collision}] = 1 - \left(1 - \frac{1}{N_{\text{slots}}}\right)^{V^n - 1}$$

With $V=94$, $n=3$, $N_{\text{slots}}=4096$: $V^n = 830{,}584$ tokens map to
4096 slots → ~203 N-grams per slot on average. The retrieved embedding is a
superposition of all N-grams mapping to that slot, introducing noise.

The RetNet avoids this: its "soft lookup" (attention) gives each token a unique
weight, no collisions.

### 3. Gate Initialization Near Zero

The Engram gate bias is initialized to $-3.0$, meaning $\sigma(-3) \approx 0.047$.
The mechanism starts nearly disabled. For it to provide benefit, the model must:

1. Learn that Engram retrieval is useful for SOME positions
2. Open the gate at those positions
3. Learn useful embeddings in the hash tables

This requires positive gradient signal to flow through the gate. But since the
RetNet already captures the same information (Argument 1), the gradient signal
at the Engram gate is small and noisy. The model wastes capacity oscillating
between using and suppressing the Engram.

### 4. Formal Capacity Waste

Let $L(\theta)$ be the LM loss with parameters $\theta$. Add Engram parameters
$\phi$ with gated residual $r_\phi$:

$$L(\theta, \phi) = -\log P(y_t | h_t + r_\phi(h_t, x_t))$$

The optimal $\phi^*$ satisfies $\nabla_\phi L = 0$. At initialization,
$r_\phi \approx 0$ (gate near zero), so the gradient is:

$$\nabla_\phi L \approx -\frac{\partial \log P}{\partial h} \cdot \frac{\partial r_\phi}{\partial \phi}$$

Since $\frac{\partial r_\phi}{\partial \phi} \approx 0$ (gate near zero),
the gradient magnitude is $O(\sigma(-3)) \approx 0.05$. The effective learning
rate for Engram is $5\%$ of the nominal rate.

Meanwhile, the RetNet parameters receive full gradient signal. The Engram
parameters learn ~20× slower than RetNet parameters, wasting optimizer capacity.

### 5. Implication for Architecture Design

The Engram mechanism is useful when:
- The task requires **exact recall** of specific associations (needle-in-haystack)
- The information must survive the retention decay chain
- TCB provides a bypass around the decay chain

The Engram mechanism is **harmful** when:
- The task is **statistical prediction** (standard LM)
- Every position contributes equally to loss
- The RetNet already captures the relevant dependencies

**Conclusion**: Memory mechanisms should be external modules activated on demand,
not integrated into the LM backbone. This validates the pipeline design:
Small Reasoner (clean RetNet) + Memory Compiler (separate module).

## Corollary: Information Novelty Criterion

For any gated residual mechanism $g(x) \cdot f(x)$ added to a RetNet layer:
- If $f(x)$ captures information already in the retention state → capacity waste
- If $f(x)$ provides information NOT in the retention state → potential benefit

This gives a clear criterion for evaluating new mechanisms: **does it provide
information that the retention state cannot capture?**

TCB passes this test: raw token embeddings are NOT in the retention state
(which stores compressed $k \otimes v$ products). TCB provides exact-copy paths
that RetNet cannot achieve.

Engram fails this test: N-gram associations ARE captured by the RetNet's
attention-weighted sum over recent positions.
