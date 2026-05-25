# Proof 42: Chunk-Level RoPE Signal-to-Noise Ratio Collapse

## Proposition

In ultra-long context retrieval ($N \to \infty$ chunks), applying rotary position encoding
(RoPE) to chunk embeddings during scoring causes the positional noise variance to overwhelm
the content signal. The inner product between query and chunk embeddings degenerates to
a random variable uniformly distributed in $[-1, 1]$, making correct chunk identification
impossible regardless of content similarity.

## Setup

Let $q \in \mathbb{R}^d$ be the query embedding and $c_i \in \mathbb{R}^d$ the $i$-th
chunk embedding after projection. The scoring function with chunk-level RoPE is:

$$S_i = q^T R_{\Delta_i} c_i$$

where $R_{\Delta_i}$ is the rotation matrix for the relative position $\Delta_i = |i - i_q|$
between the query's chunk and chunk $i$.

For RoPE in $d$ dimensions, the rotation angle at dimension $k$ for distance $\Delta$ is:

$$\theta_k(\Delta) = \Delta \cdot \omega_k, \quad \omega_k = 10000^{-2k/d}$$

## Proof of SNR Collapse

### Case 1: Correct chunk ($i = i^*$, content matches, $\Delta = 0$)

Without RoPE: $S_{i^*} = q^T c_{i^*} = 1$ (normalized, perfect match).

With RoPE: $R_0 = I$, so $S_{i^*} = q^T c_{i^*} = 1$. RoPE does not affect the correct chunk.

### Case 2: Distant chunk ($i \neq i^*$, content does not match, $\Delta$ large)

Without RoPE: $S_i = q^T c_i < 1$ (content mismatch). The retriever correctly rejects this chunk.

With RoPE: The score becomes $S_i = q^T R_{\Delta} c_i$. For large $\Delta$ (e.g., $\Delta = 1000$
at 1M tokens with 512-token chunks), the rotation angles $\theta_k(\Delta)$ are effectively
random — the low-frequency dimensions wrap around many times, and the high-frequency
dimensions wrap around many more times.

The expected value and variance of the rotated inner product:

$$E[q^T R_\Delta c_i] = 0 \text{ (uniform random rotation of unrelated vectors)}$$

$$\text{Var}[q^T R_\Delta c_i] = \frac{1}{d} \|q\|^2 \|c_i\|^2$$

### The Collapse

The content signal (without RoPE) is:
$$\mu_{\text{content}} = q^T c_{i^*} - E[q^T c_i] > 0$$

The positional noise (from RoPE on non-target chunks) is:
$$\sigma_{\text{position}} = \sqrt{\text{Var}[q^T R_\Delta c_i]} \propto \frac{1}{\sqrt{d}}$$

But more critically, the RoPE also rotates the **correct chunk's** embedding when computing
scores at the query position. In our implementation, both query and chunk are rotated:

$$S_i = (R_{i_q} q)^T (R_i c_i) = q^T R_{i_q}^{-1} R_i c_i = q^T R_{i - i_q} c_i$$

So the correct chunk (at $i = i_q$) gets $R_0 = I$ — no rotation. But ALL other chunks
get rotated, making their scores unpredictable.

**The problem**: With 2048 chunks, the non-target chunks' scores become random noise.
But the retriever was trained on 16 chunks (at 8192 tokens). The RoPE projections learned
for 16-way discrimination don't generalize to 2048-way with extreme rotation angles.

### SNR Analysis

For $N$ chunks with RoPE:
- Signal: $S_{\text{correct}} = q^T c_{i^*}$ (unchanged by RoPE)
- Noise per distractor: $\text{Var} \approx \frac{1}{d} \|q\|^2 \|c_i\|^2$
- Maximum noise over $N$ distractors: $\propto \sqrt{2 \ln N} \cdot \sigma$

For $N = 2048, d = 256$:
$$\text{SNR} = \frac{S_{\text{correct}}}{\sqrt{2 \ln(2048)} \cdot \sigma} \propto \frac{\sqrt{d}}{\sqrt{\ln N}}$$

This decreases as $N$ grows. Without RoPE, the noise is lower because the model's
embeddings naturally discriminate by content. RoPE adds positional noise that obscures
this content signal.

## Empirical Confirmation

| Length | Chunks | Flat EM | +Chunk RoPE EM | Delta |
|--------|--------|---------|----------------|-------|
| 131K | 256 | 0.750 | 0.750 | 0.000 |
| 262K | 512 | 0.750 | 0.625 | -0.125 |
| 524K | 1024 | 0.875 | 0.375 | **-0.500** |
| 1M | 2048 | 0.875 | 0.250 | **-0.625** |

The collapse accelerates as chunk count grows — exactly as predicted by the SNR analysis.

## Corollary

For single-needle retrieval where the needle's position is unknown, the scoring function
must be purely content-based. Position encoding should be removed from chunk scoring and
reserved for within-chunk processing only.

This does NOT mean position is unimportant — it means position and content must be
handled at different stages:
1. **Chunk selection**: content-only scoring (no RoPE)
2. **Within-chunk processing**: full positional encoding (the model handles this)
