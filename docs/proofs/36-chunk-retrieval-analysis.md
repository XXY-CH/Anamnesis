# Proof 36: Chunk Retrieval — Contrastive Selection + Token Embedding Readout

## Problem Statement

A small model trained on sequences of length T cannot directly process sequences
of length L >> T. The retention state decays exponentially, losing information
about early positions. We need an external memory mechanism that:

1. Selects which parts of the long context are relevant (chunk selection)
2. Extracts specific information from selected parts (readout)
3. Integrates with the model's generation without breaking existing predictions

## Analysis

### Why Hidden-State Readout Fails

For a hidden state h_t ∈ R^d at position t, the logit projection is:

$$l_{vocab} = h_t \cdot E^T$$

where E ∈ R^{V×d} is the token embedding matrix. This gives high logits for
tokens whose embeddings align with h_t. But h_t is the model's *internal
representation* after processing positions 0..t-1, not a representation of
token t itself. For a causal LM:

- h_t encodes "what should come next" (for next-token prediction)
- h_t does NOT encode "what token is at position t"

**Empirical result**: cross-attention from query to chunk hidden states,
projected through E, gives EM=0.000 at all lengths. The attention peaks at
random positions, and the readout never recovers password token identity.

### Why Token Embedding Readout Works

For a token embedding e_v = E[v] where v is the token ID:

$$l_{vocab} = e_v \cdot E^T = E[v] \cdot E^T$$

This is a dot product of token v's embedding with every row of E. By construction,
e_v · e_v = ||e_v||² > e_v · e_w for most w ≠ v (embeddings are learned to be
distributed). This is essentially a self-similarity lookup — it reliably produces
the highest logit for token v itself.

**Empirical result**: token embedding readout gives EM=1.000 at all tested lengths
(512 to 32768), because the readout directly encodes which token was stored.

### Why Contrastive Selection Works (and Generation Loss Doesn't)

The chunk selection problem: given N chunks and a query, find which chunk contains
the relevant information.

**Generation loss** trains selection by backpropagating through the answer loss.
This requires: (a) the base model to already generate somewhat correct answers,
and (b) the logit correction to be non-trivially trainable. Both fail when the
base model can't solve the task.

**Contrastive loss** trains selection directly:

$$L = -\log \frac{\exp(q \cdot c_{needle} / \sqrt{d})}{\sum_i \exp(q \cdot c_i / \sqrt{d})}$$

This only requires the model's hidden states to distinguish the needle chunk from
filler chunks. Even when the model has EM=0.000, chunk discrimination is 0.469
(vs 0.250 random), providing a learnable signal.

### Separation of Concerns

The pipeline has two independent mechanisms:

1. **Chunk selection** (learned): query_proj(q) · chunk_proj(c) → top chunk
2. **Token readout** (geometric): E[v] · E^T → logit for token v

The selection mechanism is a learned neural network. The readout is a mathematical
property of the embedding space. They can be trained and analyzed independently.

### Scaling Analysis

The retriever was trained on 4 chunks (seq_len=2048, chunk_size=512) and
generalizes to 64 chunks (seq_len=32768). Why?

The softmax normalization in contrastive training:

$$p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$

With more chunks, the denominator grows, but the numerator for the correct chunk
remains dominant (s_{needle} >> s_{filler}). The model learns a GENERAL scoring
function, not a position-specific one.

**Empirical**: chunk accuracy at 32K (64 chunks) = 7/8 to 8/8, despite training
on only 4 chunks. The scoring function generalizes because it measures semantic
similarity, not position.

## Key Results

| Mechanism | EM@2048 | EM@32K | Why |
|-----------|---------|--------|-----|
| Hidden-state readout | 0.000 | 0.000 | Hidden states ≠ token identity |
| Token embedding readout | 1.000 | 1.000 | Self-similarity in embedding space |
| Generation loss selection | 0.000 | — | Base model EM=0 → no gradient signal |
| Contrastive selection | 1.000 | 1.000 | Works with 0.469 discrimination signal |

## Limitations and Next Steps

1. **MARK_THOUGHT dependency**: The current pipeline uses MARK_THOUGHT to identify
   the password within the selected chunk. Real tasks don't have explicit markers.
   Next: learn within-chunk position identification.

2. **Single needle**: Only one piece of information is retrieved. Real tasks need
   multi-needle retrieval from different chunks. Next: multi-needle experiment.

3. **Oracle token injection**: The readout directly reads token IDs from the input.
   For real tasks, the model needs to learn what to extract. Next: learned cross-attention
   with token embeddings as values.

4. **Chunk boundary sensitivity**: Passwords near chunk boundaries may be split
   across chunks, breaking retrieval. Mitigation: overlapping chunks.
