# Proof 48: Engram is Fundamentally Ineffective on BPE Tokenization

## Statement

For BPE tokenization with vocabulary size V, the Engram hash table's signal-to-noise ratio (SNR)
decreases as O(1/sqrt(V^n)) regardless of the number of hash slots K. Increasing K cannot
recover effectiveness when V exceeds a threshold determined by training data size.

## Hash Collision Analysis

Under uniform hashing, expected n-grams per slot = D/K. SNR for a slot lookup (from Proof 47):

SNR proportional to sqrt(K * S_i / M)

- Char-level (V=94, n=3): M = 830K, D = 50K, K=8192. Collision rate ~6:1. SNR adequate.
- BPE (V=4096, n=3): M = 69B. Even with K=65536, SNR proportional to sqrt(S_i / 1.05M).
  For S_i < 1000, SNR < 0.03 — essentially noise.

## Why More Slots Don't Help

1. **No positional N-gram structure**: BPE merges are context-dependent.
2. **Exponentially larger space**: 83,000x larger than char-level, same data size.
3. **Semantic vs surface N-grams**: Hash lookup cannot capture BPE ambiguity.

## Empirical Validation

| Config | Vocab | Slots | val_ppl | Effect |
|--------|-------|-------|---------|--------|
| Char WikiText-2 | ~100 | 8K | 4.82 | -18% vs TF |
| Char WikiText-2 | ~100 | 64K | 4.34 | -9.8% vs 8K (helps) |
| BPE WikiText-2 | 4096 | 8K | 183.84 | +50% vs TF (hurts) |
| BPE WikiText-2 | 4096 | 64K | 234.26 | +27% vs 8K (worse!) |

More slots help on char-level (-9.8%) but hurt on BPE (+27.4%). The 50M extra params
learn noisy mappings that degrade the base model.

## Engram Applicability Criterion

Enable when D/K < alpha (alpha ~10).

- Shakespeare char: 50K/8192 = 6.1 < 10 (OK)
- BPE: D/65536 > 10 (FAIL)

## Conclusion

Engram is a surface-level pattern matching mechanism. Works for atomic symbols (characters),
fails for semantic abstractions (BPE subwords). Not fixable by increasing K.

Design: enable Engram only for small-vocab (V < 200). For BPE, use bare RetNet 8h+layerwise.
