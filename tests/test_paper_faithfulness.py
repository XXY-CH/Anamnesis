import torch
import torch.nn as nn
from src.layers.engram import HashedNgramEngram
from src.layers.attention_residual import BlockAttentionResidual


def test_engram_paper_faithfulness() -> None:
    """Verify that HashedNgramEngram conforms exactly to DeepSeek paper equations."""
    d_model = 16
    vocab_size = 32
    num_slots = 64
    max_ngram = 2
    num_hash_heads = 2

    module = HashedNgramEngram(
        vocab_size=vocab_size,
        d_model=d_model,
        num_slots=num_slots,
        max_ngram=max_ngram,
        num_hash_heads=num_hash_heads,
        init_scale=1e-4,
        gate_bias=-3.0,
    )

    # 1. Structural Checks
    assert hasattr(module, "key_norm"), "Engram must have self.key_norm"
    assert isinstance(module.key_norm, nn.RMSNorm), "self.key_norm must be RMSNorm"
    assert hasattr(module, "conv_norm"), "Engram must have self.conv_norm"
    assert isinstance(module.conv_norm, nn.RMSNorm), "self.conv_norm must be RMSNorm"
    assert hasattr(module, "conv"), "Engram must have self.conv"
    assert isinstance(module.conv, nn.Conv1d), "self.conv must be Conv1d"

    # Verify Conv1d configuration (Equation 5)
    assert module.conv.kernel_size == (4,), "Conv1d kernel_size must be 4"
    assert module.conv.dilation == (3,), "Conv1d dilation must be 3"
    assert module.conv.groups == d_model, "Conv1d must be depthwise (groups = d_model)"
    assert module.conv.padding == (0,), "Conv1d padding must be 0 (manual padding handled causally)"

    # 2. Forward Pass and Gate Shape Checks (Equation 4)
    hidden = torch.randn(2, 5, d_model)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 7, 8, 9]])
    
    residual, gate = module(hidden, input_ids)
    
    assert residual.shape == (2, 5, d_model), "Residual shape must match hidden shape"
    assert gate.shape == (2, 5, 1), "Gate shape must be [batch, seq_len, 1] (scalar gating)"

    # Verify that the gate is contextual and bounded in [0, 1]
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    assert not torch.allclose(gate[:, 0], gate[:, 1]), "Gating must be contextualized and token-dependent"


def test_attention_residual_paper_faithfulness() -> None:
    """Verify that BlockAttentionResidual conforms exactly to Kimi Team paper equations."""
    d_model = 16
    max_sources = 4

    module = BlockAttentionResidual(
        d_model=d_model,
        max_sources=max_sources,
        init_scale=1e-4,
    )

    # 1. Structural Checks (No linear value and output projections in Kimi Team paper)
    assert not hasattr(module, "value_proj"), "AttnRes must not have value_proj"
    assert not hasattr(module, "out_proj"), "AttnRes must not have out_proj"
    assert hasattr(module, "key_norm"), "AttnRes must have key_norm"
    assert isinstance(module.key_norm, nn.RMSNorm)

    # 2. Forward Pass and Weight Selection Checks (Equation 4 of Kimi paper)
    hidden = torch.randn(2, 5, d_model)
    sources = [torch.randn(2, 5, d_model) for _ in range(3)]
    
    residual, weights = module(hidden, sources)
    
    assert residual.shape == (2, 5, d_model), "Residual shape must match hidden shape"
    assert weights.shape == (2, 5, 3), "Weights shape must be [batch, seq_len, num_sources]"
    
    # Softmax check (weights along source dimension must sum to 1.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5))
