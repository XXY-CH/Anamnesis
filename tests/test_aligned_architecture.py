import torch

from src.layers.attention_residual import BlockAttentionResidual
from src.layers.engram import HashedNgramEngram
from src.layers.milestone_gate import MilestoneRetentionGate
from src.layers.milestone_snapshot import MilestoneSnapshotReadout
from src.models import AnamnesisConfig, AnamnesisModel
from src.training import make_toy_lm_batch, train_step


def test_hashed_engram_is_deterministic_and_scaled() -> None:
    module = HashedNgramEngram(
        vocab_size=32,
        d_model=16,
        num_slots=64,
        max_ngram=2,
        num_hash_heads=2,
        init_scale=1e-4,
    )
    hidden = torch.randn(2, 5, 16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 7, 8, 9]])

    residual_a, gate_a = module(hidden, input_ids)
    residual_b, gate_b = module(hidden, input_ids)

    assert residual_a.shape == hidden.shape
    assert gate_a.shape == (hidden.shape[0], hidden.shape[1], 1)
    assert torch.allclose(residual_a, residual_b)
    assert torch.allclose(gate_a, gate_b)
    assert module.residual_scale.abs().item() <= 1e-4


def test_hashed_engram_tables_can_stay_on_cpu_after_model_move() -> None:
    module = HashedNgramEngram(
        vocab_size=32,
        d_model=16,
        num_slots=64,
        max_ngram=2,
        num_hash_heads=2,
        table_device="cpu",
    )

    module.to(torch.device("cpu"))
    hidden = torch.randn(2, 5, 16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 7, 8, 9]])
    residual, gate = module(hidden, input_ids)

    assert module.table_devices() == {torch.device("cpu")}
    assert module.table_memory_bytes() == 2 * 2 * 64 * 16 * 4
    assert residual.device == hidden.device
    assert gate.device == hidden.device


def test_block_attnres_reads_depth_sources_with_tiny_scale() -> None:
    module = BlockAttentionResidual(d_model=16, max_sources=2, init_scale=1e-4)
    x = torch.randn(2, 5, 16)
    sources = [torch.randn(2, 5, 16) for _ in range(3)]

    residual, weights = module(x, sources)

    assert residual.shape == x.shape
    assert weights is not None
    assert weights.shape == (2, 5, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert module.residual_scale.abs().item() <= 1e-4


def test_milestone_gate_protects_configured_window() -> None:
    gate = MilestoneRetentionGate(
        n_heads=2,
        milestone_token_ids=(99,),
        protected_gamma=0.999,
        ttl=3,
    )
    input_ids = torch.tensor([[1, 99, 2, 3, 4]])
    base_gamma = torch.tensor([0.9, 0.8])

    values = gate(input_ids, base_gamma)

    assert values is not None
    assert values.shape == (1, 5, 2)
    assert torch.all(values[0, 1:4] >= 0.999)
    assert torch.allclose(values[0, 0], base_gamma)


def test_milestone_snapshot_collects_and_reads_marked_tokens() -> None:
    module = MilestoneSnapshotReadout(d_model=16, max_snapshots=2, init_scale=1e-4)
    hidden = torch.randn(2, 5, 16)
    mask = torch.tensor([[False, True, False, True, False], [False, False, False, False, True]])

    cache = module.collect(hidden, mask)
    residual, weights = module(hidden, cache)

    assert cache is not None
    snapshots, valid = cache
    assert snapshots.shape == (2, 2, 16)
    assert valid.tolist() == [[True, True], [True, False]]
    assert residual.shape == hidden.shape
    assert weights is not None
    assert weights.shape == (2, 5, 2)


def test_anamnesis_model_forward_and_backward() -> None:
    config = AnamnesisConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_layers=4,
        d_ff=32,
        max_seq_len=16,
        engram_layers=(1,),
        engram_num_slots=64,
        engram_max_ngram=2,
        engram_hash_heads=2,
        attnres_every=2,
        attnres_max_sources=2,
        milestone_token_ids=(63,),
        milestone_ttl=4,
        use_milestone_snapshots=True,
    )
    model = AnamnesisModel(config)
    input_ids = torch.randint(0, 63, (2, 8))
    input_ids[:, 2] = 63

    logits, metrics = model(input_ids, return_metrics=True)
    loss = logits.mean()
    loss.backward()

    assert logits.shape == (2, 8, 64)
    assert "milestone_gate_mean" in metrics
    assert any("engram_gate_mean" in key for key in metrics)
    assert any("attnres_entropy" in key for key in metrics)
    assert "snapshot_entropy" in metrics
    assert model.token_embedding.weight.grad is not None


def test_token_copy_buffer_reports_alignment_diagnostics() -> None:
    config = AnamnesisConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_layers=4,
        d_ff=32,
        max_seq_len=16,
        engram_layers=(),
        milestone_token_ids=(63,),
        use_token_copy_buffer=True,
        max_milestone_snapshots=4,
    )
    model = AnamnesisModel(config)
    input_ids = torch.randint(0, 63, (2, 8))
    input_ids[:, 4] = 63

    logits, metrics, diagnostics = model(
        input_ids,
        return_metrics=True,
        return_diagnostics=True,
    )

    assert logits.shape == (2, 8, 64)
    assert "token_copy_entropy" in metrics
    assert "token_copy_valid_count" in metrics
    assert diagnostics["token_copy_weights"].shape == (2, 8, 4)
    assert diagnostics["token_copy_valid"].shape == (2, 4)
    assert diagnostics["token_copy_pos_ids"].shape == (2, 4)
    assert torch.all(diagnostics["token_copy_pos_ids"] <= 3)


def test_synthetic_train_step_updates_model_and_reports_metrics() -> None:
    config = AnamnesisConfig(
        vocab_size=48,
        d_model=16,
        n_heads=4,
        n_layers=4,
        d_ff=32,
        max_seq_len=16,
        engram_layers=(1,),
        engram_num_slots=64,
        engram_max_ngram=2,
        engram_hash_heads=2,
        attnres_every=2,
        milestone_token_ids=(47,),
    )
    model = AnamnesisModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_toy_lm_batch(
        batch_size=2,
        seq_len=8,
        vocab_size=48,
        milestone_token_id=47,
    )

    before = model.token_embedding.weight.detach().clone()
    loss, metrics = train_step(model, optimizer, batch)

    assert torch.isfinite(loss)
    assert metrics["loss"] > 0
    assert "milestone_gate_mean" in metrics
    assert not torch.allclose(before, model.token_embedding.weight)
