import argparse
import csv
from pathlib import Path

import torch

from experiments.train_synthetic import build_model, build_optimizer, make_batch, run_variant
from src.layers.milestone_snapshot import MilestoneSnapshotReadout
from src.models import AnamnesisModel


def _small_args(**overrides):
    defaults = dict(
        task="needle",
        variants="ours_snapshot_logits",
        steps=1,
        batch_size=2,
        seq_len=16,
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        use_branch_optimizer=False,
        per_group_grad_clip=False,
        guard_lr_mult=5.0,
        cache_lr_mult=3.0,
        guard_beta1=0.0,
        cache_beta1=0.5,
        seed=7,
        device="cpu",
        log_interval=1,
        eval_batches=1,
        eval_seed=7007,
        eval_split="train",
        eval_drop_modules=[],
        out_dir="",
        engram_layer=1,
        engram_slots=64,
        engram_max_ngram=2,
        engram_hash_heads=2,
        attnres_every=2,
        attnres_max_sources=2,
        attnres_distance_penalty=0.0,
        branch_init_scale=1e-4,
        use_milestones=True,
        milestone_ttl=8,
        milestone_gamma=0.999,
        use_snapshots=True,
        max_snapshots=2,
        use_snapshot_logit_bias=True,
        snapshot_logit_scale=1.0,
        retention_gamma=0.95,
        needle_password_len=1,
        alien_num_pairs=2,
        alien_static_key_count=8,
        alien_static_value_count=8,
        alien_static_train_keys=6,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_snapshot_collection_respects_latest_k_budget() -> None:
    module = MilestoneSnapshotReadout(d_model=4, max_snapshots=2)
    hidden = torch.arange(1 * 5 * 4, dtype=torch.float32).view(1, 5, 4)
    mask = torch.tensor([[True, True, False, True, True]])

    cache = module.collect(hidden, mask)

    assert cache is not None
    snapshots, valid = cache
    assert valid.tolist() == [[True, True]]
    assert torch.equal(snapshots[0, 0], hidden[0, 3])
    assert torch.equal(snapshots[0, 1], hidden[0, 4])


def test_snapshot_collection_allows_zero_budget() -> None:
    module = MilestoneSnapshotReadout(d_model=4, max_snapshots=0)
    hidden = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, False, True]])

    assert module.collect(hidden, mask) is None


def test_snapshot_logit_bias_changes_logits_under_fixed_cache() -> None:
    args = _small_args()
    model = build_model(args, "ours_snapshot_logits", args.vocab_size)
    assert isinstance(model, AnamnesisModel)
    batch = make_batch(args, torch.device("cpu"))

    logits_with = model(batch.input_ids)
    logits_without = model(batch.input_ids, disable_snapshots=True)

    assert logits_with.shape == logits_without.shape
    assert not torch.allclose(logits_with, logits_without)


def test_synthetic_runner_writes_reproducible_smoke_metrics(tmp_path: Path) -> None:
    args = _small_args(out_dir=str(tmp_path), eval_drop_modules=["snapshot"])

    rows = run_variant(args, "ours_snapshot_logits", tmp_path)

    assert rows
    final = rows[-1]
    assert "eval_exact_match" in final
    assert "eval_no_snapshot_exact_match" in final
    csv_path = tmp_path / "needle_ours_snapshot_logits.csv"
    assert csv_path.exists()
    with csv_path.open() as handle:
        written = list(csv.DictReader(handle))
    assert written[-1]["variant"] == "ours_snapshot_logits"


def test_branch_optimizer_creates_base_guard_and_cache_groups() -> None:
    args = _small_args(use_branch_optimizer=True, per_group_grad_clip=True)
    model = build_model(args, "ours_snapshot_logits", args.vocab_size)

    optimizer = build_optimizer(args, model)

    names = {group["name"] for group in optimizer.param_groups}
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert names == {"base", "guard", "cache"}
    assert lrs["guard"] == args.learning_rate * args.guard_lr_mult
    assert lrs["cache"] == args.learning_rate * args.cache_lr_mult


def test_model_can_return_internal_heatmap_diagnostics() -> None:
    args = _small_args(seq_len=16)
    model = build_model(args, "ours_snapshot_logits", args.vocab_size)
    batch = make_batch(args, torch.device("cpu"))

    logits, metrics, diagnostics = model(
        batch.input_ids,
        return_metrics=True,
        return_diagnostics=True,
    )

    assert logits.shape[:2] == batch.input_ids.shape
    assert "snapshot_entropy" in metrics
    assert "snapshot_weights" in diagnostics
    assert "milestone_gate" in diagnostics
    assert any(key.endswith("_attnres_weights") for key in diagnostics)
    assert any(key.endswith("_engram_gate") for key in diagnostics)
