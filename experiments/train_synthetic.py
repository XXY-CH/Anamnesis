#!/usr/bin/env python
"""Flexible synthetic diagnostics for the aligned RetNet/Engram architecture.

Examples:
  python experiments/train_synthetic.py --task needle --variants ours,retnet,transformer
  python experiments/train_synthetic.py --task alien --variants ours,retnet --steps 500
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models import (
    AnamnesisConfig,
    AnamnesisModel,
    TransformerConfig,
    TransformerLM,
)  # noqa: E402


PAD = 0
START = 1
MARK_THOUGHT = 2
QUERY = 3
SEP = 4


@dataclass
class SyntheticBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_needle_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    filler_low: int = 16,
    password_len: int = 3,
) -> SyntheticBatch:
    """Needle-in-a-haystack copy task with explicit milestone marker."""
    total = seq_len + 1
    tokens = torch.randint(filler_low, vocab_size, (batch_size, total), device=device)
    # High-entropy answer tokens. The model must copy; it cannot solve the task
    # by learning an 11-token answer prior.
    passwords = torch.randint(5, vocab_size, (batch_size, password_len), device=device)
    tokens[:, 0] = START
    tokens[:, 1 : 1 + password_len] = passwords
    tokens[:, 1 + password_len] = MARK_THOUGHT
    tokens[:, -password_len - 1] = QUERY
    tokens[:, -password_len:] = passwords

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, -password_len:] = True
    return SyntheticBatch(tokens[:, :-1], tokens[:, 1:], mask)


def make_xor_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> SyntheticBatch:
    """Long binary recurrence task: x_t = x_{t-1} xor x_{t-2}."""
    del vocab_size
    total = seq_len + 1
    bits = torch.zeros(batch_size, total, device=device, dtype=torch.long)
    bits[:, 0:2] = torch.randint(0, 2, (batch_size, 2), device=device) + 5
    for idx in range(2, total):
        prev = (bits[:, idx - 1] - 5).clamp(0, 1)
        prev2 = (bits[:, idx - 2] - 5).clamp(0, 1)
        bits[:, idx] = (prev ^ prev2) + 5
    bits[:, 0] = START
    bits[:, seq_len // 2] = MARK_THOUGHT

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, 2:] = True
    return SyntheticBatch(bits[:, :-1], bits[:, 1:], mask)


def make_xor_final_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> SyntheticBatch:
    """Binary recurrence task with a single final answer token."""
    del vocab_size
    total = seq_len + 1
    bits = torch.zeros(batch_size, total, device=device, dtype=torch.long)
    bits[:, 0:2] = torch.randint(0, 2, (batch_size, 2), device=device) + 5
    for idx in range(2, total):
        prev = (bits[:, idx - 1] - 5).clamp(0, 1)
        prev2 = (bits[:, idx - 2] - 5).clamp(0, 1)
        bits[:, idx] = (prev ^ prev2) + 5

    tokens = torch.randint(16, 96, (batch_size, total), device=device)
    prefix_len = max(4, seq_len - 2)
    tokens[:, 0] = START
    tokens[:, 1 : prefix_len - 1] = bits[:, 1 : prefix_len - 1]
    tokens[:, prefix_len // 2] = MARK_THOUGHT
    tokens[:, -2] = QUERY
    tokens[:, -1] = bits[:, prefix_len - 2]

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, -1] = True
    return SyntheticBatch(tokens[:, :-1], tokens[:, 1:], mask)


def make_alien_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_pairs: int = 4,
) -> SyntheticBatch:
    """Synthetic key-value dictionary task.

    Keys and values are stable token ranges; the answer position repeats the
    value paired with a queried key. This stresses static lexical lookup.
    """
    del vocab_size
    key_base = 16
    value_base = 64
    filler_low = 96
    filler_high = 160
    total = seq_len + 1
    tokens = torch.randint(filler_low, filler_high, (batch_size, total), device=device)
    tokens[:, 0] = START

    keys = torch.randint(key_base, key_base + 32, (batch_size, num_pairs), device=device)
    values = torch.randint(value_base, value_base + 32, (batch_size, num_pairs), device=device)
    cursor = 1
    for pair_idx in range(num_pairs):
        tokens[:, cursor] = keys[:, pair_idx]
        tokens[:, cursor + 1] = SEP
        tokens[:, cursor + 2] = values[:, pair_idx]
        cursor += 3
    tokens[:, cursor] = MARK_THOUGHT

    choice = torch.randint(0, num_pairs, (batch_size,), device=device)
    query_key = keys.gather(1, choice.view(-1, 1)).squeeze(1)
    answer = values.gather(1, choice.view(-1, 1)).squeeze(1)
    tokens[:, -3] = QUERY
    tokens[:, -2] = query_key
    tokens[:, -1] = answer

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, -1] = True
    return SyntheticBatch(tokens[:, :-1], tokens[:, 1:], mask)


def make_alien_static_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    key_count: int = 32,
    value_count: int = 32,
    train_keys: int = 24,
    split: str = "train",
) -> SyntheticBatch:
    """Static key-value task without in-context dictionary pairs.

    The mapping is deterministic but nontrivial across batches:
    key -> value_hash(key). A train/test key split prevents us from mistaking a
    tiny arithmetic mapping for an Engram-style factual lookup.
    """
    total = seq_len + 1
    key_base = 16
    value_base = key_base + key_count
    filler_low = value_base + value_count
    filler_high = max(filler_low + 1, vocab_size)

    if split == "train":
        low, high = 0, train_keys
    elif split == "test":
        low, high = train_keys, key_count
    elif split == "all":
        low, high = 0, key_count
    else:
        raise ValueError(f"Unknown alien_static split: {split}")
    if high <= low:
        raise ValueError(f"Empty alien_static key split {split}: [{low}, {high})")

    tokens = torch.randint(filler_low, filler_high, (batch_size, total), device=device)
    key_offsets = torch.randint(low, high, (batch_size,), device=device)
    keys = key_base + key_offsets
    value_offsets = (key_offsets * 1103515245 + 12345) % value_count
    values = value_base + value_offsets
    tokens[:, 0] = START
    tokens[:, 1] = MARK_THOUGHT
    tokens[:, -3] = QUERY
    tokens[:, -2] = keys
    tokens[:, -1] = values

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, -1] = True
    return SyntheticBatch(tokens[:, :-1], tokens[:, 1:], mask)


TASKS = {"needle", "xor", "xor_final", "alien", "alien_static"}


def make_batch(
    args: argparse.Namespace, device: torch.device, split: str = "train"
) -> SyntheticBatch:
    if args.task == "needle":
        return make_needle_batch(
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            password_len=args.needle_password_len,
        )
    if args.task == "xor":
        return make_xor_batch(args.batch_size, args.seq_len, args.vocab_size, device)
    if args.task == "xor_final":
        return make_xor_final_batch(args.batch_size, args.seq_len, args.vocab_size, device)
    if args.task == "alien":
        return make_alien_batch(
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            num_pairs=args.alien_num_pairs,
        )
    if args.task == "alien_static":
        return make_alien_static_batch(
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            key_count=args.alien_static_key_count,
            value_count=args.alien_static_value_count,
            train_keys=args.alien_static_train_keys,
            split=split,
        )
    raise ValueError(f"Unknown task: {args.task}")


def masked_lm_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    return (per_token * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    correct = ((pred == targets) & mask).sum()
    total = mask.sum().clamp_min(1)
    return float((correct / total).detach().cpu())


def masked_exact_match(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    """Sequence-level exact match over masked answer positions."""
    pred = logits.argmax(dim=-1)
    per_token_ok = (pred == targets) | ~mask
    return float(per_token_ok.all(dim=1).float().mean().detach().cpu())


def override_retention_gamma(model: torch.nn.Module, gamma: float | None) -> None:
    """Force RetNet base decay for stress tests while preserving milestone override."""
    if gamma is None:
        return
    for module in model.modules():
        if hasattr(module, "gamma") and isinstance(module.gamma, torch.Tensor):
            module.gamma.fill_(gamma)


def build_model(args: argparse.Namespace, variant: str, vocab_size: int):
    common = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff or args.d_model * 4,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    if variant == "transformer":
        return TransformerLM(TransformerConfig(**common))

    uses_full_arch = variant.startswith("ours")
    uses_snapshots = (
        variant == "ours_snapshot"
        or variant == "ours_snapshot_logits"
        or (variant == "ours" and args.use_snapshots)
    )
    uses_snapshot_logit_bias = variant == "ours_snapshot_logits" or (
        variant == "ours" and args.use_snapshot_logit_bias
    )

    config = AnamnesisConfig(
        **common,
        engram_layers=(args.engram_layer,) if uses_full_arch else (),
        engram_num_slots=args.engram_slots,
        engram_max_ngram=args.engram_max_ngram,
        engram_hash_heads=args.engram_hash_heads,
        engram_table_device=getattr(args, "engram_table_device", None) if uses_full_arch else None,
        attnres_every=args.attnres_every if uses_full_arch else 0,
        attnres_max_sources=args.attnres_max_sources,
        attnres_distance_penalty=args.attnres_distance_penalty if uses_full_arch else 0.0,
        branch_init_scale=args.branch_init_scale,
        milestone_token_ids=(MARK_THOUGHT,) if uses_full_arch and args.use_milestones else (),
        milestone_ttl=args.milestone_ttl,
        milestone_gamma=args.milestone_gamma,
        use_milestone_snapshots=uses_full_arch and uses_snapshots,
        max_milestone_snapshots=args.max_snapshots,
        use_snapshot_logit_bias=uses_full_arch and uses_snapshot_logit_bias,
        snapshot_logit_scale=args.snapshot_logit_scale,
        use_token_copy_buffer=uses_full_arch and getattr(args, "use_token_copy_buffer", False),
        position_encoding_type=getattr(args, "position_encoding", "learned"),
        token_copy_sinusoidal_pos=getattr(args, "position_encoding", "learned") == "sinusoidal",
        input_dependent_gamma=bool(getattr(args, "input_dependent_gamma", False)),
        retention_output_gate=bool(getattr(args, "retention_output_gate", False)),
        input_dependent_write=bool(getattr(args, "input_dependent_write", False)),
        use_learned_gate=bool(getattr(args, "use_learned_gate", False)),
        use_engram_tcb_trigger=bool(getattr(args, "use_engram_tcb_trigger", False)),
        use_swiglu=bool(getattr(args, "use_swiglu", False)),
        learnable_gamma=bool(getattr(args, "learnable_gamma", False)),
        layerwise_gamma=bool(getattr(args, "layerwise_gamma", False)),
    )
    model = AnamnesisModel(config)
    override_retention_gamma(model, args.retention_gamma)
    return model


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build either a plain AdamW optimizer or branch-aware parameter groups."""
    if not args.use_branch_optimizer or not isinstance(model, AnamnesisModel):
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    base_params: list[torch.nn.Parameter] = []
    guard_params: list[torch.nn.Parameter] = []
    cache_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("snapshot_readout"):
            cache_params.append(param)
        elif ".engram." in name or ".attnres." in name:
            guard_params.append(param)
        else:
            base_params.append(param)

    groups: list[dict[str, object]] = []
    if base_params:
        groups.append(
            {
                "params": base_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": (0.9, 0.95),
                "name": "base",
            }
        )
    if guard_params:
        groups.append(
            {
                "params": guard_params,
                "lr": args.learning_rate * args.guard_lr_mult,
                "weight_decay": 0.0,
                "betas": (args.guard_beta1, 0.95),
                "name": "guard",
            }
        )
    if cache_params:
        groups.append(
            {
                "params": cache_params,
                "lr": args.learning_rate * args.cache_lr_mult,
                "weight_decay": 0.0,
                "betas": (args.cache_beta1, 0.95),
                "name": "cache",
            }
        )
    return torch.optim.AdamW(groups)


def clip_gradients(
    args: argparse.Namespace, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    if not args.grad_clip:
        return
    if args.use_branch_optimizer and args.per_group_grad_clip:
        for group in optimizer.param_groups:
            params = [param for param in group["params"] if param.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)


@torch.no_grad()
def evaluate_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    step: int,
    drop_modules: set[str] | None = None,
) -> dict[str, float]:
    """Evaluate on deterministic held-out batches without advancing train RNG."""
    if args.eval_batches <= 0:
        return {}

    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training

    try:
        set_seed(args.eval_seed + step)
        model.eval()
        losses: list[float] = []
        accuracies: list[float] = []
        exact_matches: list[float] = []
        for _ in range(args.eval_batches):
            batch = make_batch(args, device, split=args.eval_split)
            model_kwargs = {"return_metrics": True}
            if isinstance(model, AnamnesisModel):
                drop_modules = drop_modules or set()
                model_kwargs.update(
                    {
                        "disable_engram": "engram" in drop_modules,
                        "disable_attnres": "attnres" in drop_modules,
                        "disable_snapshots": "snapshot" in drop_modules,
                    }
                )
            logits, _ = model(batch.input_ids, **model_kwargs)
            loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
            losses.append(float(loss.detach().cpu()))
            accuracies.append(masked_accuracy(logits, batch.target_ids, batch.loss_mask))
            exact_matches.append(masked_exact_match(logits, batch.target_ids, batch.loss_mask))
        return {
            "eval_loss": sum(losses) / len(losses),
            "eval_accuracy": sum(accuracies) / len(accuracies),
            "eval_exact_match": sum(exact_matches) / len(exact_matches),
        }
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_model_recurrent(
    args: argparse.Namespace,
    model: AnamnesisModel,
    device: torch.device,
    step: int,
) -> dict[str, float]:
    """Evaluate using O(1) recurrent step-by-step inference.

    Processes each token individually through forward_recurrent_step,
    verifying that parallel-trained models work in recurrent mode.
    """
    if args.eval_batches <= 0:
        return {}

    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training

    try:
        set_seed(args.eval_seed + step)
        model.eval()
        losses: list[float] = []
        accuracies: list[float] = []
        exact_matches: list[float] = []
        for _ in range(args.eval_batches):
            batch = make_batch(args, device, split=args.eval_split)
            batch_size, seq_len = batch.input_ids.shape

            state = model.init_recurrent_state(batch_size, device=device)
            all_logits: list[torch.Tensor] = []
            for t in range(seq_len):
                step_logits, state = model.forward_recurrent_step(
                    batch.input_ids[:, t], state
                )
                all_logits.append(step_logits)
            logits = torch.stack(all_logits, dim=1)

            loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
            losses.append(float(loss.detach().cpu()))
            accuracies.append(masked_accuracy(logits, batch.target_ids, batch.loss_mask))
            exact_matches.append(
                masked_exact_match(logits, batch.target_ids, batch.loss_mask)
            )
        return {
            "recurrent_eval_loss": sum(losses) / len(losses),
            "recurrent_eval_accuracy": sum(accuracies) / len(accuracies),
            "recurrent_eval_exact_match": sum(exact_matches) / len(exact_matches),
        }
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            model.train()


def run_variant(
    args: argparse.Namespace, variant: str, out_dir: Path
) -> list[dict[str, float | int | str]]:
    set_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(args, variant, args.vocab_size).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = None
    if getattr(args, "cosine_lr", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.01,
        )
    rows: list[dict[str, float | int | str]] = []

    for step in range(1, args.steps + 1):
        batch = make_batch(args, device, split="train")
        optimizer.zero_grad(set_to_none=True)
        if getattr(args, "gradient_checkpointing", False) and isinstance(model, AnamnesisModel):
            logits = model.forward_chunked(
                batch.input_ids,
                chunk_size=getattr(args, "chunk_size", 512),
            )
            metrics: dict[str, torch.Tensor | None] = {}
        elif getattr(args, "gradient_checkpointing", False):
            from torch.utils.checkpoint import checkpoint as cp
            logits, metrics = cp(
                lambda ids: model(ids, return_metrics=True),
                batch.input_ids,
                use_reentrant=False,
            )
        else:
            logits, metrics = model(batch.input_ids, return_metrics=True)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        if isinstance(metrics, dict) and "gate_loss" in metrics:
            loss = loss + 0.5 * metrics["gate_loss"]
        if isinstance(metrics, dict) and "engram_tcb_distill_loss" in metrics:
            loss = loss + 0.5 * metrics["engram_tcb_distill_loss"]
        loss.backward()
        clip_gradients(args, model, optimizer)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            row: dict[str, float | int | str] = {
                "variant": variant,
                "task": args.task,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "accuracy": masked_accuracy(logits, batch.target_ids, batch.loss_mask),
                "exact_match": masked_exact_match(logits, batch.target_ids, batch.loss_mask),
            }
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    row[key] = float(value.detach().cpu())
            row.update(evaluate_model(args, model, device, step))
            for module_name in args.eval_drop_modules:
                dropped = evaluate_model(args, model, device, step, {module_name})
                for key, value in dropped.items():
                    row[f"eval_no_{module_name}_{key.removeprefix('eval_')}"] = value
            if (
                getattr(args, "eval_recurrent", False)
                and isinstance(model, AnamnesisModel)
                and "eval_loss" in row
            ):
                recurrent = evaluate_model_recurrent(args, model, device, step)
                row.update(recurrent)
            rows.append(row)
            print(
                f"{variant:12s} step={step:5d} "
                f"loss={row['loss']:.4f} acc={row['accuracy']:.3f} "
                f"em={row['exact_match']:.3f}"
                + (
                    f" eval_loss={row['eval_loss']:.4f} eval_em={row['eval_exact_match']:.3f}"
                    if "eval_loss" in row
                    else ""
                )
                + (
                    f" rec_em={row['recurrent_eval_exact_match']:.3f}"
                    if "recurrent_eval_exact_match" in row
                    else ""
                )
            )

    write_csv(out_dir / f"{args.task}_{variant}.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping plot: {exc}")
        return

    by_variant: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    plt.figure(figsize=(8, 5))
    for variant, variant_rows in by_variant.items():
        xs = [int(row["step"]) for row in variant_rows]
        ys = [float(row["loss"]) for row in variant_rows]
        plt.plot(xs, ys, marker="o", label=variant)
    plt.xlabel("step")
    plt.ylabel("masked loss")
    plt.title("Synthetic diagnostic loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="needle")
    parser.add_argument("--variants", default="ours,retnet,transformer")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-branch-optimizer", action="store_true")
    parser.add_argument("--per-group-grad-clip", action="store_true")
    parser.add_argument("--guard-lr-mult", type=float, default=5.0)
    parser.add_argument("--cache-lr-mult", type=float, default=3.0)
    parser.add_argument("--guard-beta1", type=float, default=0.0)
    parser.add_argument("--cache-beta1", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="Device (auto-detects MPS/CUDA if omitted).")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--eval-split", choices=("train", "test", "all"), default="train")
    parser.add_argument(
        "--eval-drop-modules",
        default="",
        help="Comma-separated module-drop eval list: snapshot,engram,attnres.",
    )
    parser.add_argument("--out-dir", default="experiments/results/synthetic")
    parser.add_argument("--engram-layer", type=int, default=2)
    parser.add_argument("--engram-slots", type=int, default=8192)
    parser.add_argument("--engram-max-ngram", type=int, default=3)
    parser.add_argument("--engram-hash-heads", type=int, default=4)
    parser.add_argument(
        "--engram-table-device",
        default=None,
        help="Optional device for Engram hash tables, e.g. cpu for host-memory offload.",
    )
    parser.add_argument("--attnres-every", type=int, default=4)
    parser.add_argument("--attnres-max-sources", type=int, default=8)
    parser.add_argument("--attnres-distance-penalty", type=float, default=0.0)
    parser.add_argument("--branch-init-scale", type=float, default=1e-4)
    parser.add_argument("--use-milestones", action="store_true")
    parser.add_argument("--milestone-ttl", type=int, default=256)
    parser.add_argument("--milestone-gamma", type=float, default=0.999)
    parser.add_argument("--use-snapshots", action="store_true")
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--use-snapshot-logit-bias", action="store_true")
    parser.add_argument("--snapshot-logit-scale", type=float, default=1.0)
    parser.add_argument("--use-token-copy-buffer", action="store_true")
    parser.add_argument("--eval-recurrent", action="store_true",
                        help="Also evaluate with O(1) recurrent step-by-step inference.")
    parser.add_argument("--position-encoding", choices=["learned", "sinusoidal"],
                        default="learned",
                        help="Position encoding type (sinusoidal has no length limit).")
    parser.add_argument(
        "--retention-gamma",
        type=float,
        default=None,
        help="Override all RetNet base decay rates for stress tests.",
    )
    parser.add_argument("--input-dependent-gamma", action="store_true",
                        help="Make retention decay gamma input-dependent (like Mamba).")
    parser.add_argument("--retention-output-gate", action="store_true",
                        help="Add input-dependent output gate to retention (like Mamba).")
    parser.add_argument("--input-dependent-write", action="store_true",
                        help="Add input-dependent write gate: scale k^T v by per-position gate.")
    parser.add_argument("--use-swiglu", action="store_true",
                        help="Use SwiGLU FFN instead of standard FFN.")
    parser.add_argument("--cosine-lr", action="store_true",
                        help="Use cosine LR schedule (decay to 1%% of initial LR).")
    parser.add_argument("--learnable-gamma", action="store_true",
                        help="Make retention decay gamma a learnable parameter.")
    parser.add_argument("--layerwise-gamma", action="store_true",
                        help="Shift gamma range per layer: shallow=short memory, deep=long memory.")
    parser.add_argument("--use-learned-gate", action="store_true",
                        help="Use learned token gate instead of MARK_THOUGHT for TCB selection.")
    parser.add_argument("--use-engram-tcb-trigger", action="store_true",
                        help="Use Engram gate surprise to trigger TCB storage (replaces oracle).")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Use gradient checkpointing to reduce memory at long sequences.")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Chunk size for chunked retention training (used with --gradient-checkpointing).")
    parser.add_argument("--needle-password-len", type=int, default=3)
    parser.add_argument("--alien-num-pairs", type=int, default=4)
    parser.add_argument("--alien-static-key-count", type=int, default=32)
    parser.add_argument("--alien-static-value-count", type=int, default=32)
    parser.add_argument("--alien-static-train-keys", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    if args.d_ff == 0:
        args.d_ff = args.d_model * 4
    args.eval_drop_modules = [
        item.strip() for item in args.eval_drop_modules.split(",") if item.strip()
    ]
    unknown_drop_modules = sorted(set(args.eval_drop_modules) - {"snapshot", "engram", "attnres"})
    if unknown_drop_modules:
        raise ValueError(f"Unknown eval-drop modules: {unknown_drop_modules}")
    if args.task == "alien_static":
        if not (0 < args.alien_static_train_keys < args.alien_static_key_count):
            raise ValueError("--alien-static-train-keys must be in (0, --alien-static-key-count)")
        required_vocab = 16 + args.alien_static_key_count + args.alien_static_value_count + 1
        if args.vocab_size <= required_vocab:
            raise ValueError(
                f"--vocab-size must be > {required_vocab} for alien_static key/value ranges"
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    known = {
        "ours",
        "ours_nosnapshot",
        "ours_snapshot",
        "ours_snapshot_logits",
        "retnet",
        "transformer",
    }
    unknown = sorted(set(variants) - known)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    all_rows: list[dict[str, float | int | str]] = []
    for variant in variants:
        all_rows.extend(run_variant(args, variant, out_dir))

    write_csv(out_dir / f"{args.task}_all.csv", all_rows)
    write_plot(out_dir / f"{args.task}_loss.png", all_rows)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
