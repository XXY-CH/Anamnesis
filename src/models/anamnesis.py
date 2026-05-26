"""Dense RetNet + Block AttnRes + hashed Engram language model.

This is the small practice architecture aligned with the current proof stack:
- RetNet: small-state sequence mixing.
- Dense FFN: phase-1 channel-mixing control baseline; MoE is deferred.
- Block AttnRes: depth-axis residual readout.
- Engram: deterministic N-gram hash lookup with gated residual injection.
- Milestone gate: optional project extension for time-axis preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.attention_residual import BlockAttentionResidual
from ..layers.engram import HashedNgramEngram
from ..layers.milestone_gate import MilestoneRetentionGate
from ..layers.milestone_snapshot import MilestoneSnapshotReadout
from ..layers.retention import RetentionLayer
from ..layers.token_copy_buffer import TokenCopyBuffer
from ..layers.learned_token_gate import LearnedTokenGate
from .recurrent_state import RecurrentState


@dataclass
class AnamnesisConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 8
    d_ff: int | None = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    engram_layers: tuple[int, ...] = (2,)
    engram_num_slots: int = 4096
    engram_max_ngram: int = 3
    engram_hash_heads: int = 4
    engram_table_device: str | None = None
    engram_use_conv: bool = True
    engram_vector_gate: bool = False
    attnres_every: int = 4
    attnres_max_sources: int = 8
    attnres_distance_penalty: float = 0.0
    branch_init_scale: float = 1e-4
    milestone_token_ids: tuple[int, ...] = field(default_factory=tuple)
    milestone_ttl: int = 256
    milestone_gamma: float = 0.999
    use_milestone_snapshots: bool = False
    max_milestone_snapshots: int = 8
    use_snapshot_logit_bias: bool = False
    snapshot_logit_scale: float = 1.0
    use_token_copy_buffer: bool = False
    token_copy_use_pos_keys: bool = True
    token_copy_sinusoidal_pos: bool = False
    position_encoding_type: str = "learned"  # "learned" or "sinusoidal"
    input_dependent_gamma: bool = False
    retention_output_gate: bool = False
    use_learned_gate: bool = False
    use_engram_tcb_trigger: bool = False


def sinusoidal_encoding(
    positions: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    """Functional sinusoidal position encoding, works at any position index.

    Args:
        positions: [...,] integer position indices.
        d_model: encoding dimension (must be even).

    Returns:
        [..., d_model] position encodings.
    """
    half = d_model // 2
    freq = torch.arange(half, device=positions.device, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (freq / half))
    angles = positions.float().unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class AnamnesisLayer(nn.Module):
    """One Dense RetNet block with optional Engram and Block AttnRes branches."""

    def __init__(
        self,
        config: AnamnesisConfig,
        layer_idx: int,
        use_engram: bool,
        use_attnres: bool,
    ) -> None:
        super().__init__()
        d_ff = config.d_ff or config.d_model * 4
        self.layer_idx = layer_idx
        self.use_engram = use_engram
        self.use_attnres = use_attnres

        self.retention_norm = nn.RMSNorm(config.d_model)
        self.retention = RetentionLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            input_dependent_gamma=config.input_dependent_gamma,
        )

        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_ff, config.d_model),
        )

        self.engram = (
            HashedNgramEngram(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                num_slots=config.engram_num_slots,
                max_ngram=config.engram_max_ngram,
                num_hash_heads=config.engram_hash_heads,
                init_scale=config.branch_init_scale,
                dropout=config.dropout,
                table_device=config.engram_table_device,
                use_conv=config.engram_use_conv,
                vector_gate=config.engram_vector_gate,
            )
            if use_engram
            else None
        )
        self.attnres = (
            BlockAttentionResidual(
                d_model=config.d_model,
                max_sources=config.attnres_max_sources,
                init_scale=config.branch_init_scale,
                distance_penalty=config.attnres_distance_penalty,
                dropout=config.dropout,
            )
            if use_attnres
            else None
        )
        self.retention_output_gate = (
            nn.Linear(config.d_model, config.d_model, bias=True)
            if config.retention_output_gate
            else None
        )
        if self.retention_output_gate is not None:
            nn.init.zeros_(self.retention_output_gate.weight)
            nn.init.constant_(self.retention_output_gate.bias, 2.0)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        depth_sources: list[torch.Tensor],
        retention_gate: torch.Tensor | None,
        disable_engram: bool = False,
        disable_attnres: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        metrics: dict[str, torch.Tensor | None] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        u = self.retention_norm(x)
        retention_out = self.retention(u, retention_gate=retention_gate)
        if not isinstance(retention_out, torch.Tensor):
            retention_out = retention_out[0]

        if self.retention_output_gate is not None:
            retention_out = retention_out * torch.sigmoid(
                self.retention_output_gate(u)
            )

        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + retention_out + ffn_out

        if self.engram is not None and not disable_engram:
            engram_residual, engram_gate = self.engram(self.ffn_norm(x), input_ids)
            x = x + engram_residual
            metrics[f"layer_{self.layer_idx}_engram_gate"] = engram_gate
            metrics[f"layer_{self.layer_idx}_engram_gate_mean"] = engram_gate.mean()
            metrics[f"layer_{self.layer_idx}_engram_scale"] = self.engram.residual_scale.detach()
            if return_diagnostics:
                diagnostics[f"layer_{self.layer_idx}_engram_gate"] = engram_gate.detach()

        if self.attnres is not None and not disable_attnres:
            attnres_residual, attn_weights = self.attnres(self.ffn_norm(x), depth_sources)
            x = x + attnres_residual
            metrics[f"layer_{self.layer_idx}_attnres_scale"] = self.attnres.residual_scale.detach()
            if attn_weights is not None:
                metrics[f"layer_{self.layer_idx}_attnres_entropy"] = (
                    -(attn_weights.clamp_min(1e-9).log() * attn_weights).sum(dim=-1).mean()
                )
                if return_diagnostics:
                    diagnostics[f"layer_{self.layer_idx}_attnres_weights"] = attn_weights.detach()

        return x, metrics, diagnostics


class AnamnesisModel(nn.Module):
    """Small language model aligned to the Dense-first phase-1 proof architecture."""

    def __init__(self, config: AnamnesisConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.use_sinusoidal_pe = config.position_encoding_type == "sinusoidal"
        if not self.use_sinusoidal_pe:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        engram_layers = set(config.engram_layers)
        self.layers = nn.ModuleList(
            [
                AnamnesisLayer(
                    config=config,
                    layer_idx=i,
                    use_engram=i in engram_layers,
                    use_attnres=config.attnres_every > 0 and (i + 1) % config.attnres_every == 0,
                )
                for i in range(config.n_layers)
            ]
        )
        self.milestone_gate = MilestoneRetentionGate(
            n_heads=config.n_heads,
            milestone_token_ids=config.milestone_token_ids,
            protected_gamma=config.milestone_gamma,
            ttl=config.milestone_ttl,
        )
        self.snapshot_readout = (
            MilestoneSnapshotReadout(
                d_model=config.d_model,
                max_snapshots=config.max_milestone_snapshots,
                init_scale=config.branch_init_scale,
                dropout=config.dropout,
            )
            if config.use_milestone_snapshots
            else None
        )
        self.token_copy_buffer = (
            TokenCopyBuffer(
                d_model=config.d_model,
                max_snapshots=config.max_milestone_snapshots,
                max_seq_len=config.max_seq_len,
                init_scale=config.branch_init_scale,
                dropout=config.dropout,
                use_pos_keys=config.token_copy_use_pos_keys,
                use_sinusoidal_pos=config.token_copy_sinusoidal_pos,
            )
            if config.use_token_copy_buffer
            else None
        )
        self.learned_gate = (
            LearnedTokenGate(
                d_model=config.d_model,
                max_store=config.max_milestone_snapshots,
            )
            if config.use_learned_gate and config.use_token_copy_buffer
            else None
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_metrics: bool = False,
        disable_engram: bool = False,
        disable_attnres: bool = False,
        disable_snapshots: bool = False,
        return_diagnostics: bool = False,
        return_hidden_only: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[str, torch.Tensor | None]]
        | tuple[torch.Tensor, dict[str, torch.Tensor | None], dict[str, torch.Tensor]]
    ):
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len and not self.use_sinusoidal_pe:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        if self.use_sinusoidal_pe:
            pos_enc = sinusoidal_encoding(positions, self.config.d_model)
        else:
            pos_enc = self.position_embedding(positions)
        x = self.dropout(self.token_embedding(input_ids) + pos_enc)

        retention_gate = self.milestone_gate(
            input_ids,
            base_gamma=self.layers[0].retention.gamma,
        )
        milestone_mask = self._milestone_mask(input_ids)
        snapshot_source_mask = self._pre_milestone_mask(milestone_mask)

        depth_sources: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor | None] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        token_copy_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        engram_gates_for_tcb: list[torch.Tensor] = []
        use_engram_trigger = self.config.use_engram_tcb_trigger and not disable_snapshots

        if self.token_copy_buffer is not None and not disable_snapshots and not use_engram_trigger:
            token_emb = self.token_embedding(input_ids)
            oracle = self._content_before_milestone_mask(milestone_mask)
            if self.learned_gate is not None:
                gate_mask, gate_scores = self.learned_gate(x)
                if self.training and oracle.any():
                    copy_source_mask = oracle
                    metrics["gate_loss"] = self.learned_gate.compute_loss(x, oracle)
                elif gate_mask is not None:
                    copy_source_mask = gate_mask
                else:
                    copy_source_mask = torch.zeros_like(milestone_mask, dtype=torch.bool)
            else:
                copy_source_mask = oracle
            collected = self.token_copy_buffer.collect(token_emb, copy_source_mask)
            if collected is not None:
                token_copy_cache = collected

        snapshot_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        for layer in self.layers:
            x, layer_metrics, layer_diagnostics = layer(
                x=x,
                input_ids=input_ids,
                depth_sources=depth_sources,
                retention_gate=retention_gate,
                disable_engram=disable_engram,
                disable_attnres=disable_attnres,
                return_diagnostics=return_diagnostics,
            )
            if use_engram_trigger and layer.use_engram:
                gate_key = f"layer_{layer.layer_idx}_engram_gate"
                if gate_key in layer_metrics:
                    engram_gates_for_tcb.append(layer_metrics[gate_key])
            metrics.update(layer_metrics)
            diagnostics.update(layer_diagnostics)
            if self.snapshot_readout is not None and not disable_snapshots:
                collected = self.snapshot_readout.collect(x, snapshot_source_mask)
                if collected is not None:
                    snapshot_cache = collected
            depth_sources.append(x.detach() if not self.training else x)

        # Engram-triggered TCB: collect after layers using Engram gate surprise
        if use_engram_trigger and engram_gates_for_tcb:
            token_emb = self.token_embedding(input_ids)
            stacked = torch.stack(engram_gates_for_tcb, dim=0)
            surprise = stacked.mean(dim=(0, -1))  # [batch, seq_len]
            # During training: oracle drives TCB, Engram learns via distillation loss
            if self.training:
                oracle = self._content_before_milestone_mask(milestone_mask)
                if oracle.any():
                    metrics["engram_tcb_distill_loss"] = self._engram_tcb_distill_loss(
                        surprise, oracle
                    )
                copy_source_mask = oracle
            else:
                # At inference: Engram surprise drives TCB
                copy_source_mask = surprise
            collected = self.token_copy_buffer.collect(token_emb, copy_source_mask)
            if collected is not None:
                token_copy_cache = collected
            metrics["engram_tcb_surprise_mean"] = surprise.mean()
            metrics["engram_tcb_surprise_max"] = surprise.max()

        if self.snapshot_readout is not None and not disable_snapshots:
            snapshot_residual, snapshot_weights = self.snapshot_readout(
                self.final_norm(x),
                snapshot_cache,
            )
            x = x + snapshot_residual
            metrics["snapshot_scale"] = self.snapshot_readout.residual_scale.detach()
            if snapshot_weights is not None:
                metrics["snapshot_entropy"] = (
                    -(snapshot_weights.clamp_min(1e-9).log() * snapshot_weights).sum(dim=-1).mean()
                )
                metrics["snapshot_valid_count"] = snapshot_cache[1].sum(dim=-1).float().mean()
                if return_diagnostics:
                    diagnostics["snapshot_weights"] = snapshot_weights.detach()
                    diagnostics["snapshot_valid"] = snapshot_cache[1].detach()

        final_hidden = self.final_norm(x)
        if return_hidden_only:
            return final_hidden
        logits = self.output_head(final_hidden)
        if self.token_copy_buffer is not None and not disable_snapshots and token_copy_cache is not None:
            copy_readout, copy_weights = self.token_copy_buffer(final_hidden, token_copy_cache)
            copy_logits = torch.matmul(copy_readout, self.token_embedding.weight.t())
            logits = logits + copy_logits
            if return_metrics:
                metrics["token_copy_scale"] = self.token_copy_buffer.residual_scale.abs().detach()
                if copy_weights is not None:
                    metrics["token_copy_entropy"] = (
                        -(copy_weights.clamp_min(1e-9).log() * copy_weights).sum(dim=-1).mean()
                    )
                    metrics["token_copy_valid_count"] = token_copy_cache[1].sum(dim=-1).float().mean()
            if return_diagnostics and copy_weights is not None:
                diagnostics["token_copy_weights"] = copy_weights.detach()
                diagnostics["token_copy_valid"] = token_copy_cache[1].detach()
                diagnostics["token_copy_pos_ids"] = token_copy_cache[2].detach()
        if (
            self.snapshot_readout is not None
            and self.config.use_snapshot_logit_bias
            and not disable_snapshots
            and snapshot_cache is not None
        ):
            snapshot_logits, _ = self.snapshot_readout(final_hidden, snapshot_cache)
            logits = logits + self.config.snapshot_logit_scale * torch.matmul(
                snapshot_logits,
                self.token_embedding.weight.t(),
            )
        if return_metrics:
            if retention_gate is not None:
                metrics["milestone_gate_mean"] = retention_gate.mean()
                if return_diagnostics:
                    diagnostics["milestone_gate"] = retention_gate.detach()
            if return_diagnostics:
                return logits, metrics, diagnostics
            return logits, metrics
        if return_diagnostics:
            return logits, metrics, diagnostics
        return logits

    def forward_chunked(
        self,
        input_ids: torch.Tensor,
        chunk_size: int = 512,
        disable_engram: bool = False,
        disable_attnres: bool = False,
        disable_snapshots: bool = False,
    ) -> torch.Tensor:
        """Forward pass with chunked retention for long-sequence evaluation.

        Identical to forward() but processes each layer's retention in chunks
        using chunkwise_retention, enabling evaluation beyond training length.
        All masks, gates, and readout mechanisms use the full sequence.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        cfg = self.config

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        if self.use_sinusoidal_pe:
            pos_enc = sinusoidal_encoding(positions, cfg.d_model)
        else:
            pos_enc = self.position_embedding(positions)
        full_emb = self.dropout(self.token_embedding(input_ids) + pos_enc)

        retention_gate = self.milestone_gate(
            input_ids, base_gamma=self.layers[0].retention.gamma,
        )
        milestone_mask = self._milestone_mask(input_ids)
        snapshot_source_mask = self._pre_milestone_mask(milestone_mask)

        token_copy_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if self.token_copy_buffer is not None and not disable_snapshots:
            token_emb = self.token_embedding(input_ids)
            oracle = self._content_before_milestone_mask(milestone_mask)
            if self.learned_gate is not None:
                gate_mask, gate_scores = self.learned_gate(full_emb)
                if self.training and oracle is not None and oracle.any():
                    copy_source_mask = oracle
                else:
                    copy_source_mask = gate_mask
            else:
                copy_source_mask = oracle
            collected = self.token_copy_buffer.collect(token_emb, copy_source_mask)
            if collected is not None:
                token_copy_cache = collected

        ret_states: list[torch.Tensor | None] = [None] * cfg.n_layers
        depth_sources: list[torch.Tensor] = []
        snapshot_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        layer_input = full_emb

        for layer_idx, layer in enumerate(self.layers):
            chunk_outputs: list[torch.Tensor] = []

            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                x = layer_input[:, chunk_start:chunk_end]
                chunk_ids = input_ids[:, chunk_start:chunk_end]

                chunk_gate = (
                    retention_gate[:, chunk_start:chunk_end]
                    if retention_gate is not None
                    else None
                )

                u = layer.retention_norm(x)
                ret_out, new_state = layer.retention.chunkwise_retention(
                    u, state=ret_states[layer_idx], retention_gate=chunk_gate,
                )
                ret_states[layer_idx] = new_state

                if layer.retention_output_gate is not None:
                    ret_out = ret_out * torch.sigmoid(
                        layer.retention_output_gate(u)
                    )

                ffn_out = layer.ffn(layer.ffn_norm(x))
                x = x + ret_out + ffn_out

                if layer.engram is not None and not disable_engram:
                    eng_res, _ = layer.engram(layer.ffn_norm(x), chunk_ids)
                    x = x + eng_res

                if layer.attnres is not None and not disable_attnres and depth_sources:
                    active = [
                        src[:, chunk_start:chunk_end]
                        for src in depth_sources[-layer.attnres.max_sources:]
                    ]
                    attnres_res, _ = layer.attnres(layer.ffn_norm(x), active)
                    x = x + attnres_res

                chunk_outputs.append(x)

            x_full = torch.cat(chunk_outputs, dim=1)

            if self.snapshot_readout is not None and not disable_snapshots:
                collected = self.snapshot_readout.collect(x_full, snapshot_source_mask)
                if collected is not None:
                    snapshot_cache = collected

            depth_sources.append(x_full.detach() if not self.training else x_full)
            layer_input = x_full

        x = x_full

        if (
            self.snapshot_readout is not None
            and not disable_snapshots
            and snapshot_cache is not None
        ):
            snapshot_residual, _ = self.snapshot_readout(
                self.final_norm(x), snapshot_cache,
            )
            x = x + snapshot_residual

        final_hidden = self.final_norm(x)
        logits = self.output_head(final_hidden)

        if (
            self.token_copy_buffer is not None
            and not disable_snapshots
            and token_copy_cache is not None
        ):
            copy_readout, _ = self.token_copy_buffer(final_hidden, token_copy_cache)
            copy_logits = torch.matmul(copy_readout, self.token_embedding.weight.t())
            logits = logits + copy_logits

        return logits

    def forward_recurrent_step(
        self,
        input_id: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Single-step O(1) recurrent inference.

        Processes one token position, updating the recurrent state.
        Memory usage is constant regardless of total sequence length.

        Args:
            input_id: Token IDs for this step, shape [batch].
            state: Current recurrent state.

        Returns:
            (logits [batch, vocab_size], updated RecurrentState)
        """
        device = input_id.device
        batch = input_id.shape[0]
        pos = state.position
        cfg = self.config

        # --- Embed ---
        positions = torch.full((batch,), pos, device=device, dtype=torch.long)
        if self.use_sinusoidal_pe:
            pos_enc = sinusoidal_encoding(positions, cfg.d_model)
        else:
            pos_enc = self.position_embedding(positions)
        x = self.dropout(
            self.token_embedding(input_id) + pos_enc
        ).unsqueeze(1)

        # --- Milestone detection ---
        is_milestone = torch.zeros(batch, device=device, dtype=torch.bool)
        for tid in cfg.milestone_token_ids:
            is_milestone |= input_id == tid

        # --- Retention gate (manual TTL window, bypassing MilestoneRetentionGate) ---
        steps_since = state.steps_since_milestone
        if is_milestone.any():
            steps_since = 0
        else:
            steps_since += 1

        if cfg.milestone_token_ids and steps_since < cfg.milestone_ttl:
            base_gamma = self.layers[0].retention.gamma.to(device=device)
            retention_gate = torch.maximum(
                base_gamma.unsqueeze(0).expand(batch, -1),
                torch.full((batch, cfg.n_heads), cfg.milestone_gamma, device=device),
            )
        else:
            retention_gate = None

        # --- Layer processing ---
        new_ret_states: list[torch.Tensor] = []
        depth_sources: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.layers):
            u = layer.retention_norm(x)
            ret_out, new_ret = layer.retention.recurrent_retention(
                u,
                state=state.retention_states[layer_idx],
                retention_gate=retention_gate,
            )
            new_ret_states.append(new_ret)

            if layer.retention_output_gate is not None:
                ret_out = ret_out * torch.sigmoid(
                    layer.retention_output_gate(u)
                )

            ffn_out = layer.ffn(layer.ffn_norm(x))
            x = x + ret_out + ffn_out

            if layer.engram is not None:
                eng_res, _ = layer.engram(layer.ffn_norm(x), input_id.unsqueeze(1))
                x = x + eng_res

            if layer.attnres is not None and depth_sources:
                active = depth_sources[-layer.attnres.max_sources :]
                attnres_res, _ = layer.attnres(layer.ffn_norm(x), active)
                x = x + attnres_res

            depth_sources.append(x.detach() if not self.training else x)

        # Save post-layer hidden for snapshot timing (before readouts modify x).
        x_for_snap = x.squeeze(1)

        # --- Snapshot collection: store prev_hidden when current is milestone ---
        snap_stored = state.snap_stored
        snap_valid = state.snap_valid
        snap_write_idx = state.snap_write_idx

        if (
            self.snapshot_readout is not None
            and is_milestone.any()
            and state.prev_hidden is not None
            and snap_stored is not None
            and snap_write_idx < cfg.max_milestone_snapshots
        ):
            snap_stored = snap_stored.clone()
            snap_valid = snap_valid.clone()
            snap_stored[:, snap_write_idx] = state.prev_hidden
            snap_valid[:, snap_write_idx] = is_milestone
            snap_write_idx += 1

        # --- Snapshot readout ---
        if (
            self.snapshot_readout is not None
            and snap_stored is not None
            and snap_valid is not None
        ):
            snap_cache = (snap_stored, snap_valid)
            snap_residual, _ = self.snapshot_readout(self.final_norm(x), snap_cache)
            x = x + snap_residual

        # --- TokenCopyBuffer: collect non-milestone tokens, freeze at milestone ---
        # Only active when milestones are configured (parallel mode uses a look-ahead
        # mask that is all-zeros without milestones, making the buffer a no-op).
        has_milestones = bool(cfg.milestone_token_ids)
        copy_stored = state.copy_stored
        copy_valid = state.copy_valid
        copy_pos_ids = state.copy_pos_ids
        copy_frozen = state.copy_frozen
        copy_write_idx = state.copy_write_idx

        if (
            self.token_copy_buffer is not None
            and has_milestones
            and not copy_frozen
            and copy_stored is not None
            and not is_milestone.all()
            and copy_write_idx < cfg.max_milestone_snapshots
        ):
            copy_stored = copy_stored.clone()
            copy_valid = copy_valid.clone()
            copy_pos_ids = copy_pos_ids.clone()
            store_mask = ~is_milestone
            token_emb = self.token_embedding(input_id)
            copy_stored[:, copy_write_idx] = torch.where(
                store_mask.unsqueeze(-1),
                token_emb,
                copy_stored[:, copy_write_idx],
            )
            copy_pos_ids[:, copy_write_idx] = torch.where(
                store_mask,
                positions,
                copy_pos_ids[:, copy_write_idx],
            )
            copy_valid[:, copy_write_idx] = (
                copy_valid[:, copy_write_idx] | store_mask
            )
            copy_write_idx += 1

        if is_milestone.any():
            copy_frozen = True

        # --- Final norm + output ---
        final_hidden = self.final_norm(x)
        logits = self.output_head(final_hidden)

        # --- Copy buffer readout ---
        if (
            self.token_copy_buffer is not None
            and has_milestones
            and copy_stored is not None
            and copy_valid is not None
            and copy_pos_ids is not None
            and copy_valid.any()
        ):
            buffer_cache = (copy_stored, copy_valid, copy_pos_ids)
            copy_readout, _ = self.token_copy_buffer(final_hidden, buffer_cache)
            copy_logits = torch.matmul(
                copy_readout, self.token_embedding.weight.t()
            )
            logits = logits + copy_logits

        # --- Snapshot logit bias ---
        if (
            self.snapshot_readout is not None
            and cfg.use_snapshot_logit_bias
            and snap_stored is not None
            and snap_valid is not None
        ):
            snap_cache = (snap_stored, snap_valid)
            snap_logits, _ = self.snapshot_readout(final_hidden, snap_cache)
            logits = logits + cfg.snapshot_logit_scale * torch.matmul(
                snap_logits, self.token_embedding.weight.t()
            )

        # --- Build updated state ---
        new_state = RecurrentState(
            position=pos + 1,
            retention_states=new_ret_states,
            copy_stored=copy_stored,
            copy_valid=copy_valid,
            copy_pos_ids=copy_pos_ids,
            copy_frozen=copy_frozen,
            copy_write_idx=copy_write_idx,
            snap_stored=snap_stored,
            snap_valid=snap_valid,
            snap_write_idx=snap_write_idx,
            prev_hidden=(
                x_for_snap.detach() if self.training else x_for_snap
            ),
            steps_since_milestone=steps_since,
        )

        return logits.squeeze(1), new_state

    def init_recurrent_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> RecurrentState:
        """Create an empty recurrent state for this model's configuration."""
        head_dim = self.config.d_model // self.config.n_heads
        return RecurrentState.init_empty(
            batch_size=batch_size,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            head_dim=head_dim,
            max_snapshots=self.config.max_milestone_snapshots,
            d_model=self.config.d_model,
            device=device,
            with_copy_buffer=self.token_copy_buffer is not None,
            with_snapshots=self.snapshot_readout is not None,
        )

    def _milestone_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.config.milestone_token_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.config.milestone_token_ids:
            mask |= input_ids == token_id
        return mask

    @staticmethod
    def _pre_milestone_mask(milestone_mask: torch.Tensor) -> torch.Tensor:
        source_mask = torch.zeros_like(milestone_mask)
        source_mask[:, :-1] = milestone_mask[:, 1:]
        return source_mask

    @staticmethod
    def _content_before_milestone_mask(milestone_mask: torch.Tensor) -> torch.Tensor:
        """Mark all positions before any milestone for token copy buffer."""
        has_future = milestone_mask.flip(dims=[1]).cummax(dim=1).values.flip(dims=[1])
        source_mask = has_future.roll(-1, dims=1)
        source_mask[:, -1] = False
        return source_mask & ~milestone_mask

    @staticmethod
    def _engram_tcb_distill_loss(
        surprise: torch.Tensor, oracle: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted BCE: train Engram surprise to predict oracle TCB mask."""
        n_pos = oracle.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        n_neg = oracle.shape[-1] - n_pos
        pos_weight = (n_neg / n_pos).clamp(max=200.0)
        return F.binary_cross_entropy_with_logits(
            surprise, oracle.float(), pos_weight=pos_weight,
        )
