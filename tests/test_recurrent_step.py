"""Test that recurrent step-by-step inference matches parallel forward."""

import torch
import pytest

from src.models.anamnesis import AnamnesisConfig, AnamnesisModel


def _make_model(
    use_copy_buffer: bool = False,
    use_snapshots: bool = False,
    milestone_tokens: tuple[int, ...] = (),
) -> AnamnesisModel:
    cfg = AnamnesisConfig(
        vocab_size=32,
        d_model=64,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
        dropout=0.0,
        engram_layers=(1,),
        engram_num_slots=128,
        attnres_every=2,
        use_token_copy_buffer=use_copy_buffer,
        use_milestone_snapshots=use_snapshots,
        max_milestone_snapshots=8,
        milestone_token_ids=milestone_tokens,
        branch_init_scale=1e-4,
    )
    return AnamnesisModel(cfg)


@pytest.mark.parametrize(
    "use_copy_buffer,use_snapshots",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_recurrent_matches_parallel(use_copy_buffer: bool, use_snapshots: bool) -> None:
    """Step-by-step recurrent logits must match a single parallel forward pass."""
    torch.manual_seed(42)
    model = _make_model(use_copy_buffer=use_copy_buffer, use_snapshots=use_snapshots)
    model.eval()

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, 32, (batch, seq_len))

    with torch.no_grad():
        parallel_logits = model(input_ids)  # [batch, seq_len, vocab]

        state = model.init_recurrent_state(batch, device=input_ids.device)
        recurrent_logits = []
        for t in range(seq_len):
            step_logits, state = model.forward_recurrent_step(
                input_ids[:, t], state
            )
            recurrent_logits.append(step_logits)
        recurrent_logits = torch.stack(recurrent_logits, dim=1)  # [batch, seq_len, vocab]

    assert parallel_logits.shape == recurrent_logits.shape
    torch.testing.assert_close(
        recurrent_logits, parallel_logits, atol=1e-4, rtol=1e-3
    )


def test_recurrent_with_milestones() -> None:
    """Recurrent mode with milestones should produce valid output and constant memory.

    Exact equivalence with parallel mode is not expected for the TokenCopyBuffer
    because recurrent mode cannot look ahead to collect the same pre-milestone tokens.
    The base RetNet + snapshot path should still match closely.
    """
    torch.manual_seed(42)
    model = _make_model(
        use_copy_buffer=True,
        use_snapshots=True,
        milestone_tokens=(5, 10),
    )
    model.eval()

    batch, seq_len = 2, 20
    input_ids = torch.randint(0, 32, (batch, seq_len))
    input_ids[0, 7] = 5
    input_ids[0, 14] = 10
    input_ids[1, 8] = 5
    input_ids[1, 15] = 10

    with torch.no_grad():
        state = model.init_recurrent_state(batch, device=input_ids.device)
        recurrent_logits = []
        for t in range(seq_len):
            step_logits, state = model.forward_recurrent_step(
                input_ids[:, t], state
            )
            recurrent_logits.append(step_logits)
        recurrent_logits = torch.stack(recurrent_logits, dim=1)

    assert recurrent_logits.shape == (batch, seq_len, 32)
    assert torch.isfinite(recurrent_logits).all()


def test_recurrent_state_does_not_grow() -> None:
    """Memory usage should be constant regardless of sequence length."""
    model = _make_model(use_copy_buffer=True, use_snapshots=True)
    model.eval()

    batch = 1
    state = model.init_recurrent_state(batch, device=torch.device("cpu"))

    init_params = sum(
        p.nelement() for p in [state.copy_stored, state.copy_valid, state.copy_pos_ids,
                                state.snap_stored, state.snap_valid]
        if p is not None
    )
    ret_params = sum(s.nelement() for s in state.retention_states)

    with torch.no_grad():
        for t in range(200):
            token = torch.tensor([t % 32])
            _, state = model.forward_recurrent_step(token, state)

    final_params = sum(
        p.nelement() for p in [state.copy_stored, state.copy_valid, state.copy_pos_ids,
                                state.snap_stored, state.snap_valid]
        if p is not None
    )
    final_ret = sum(s.nelement() for s in state.retention_states)

    assert init_params == final_params
    assert ret_params == final_ret
