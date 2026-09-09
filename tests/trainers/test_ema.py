"""Numeric tests for the EMA shadow-weight wrapper (online/ema.py).

These assert the decay *math* (warmup schedule + update recurrence + interval
gating + temp swap), not just that store/restore runs without crashing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vrl.trainers.online.ema import EMAModuleWrapper


def _single_param(value: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor([value], dtype=torch.float64))


def test_get_current_decay_warmup_schedule() -> None:
    """Warmup ramps as (1+s)/(10+s), capped at the configured decay."""
    ema = EMAModuleWrapper([_single_param(0.0)], decay=0.99)
    assert ema.get_current_decay(0) == pytest.approx(1 / 10)
    assert ema.get_current_decay(9) == pytest.approx(10 / 19)
    # Far enough along, the ramp exceeds 0.99 and is clamped to it.
    assert ema.get_current_decay(100_000) == pytest.approx(0.99)


def test_step_matches_analytic_recurrence() -> None:
    """ema += (1-decay)*(param-ema) at each on-interval step, decay=warmup."""
    param = _single_param(10.0)
    ema = EMAModuleWrapper([param], decay=1.0, update_step_interval=1)
    # Start the shadow at a known offset from the live param.
    ema.ema_parameters[0].fill_(0.0)

    expected = 0.0
    for step in range(5):
        ema.step([param], step)
        decay = min((1 + step) / (10 + step), 1.0)
        expected += (1 - decay) * (10.0 - expected)
        assert ema.ema_parameters[0].item() == pytest.approx(expected)
    assert ema.num_updates == 5


def test_update_step_interval_gates_updates() -> None:
    """Only steps where (step+1) % interval == 0 mutate the shadow."""
    param = _single_param(10.0)
    ema = EMAModuleWrapper([param], decay=1.0, update_step_interval=4)
    ema.ema_parameters[0].fill_(0.0)

    # Steps 0,1,2 are off-boundary → no change, no update count.
    for step in (0, 1, 2):
        ema.step([param], step)
        assert ema.ema_parameters[0].item() == pytest.approx(0.0)
        assert ema.num_updates == 0

    # Step 3 → (3+1)%4 == 0 → one update fires.
    ema.step([param], 3)
    assert ema.num_updates == 1
    assert ema.ema_parameters[0].item() != pytest.approx(0.0)


def test_non_trainable_params_are_skipped() -> None:
    """Frozen params (requires_grad=False) are not pulled into the EMA."""
    frozen = nn.Parameter(torch.tensor([5.0], dtype=torch.float64), requires_grad=False)
    ema = EMAModuleWrapper([frozen], decay=1.0, update_step_interval=1)
    ema.ema_parameters[0].fill_(0.0)
    ema.step([frozen], 0)
    assert ema.ema_parameters[0].item() == pytest.approx(0.0)


def test_copy_ema_to_then_copy_temp_to_restores_exactly() -> None:
    """Swap EMA weights in for eval, then restore the originals byte-for-byte."""
    param = _single_param(10.0)
    ema = EMAModuleWrapper([param], decay=1.0, update_step_interval=1)
    ema.ema_parameters[0].fill_(3.0)

    original = param.detach().clone()
    ema.copy_ema_to([param])
    assert param.item() == pytest.approx(3.0)  # now holding EMA value

    ema.copy_temp_to([param])
    assert torch.equal(param.detach(), original)
    assert ema.temp_stored_parameters is None


def test_failed_ema_swap_restores_every_parameter_before_raising() -> None:
    first = _single_param(1.0)
    second = _single_param(2.0)
    ema = EMAModuleWrapper([first, second])
    ema.ema_parameters[0].fill_(7.0)
    ema.ema_parameters[1] = torch.ones(2)

    with pytest.raises(RuntimeError):
        ema.copy_ema_to([first, second])

    assert first.item() == pytest.approx(1.0)
    assert second.item() == pytest.approx(2.0)
    assert ema.temp_stored_parameters is None


def test_copy_temp_to_rejects_missing_snapshot() -> None:
    param = _single_param(1.0)
    ema = EMAModuleWrapper([param])

    with pytest.raises(RuntimeError, match="snapshot is not available"):
        ema.copy_temp_to([param])


def test_repeated_ema_swap_preserves_original_snapshot() -> None:
    param = _single_param(10.0)
    ema = EMAModuleWrapper([param])
    ema.ema_parameters[0].fill_(3.0)
    ema.copy_ema_to([param])
    snapshot = ema.temp_stored_parameters

    with pytest.raises(RuntimeError, match="restore original parameters"):
        ema.copy_ema_to([param])

    assert ema.temp_stored_parameters is snapshot
    assert param.item() == pytest.approx(3.0)
    ema.copy_temp_to([param])
    assert param.item() == pytest.approx(10.0)


@pytest.mark.parametrize("is_primary", [True, False])
def test_both_checkpoint_paths_snapshot_away_from_the_live_shadow(is_primary: bool) -> None:
    """A written checkpoint must not change when the next step moves the shadow.

    `state_dict` used to hand back the live shadow tensor whenever it was not a
    DTensor, so a later EMA step silently rewrote an already-taken snapshot;
    `checkpoint_state_dict` always cloned. Both take the same copy now.
    """

    param = _single_param(1.0)
    ema = EMAModuleWrapper([param], decay=0.5, update_step_interval=1)

    plain = ema.state_dict()["ema_parameters"][0]
    sharded = ema.checkpoint_state_dict(is_primary=is_primary).get("ema_parameters")

    param.data.fill_(100.0)
    ema.step([param], 0)

    assert ema.ema_parameters[0].item() != pytest.approx(1.0)
    assert plain.item() == pytest.approx(1.0)
    assert plain.data_ptr() != ema.ema_parameters[0].data_ptr()
    if is_primary:
        assert sharded is not None
        assert sharded[0].item() == pytest.approx(1.0)
    else:
        # Non-primary ranks join the gather and keep nothing.
        assert sharded is None
