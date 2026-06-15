"""SingleProcessStrategy parity (readiness P3).

The strategy seam moves the existing backward / clip / state-export logic behind
``TrainingStrategy``. These lock that the moved methods reproduce the underlying
helpers exactly, so installing the seam changes nothing for a single-GPU run.
"""

from __future__ import annotations

import torch
from torch import nn

from vrl.trainers.strategy import SingleProcessStrategy
from vrl.trainers.weight_sync import build_trainable_state_sync_getter


class _Bundle:
    def __init__(self) -> None:
        self.trainable_modules = {"adapter": nn.Linear(2, 1, bias=False)}


def test_default_context_is_single_process() -> None:
    ctx = SingleProcessStrategy().context
    assert (ctx.strategy, ctx.rank, ctx.world_size, ctx.is_primary) == (
        "single_process",
        0,
        1,
        True,
    )


def test_backward_matches_plain_backward() -> None:
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    ref = nn.Linear(3, 1)
    strat = nn.Linear(3, 1)
    strat.load_state_dict(ref.state_dict())

    ref(x).sum().backward()
    SingleProcessStrategy().backward(strat(x).sum())

    for p_ref, p_strat in zip(ref.parameters(), strat.parameters(), strict=True):
        assert torch.equal(p_ref.grad, p_strat.grad)


def test_backward_scales_loss_when_grad_scaler_present() -> None:
    calls: list[str] = []

    class _Scaler:
        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            calls.append("scale")
            return loss

    leaf = nn.Linear(1, 1)
    SingleProcessStrategy().backward(leaf(torch.ones(1, 1)).sum(), grad_scaler=_Scaler())

    assert calls == ["scale"]


def test_clip_grad_norm_matches_torch_in_place() -> None:
    ref = nn.Linear(4, 4)
    strat = nn.Linear(4, 4)
    strat.load_state_dict(ref.state_dict())
    for p in ref.parameters():
        p.grad = torch.ones_like(p)
    for p in strat.parameters():
        p.grad = torch.ones_like(p)

    ref_norm = float(nn.utils.clip_grad_norm_(ref.parameters(), 0.5))
    strat_norm = SingleProcessStrategy().clip_grad_norm(strat.parameters(), 0.5)

    assert strat_norm == ref_norm
    for p_ref, p_strat in zip(ref.parameters(), strat.parameters(), strict=True):
        assert torch.equal(p_ref.grad, p_strat.grad)  # clipped identically


def test_export_rollout_state_matches_helper() -> None:
    bundle = _Bundle()
    got = SingleProcessStrategy().export_rollout_state(bundle)
    expected = build_trainable_state_sync_getter(bundle)()

    assert got.keys() == expected.keys()
    for key in got:
        assert torch.equal(got[key], expected[key])


def test_export_and_load_trainable_state_round_trip() -> None:
    strat = SingleProcessStrategy()
    src = _Bundle()
    with torch.no_grad():
        src.trainable_modules["adapter"].weight.fill_(3.0)
    snapshot = strat.export_trainable_state(src)
    assert set(snapshot) == {"adapter"}

    dst = _Bundle()
    strat.load_trainable_state(dst, snapshot)
    assert torch.equal(
        dst.trainable_modules["adapter"].weight,
        src.trainable_modules["adapter"].weight,
    )


def test_barrier_is_noop() -> None:
    assert SingleProcessStrategy().barrier() is None
