"""FP16 GradScaler acceptance gates for OnlineTrainer (SPRINT_fp16_training_gradscaler).

G1 enable matrix, G2 unscale-before-clip ordering, G4 skipped-step propagation
(the scaler-skipped step must not update EMA or publish rollout weights; it runs
on the real GRPO trainer, whose algorithm has no ``after_optimizer_step``
adapter hook). G3 (checkpoint round-trip) lives in test_state_restore.py.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.rollouts.collector._helpers import Trace
from tests.trainers.online._helpers import TrainerBench, bare_trainer, real_trainer
from vrl.config.precision import RolePrecision
from vrl.trainers.online.ema import EMAWeights
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.optimizer import FP32MasterWeightOptimizer
from vrl.trainers.strategy import SingleProcessStrategy


# --------------------------------------------------------------------------
# G1 — scaler follows effective FP16 backward risk, not outer autocast alone
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "outer_autocast", "device", "parameter_dtype", "expected"),
    [
        ("fp16", True, "cuda", torch.float32, True),  # ordinary AMP
        ("fp16", True, "cpu", torch.float32, False),
        ("bf16", True, "cuda", torch.float32, False),
        ("fp32", False, "cuda", torch.float32, False),
        # SANA: no outer autocast, but its native FP16 gradient buffers still
        # need scaling before they are copied to FP32 master parameters.
        ("fp16", False, "cuda", torch.float16, True),
        ("bf16", False, "cuda", torch.bfloat16, False),
    ],
)
def test_create_grad_scaler_matrix(
    monkeypatch: pytest.MonkeyPatch,
    dtype,
    outer_autocast,
    device,
    parameter_dtype,
    expected,
) -> None:
    model = nn.Linear(1, 1, bias=False).to(dtype=parameter_dtype)
    # The role precision a RuntimeBundle stamps on its model.
    model.precision = RolePrecision(
        dtype=dtype,
        float32_precision="ieee",
        outer_autocast=outer_autocast,
    )
    sentinel = object()
    monkeypatch.setattr(torch.amp, "GradScaler", lambda _device: sentinel)

    scaler = OnlineTrainer._create_grad_scaler(torch.device(device), model=model)

    assert (scaler is sentinel) is expected
    assert (scaler is None) is (not expected)


# --------------------------------------------------------------------------
# Real cpu GradScaler for calling OnlineTrainer._clip_and_step in isolation:
# scale(loss).backward() primes genuinely scaled grads, an injected inf grad
# drives the real backoff path, and the weight itself witnesses stepped/skipped.
# --------------------------------------------------------------------------
def _scaler_trainer(*, growth_interval: int = 2000, max_norm: float = 1.0):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    scaler = torch.amp.GradScaler(
        "cpu", enabled=True, init_scale=1024.0, growth_interval=growth_interval
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler.scale(model.weight.sum()).backward()  # grad now carries the 1024 scale
    trainer = bare_trainer(
        config=SimpleNamespace(max_norm=max_norm),
        model=model,
        _grad_scaler=scaler,
        _strategy=SingleProcessStrategy(),
    )
    return trainer, optimizer, model


# --------------------------------------------------------------------------
# G2 — unscale_ must happen before clip_grad_norm_ (else clip sees scaled grads)
# --------------------------------------------------------------------------
def test_unscale_runs_before_clip(monkeypatch) -> None:
    """The grad magnitude observed inside clip is the ordering witness: the
    scaled grad reads 1024.0, the unscaled one exactly 1.0."""
    seen: list[float] = []

    def probe_clip(parameters, max_norm, **kwargs):
        del max_norm, kwargs
        seen.extend(float(p.grad.abs().max()) for p in parameters if p.grad is not None)
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", probe_clip)
    trainer, optimizer, _model = _scaler_trainer()

    OnlineTrainer._clip_and_step(trainer, optimizer)

    assert seen == [1.0], seen


def test_clip_and_step_prepares_fp32_master_before_standard_unscale() -> None:
    model = nn.Linear(1, 1, bias=False).half()
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = FP32MasterWeightOptimizer(
        model.parameters(),
        lambda parameters: torch.optim.SGD(parameters, lr=0.1),
    )
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    scaler.scale(model.weight.float().sum()).backward()
    trainer = bare_trainer(
        config=SimpleNamespace(max_norm=1.0),
        model=model,
        _grad_scaler=scaler,
        _strategy=SingleProcessStrategy(),
    )

    grad_norm, stepped = OnlineTrainer._clip_and_step(trainer, optimizer)

    assert grad_norm == pytest.approx(1.0)
    assert stepped is True
    assert model.weight.item() == pytest.approx(0.89990234375)


def test_clip_and_step_refuses_nonfinite_gradient_without_scaler(monkeypatch, tmp_path) -> None:
    """Without a GradScaler (CPU, BF16, FP32) a non-finite norm is never committed."""

    tb = real_trainer(monkeypatch, tmp_path)
    assert tb.trainer._grad_scaler is None
    optimizer = tb.trainer._ensure_optimizer()
    parameters = tb.trainable_parameters()
    before = {name: value.detach().clone() for name, value in parameters.items()}
    for value in parameters.values():
        value.grad = torch.full_like(value, float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite gradient norm"):
        tb.trainer._clip_and_step(optimizer)

    for name, value in parameters.items():
        assert torch.equal(value.detach(), before[name])
        assert value.grad is None


def test_clip_and_step_lets_scaler_skip_nonfinite_gradient() -> None:
    """inf grads under a GradScaler are its designed backoff event, not an error.

    Unlike test_clip_and_step_reports_skipped this does NOT monkeypatch
    clip_grad_norm_, so the real (infinite) norm reaches the finiteness guard —
    the exact path the guard must leave to the scaler.
    """
    trainer, optimizer, model = _scaler_trainer()
    model.weight.grad.fill_(float("inf"))
    scale_before = trainer._grad_scaler.get_scale()

    grad_norm, stepped = OnlineTrainer._clip_and_step(trainer, optimizer)

    assert grad_norm == float("inf")
    assert stepped is False
    assert model.weight.item() == pytest.approx(1.0)  # skipped, not corrupted
    assert trainer._grad_scaler.get_scale() < scale_before  # real backoff happened


def test_low_precision_trainables_derive_master_optimizer_without_config_knob(
    monkeypatch, tmp_path
) -> None:
    tb = real_trainer(monkeypatch, tmp_path, overrides=("precision.training.dtype=fp16",))
    trainable = next(iter(tb.trainable_parameters().values()))

    optimizer = tb.trainer._ensure_optimizer()

    assert isinstance(optimizer, FP32MasterWeightOptimizer)
    assert next(optimizer.parameters()).dtype is torch.float32
    assert trainable.dtype is torch.float16


def test_fp32_trainables_use_direct_optimizer_without_duplicate_master(
    monkeypatch, tmp_path
) -> None:
    tb = real_trainer(monkeypatch, tmp_path)
    trainables = list(tb.trainable_parameters().values())

    optimizer = tb.trainer._ensure_optimizer()

    assert not isinstance(optimizer, FP32MasterWeightOptimizer)
    assert optimizer.param_groups[0]["params"][0] is trainables[0]
    assert all(value.dtype is torch.float32 for value in trainables)


# --------------------------------------------------------------------------
# G4 (unit) — stepped reflects whether the scaler applied the update
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("inf_grad", "growth_interval", "expected_stepped"),
    [
        (True, 2000, False),  # inf grad -> real backoff (1024 -> 512), step skipped
        (False, 2000, True),  # finite, scale unchanged -> stepped
        (False, 1, True),  # finite, scale grows (1024 -> 2048) -> stepped
    ],
)
def test_clip_and_step_reports_skipped(
    monkeypatch, inf_grad, growth_interval, expected_stepped
) -> None:
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(0.0))
    trainer, optimizer, model = _scaler_trainer(growth_interval=growth_interval)
    if inf_grad:
        model.weight.grad.fill_(float("inf"))

    _grad_norm, stepped = OnlineTrainer._clip_and_step(trainer, optimizer)

    assert stepped is expected_stepped
    # stepped must reflect what really happened to the weights.
    assert (model.weight.item() != 1.0) is expected_stepped


# --------------------------------------------------------------------------
# G4 (integration) — a skipped step must not fire EMA or after_optimizer_step
# --------------------------------------------------------------------------
# The only producer of a skipped step is the CUDA fp16 GradScaler's backoff; a
# CPU trainer has no scaler. The skip is therefore the scaler's outcome applied
# to the real trainer: no optimizer.step, gradients cleared, stepped=False.
def _scaler_skips_every_step(monkeypatch, tb: TrainerBench) -> None:
    def skipped(optimizer):
        optimizer.zero_grad()
        return float("inf"), False

    monkeypatch.setattr(tb.trainer, "_clip_and_step", skipped)


def _trainer_with_ema(monkeypatch, tmp_path) -> tuple[TrainerBench, Trace]:
    tb = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            "actor.ppo_epochs=1",
            "actor.drop_zero_advantage=false",
            "actor.ema.enable=true",
            "actor.ema.update_interval=1",
        ),
    )
    shadow = Trace(monkeypatch)
    shadow.watch(EMAWeights, "step", "ema_step")
    return tb, shadow


def test_skipped_step_does_not_update_ema_or_rollout_weights(monkeypatch, tmp_path) -> None:
    tb, shadow = _trainer_with_ema(monkeypatch, tmp_path)
    _scaler_skips_every_step(monkeypatch, tb)
    before = {name: value.detach().clone() for name, value in tb.trainable_parameters().items()}

    asyncio.run(tb.trainer.step(["p"]))

    assert shadow.events == []
    # Only the initial policy push: the skipped update is never published.
    assert tb.collector.trace.events.count("update_weights") == 1
    assert tb.collector.runtime.current_policy_version == 1
    for name, value in tb.trainable_parameters().items():
        assert torch.equal(value.detach(), before[name])
    assert tb.trainer.state.global_step == 1  # the step still counts as an iteration


def test_applied_step_updates_ema_and_rollout_weights(monkeypatch, tmp_path) -> None:
    tb, shadow = _trainer_with_ema(monkeypatch, tmp_path)

    asyncio.run(tb.trainer.step(["p"]))

    assert shadow.events == ["ema_step"]
    assert tb.collector.trace.events.count("update_weights") == 2
    assert tb.collector.runtime.current_policy_version == 2
    assert tb.trainer.state.global_step == 1
