"""FP16 GradScaler acceptance gates for OnlineTrainer (SPRINT_fp16_training_gradscaler).

G1 enable matrix, G2 unscale-before-clip ordering, G4 skipped-step propagation
(the scaler-skipped step must not update EMA or the algorithm adapter). G3
(checkpoint round-trip) lives in test_state_restore.py.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    DEFAULT_FORWARD_PRECISION,
    _algorithm_inputs,
    _rollout_context,
    _trajectory_signals,
)
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig, TrainerConfig
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.trainer import _create_grad_scaler
from vrl.trainers.optimizer import FP32MasterWeightOptimizer
from vrl.trainers.strategy import SingleProcessStrategy


# --------------------------------------------------------------------------
# G1 — scaler follows effective FP16 backward risk, not outer-autocast alone
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("autocast", "device", "parameter_dtype", "expected"),
    [
        ("fp16", "cuda", torch.float32, True),  # ordinary AMP
        ("fp16", "cpu", torch.float32, False),
        ("bf16", "cuda", torch.float32, False),
        ("off", "cuda", torch.float32, False),
        # SANA: no outer autocast, but its native FP16 gradient buffers still
        # need scaling before they are copied to FP32 master parameters.
        ("off", "cuda", torch.float16, True),
        ("off", "cuda", torch.bfloat16, False),
    ],
)
def test_create_grad_scaler_matrix(
    monkeypatch: pytest.MonkeyPatch,
    autocast,
    device,
    parameter_dtype,
    expected,
) -> None:
    from vrl.models.interfaces import ResolvedForwardPrecision

    precision = ResolvedForwardPrecision(autocast=autocast, float32_precision="ieee")
    model = nn.Linear(1, 1, bias=False).to(dtype=parameter_dtype)
    sentinel = object()
    monkeypatch.setattr(torch.amp, "GradScaler", lambda _device: sentinel)

    scaler = _create_grad_scaler(precision, torch.device(device), model=model)

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
    trainer = SimpleNamespace(
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
    trainer = SimpleNamespace(
        config=SimpleNamespace(max_norm=1.0),
        model=model,
        _grad_scaler=scaler,
        _strategy=SingleProcessStrategy(),
    )

    grad_norm, stepped = OnlineTrainer._clip_and_step(trainer, optimizer)

    assert grad_norm == pytest.approx(1.0)
    assert stepped is True
    assert model.weight.item() == pytest.approx(0.89990234375)


def test_low_precision_trainables_derive_master_optimizer_without_config_knob() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.model = nn.Linear(1, 1, bias=False).half()
    trainer.config = SimpleNamespace(optim=OptimConfig(lr=1.0e-5))
    trainer._optimizer = None

    optimizer = trainer._ensure_optimizer()

    assert isinstance(optimizer, FP32MasterWeightOptimizer)
    assert next(optimizer.parameters()).dtype is torch.float32
    assert trainer.model.weight.dtype is torch.float16


def test_fp32_trainables_use_direct_optimizer_without_duplicate_master() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.model = nn.Linear(1, 1, bias=False).float()
    trainer.config = SimpleNamespace(optim=OptimConfig(lr=1.0e-5))
    trainer._optimizer = None

    optimizer = trainer._ensure_optimizer()

    assert not isinstance(optimizer, FP32MasterWeightOptimizer)
    assert optimizer.param_groups[0]["params"][0] is trainer.model.weight
    assert trainer.model.weight.dtype is torch.float32


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
class _Algorithm:
    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        kl_coef = 0.0

    config = _Config()

    def __init__(self):
        self.after_step_calls: list[int] = []

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def compute_loss(self, inputs):
        signals, _adv, _old = _algorithm_inputs(inputs)
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

    def after_optimizer_step(self, model, global_step) -> None:
        del model
        self.after_step_calls.append(global_step)


class _Collector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        group_size = int(kwargs["group_size"])
        return RolloutBatch(
            observations=torch.zeros(group_size, 2, 1),
            actions=torch.zeros(group_size, 2, 1),
            rewards=torch.arange(group_size, dtype=torch.float32),
            dones=torch.ones(group_size, dtype=torch.bool),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            context=_rollout_context(),
            prompts=list(prompts) * group_size,
        )


class _Evaluator(Evaluator):
    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw
        return _trajectory_signals(
            batch,
            model.weight.view(1).expand(batch.rewards.shape[0]),
            timestep_idx,
        )


class _SpyEMA:
    def __init__(self):
        self.steps: list[int] = []

    def step(self, trainable, global_step):
        del trainable
        self.steps.append(global_step)


def _build_trainer(tmp_path):
    algorithm = _Algorithm()
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            prompts_per_batch=1,
            timestep_fraction=1.0,
            total_epochs=1,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(enable=True, update_interval=1),
            n_samples_per_prompt=2,
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        forward_precision=DEFAULT_FORWARD_PRECISION,
        device="cpu",
    )
    spy_ema = _SpyEMA()
    trainer._ema = spy_ema  # _ensure_ema returns the existing instance
    return trainer, algorithm, spy_ema


def test_skipped_step_does_not_update_ema_or_adapter(tmp_path) -> None:
    trainer, algorithm, spy_ema = _build_trainer(tmp_path)
    trainer._clip_and_step = lambda optimizer: (0.0, False)  # type: ignore[method-assign]

    asyncio.run(trainer.step(["p"]))

    assert algorithm.after_step_calls == []
    assert spy_ema.steps == []
    assert trainer.state.global_step == 1  # the step still counts as an iteration


def test_applied_step_updates_ema_and_adapter(tmp_path) -> None:
    trainer, algorithm, spy_ema = _build_trainer(tmp_path)
    trainer._clip_and_step = lambda optimizer: (0.0, True)  # type: ignore[method-assign]

    asyncio.run(trainer.step(["p"]))

    assert algorithm.after_step_calls  # fired
    assert spy_ema.steps  # fired
    assert trainer.state.global_step == 1
