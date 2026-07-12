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
from tests.trainers.online._helpers import _algorithm_inputs, _trajectory_signals
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig, TrainerConfig
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.trainer import _needs_grad_scaler
from vrl.trainers.strategy import SingleProcessStrategy


# --------------------------------------------------------------------------
# G1 — scaler is enabled only for native (no-accelerator) cuda fp16 training
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mixed_precision", "device", "accelerator", "expected"),
    [
        ("fp16", "cuda", None, True),  # the one path that needs scaling
        ("fp16", "cpu", None, False),  # fp16 autocast only resolves on cuda
        ("bf16", "cuda", None, False),  # bf16 has fp32 range — no underflow
        ("no", "cuda", None, False),  # fp32 — nothing to scale
        ("fp16", "cuda", object(), False),  # accelerator self-manages a scaler
    ],
)
def test_needs_grad_scaler_matrix(mixed_precision, device, accelerator, expected) -> None:
    config = SimpleNamespace(train_precision=mixed_precision)
    assert (
        _needs_grad_scaler(config, torch.device(device), model=None, accelerator=accelerator)
        is expected
    )


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
        accelerator=None,
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
            context={},
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
