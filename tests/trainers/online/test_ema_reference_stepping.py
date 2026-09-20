"""``actor.ema.step_per_microbatch``: the reference loop's shadow stepping.

Flash-GRPO's reference calls ``ema.step`` after EVERY replay microbatch with
the optimizer-step counter, and increments that counter before the last
microbatch's call (it steps the optimizer, bumps ``global_step``, then steps
the shadow). With ``n`` microbatches per update the shadow therefore sees
``n - 1`` calls at the current counter and one at the next. Off, the trainer
steps the shadow once per optimizer step at the current counter.
"""

from __future__ import annotations

import asyncio

import torch
import torch.nn as nn

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _diffusion_rollout_batch,
    _EvaluatorAlgorithmFake,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.online import ema as ema_module
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import OnlineTrainer

_GROUP = 3


class _Algorithm(_EvaluatorAlgorithmFake):
    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        kl_coef = 0.0

    config = _Config()

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def compute_loss(self, inputs):
        signals, _advantages, _old = _algorithm_inputs(inputs)
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


class _Collector(CollectorControlFake):
    async def evaluate_rollout(self, pendings):
        return list(pendings)

    async def generate_rollout(self, request):
        group_size = int(request.options["group_size"])
        return _diffusion_rollout_batch(
            rewards=torch.arange(group_size, dtype=torch.float32),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            num_steps=1,
        )


class _Evaluator(Evaluator):
    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw
        return _trajectory_signals(
            batch,
            model.weight.view(1).expand(batch.rewards.shape[0]),
            timestep_idx,
        )


def _trainer(tmp_path, *, step_per_microbatch: bool) -> OnlineTrainer:
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    return OnlineTrainer(
        algorithm=_Algorithm(),
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            # One collected group of three samples, one sample per replay
            # microbatch: three microbatches per optimizer update.
            batch_plan=OnlineBatchPlan(
                prompts_per_batch=1,
                n_samples_per_prompt=_GROUP,
                training_microbatch_size=1,
            ),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(
                enable=True,
                decay=0.9,
                update_interval=1,
                step_per_microbatch=step_per_microbatch,
            ),
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
    )


def _record_ema_steps(monkeypatch) -> list[int]:
    counters: list[int] = []
    original = ema_module.EMAWeights.step

    def spy(self, parameters, optimization_step):
        counters.append(int(optimization_step))
        return original(self, parameters, optimization_step)

    monkeypatch.setattr(ema_module.EMAWeights, "step", spy)
    return counters


def test_reference_stepping_calls_the_shadow_per_microbatch(tmp_path, monkeypatch) -> None:
    counters = _record_ema_steps(monkeypatch)
    trainer = _trainer(tmp_path, step_per_microbatch=True)

    asyncio.run(trainer.step(["p1"]))
    asyncio.run(trainer.step(["p1"]))

    # Update 0: two microbatch calls at counter 0, the last one after the
    # optimizer step at counter 1; update 1 likewise at 1 then 2.
    assert counters == [0, 0, 1, 1, 1, 2]


def test_default_stepping_is_once_per_optimizer_step(tmp_path, monkeypatch) -> None:
    counters = _record_ema_steps(monkeypatch)
    trainer = _trainer(tmp_path, step_per_microbatch=False)

    asyncio.run(trainer.step(["p1"]))
    asyncio.run(trainer.step(["p1"]))

    assert counters == [0, 1]
