"""Trajectory-granularity replay scheduling in the online trainer."""

from __future__ import annotations

import asyncio

import pytest
import torch
import torch.nn as nn

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig


class _Algorithm:
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
        signals, advantages, old_log_probs = _algorithm_inputs(inputs)
        del advantages, old_log_probs
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


class _Collector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        prompts = [getattr(item, "prompt", item) for item in prompts]
        group_size = int(kwargs["group_size"])
        batch_size = len(prompts) * group_size
        return RolloutBatch(
            # [sample, temporal_chunk, denoise_transition, feature]
            observations=torch.zeros(batch_size, 4, 3, 1),
            actions=torch.zeros(batch_size, 4, 3, 1),
            rewards=torch.arange(batch_size, dtype=torch.float32),
            dones=torch.ones(batch_size, dtype=torch.bool),
            group_ids=torch.zeros(batch_size, dtype=torch.long),
            prompts=[prompt for prompt in prompts for _ in range(group_size)],
            context={},
        )


class _TrajectoryEvaluator:
    replay_granularity = "trajectory"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def evaluate(self, model, batch, timestep_idx, **kwargs):
        del kwargs
        assert tuple(batch.observations.shape[1:3]) == (4, 3)
        self.calls.append(int(timestep_idx))
        log_prob = model.weight.reshape(()).expand(batch.rewards.shape[0])
        return _trajectory_signals(batch, log_prob, timestep_idx)


@pytest.mark.parametrize("streaming", [False, True])
def test_trajectory_evaluator_runs_once_for_chunk_transition_axes(streaming: bool) -> None:
    evaluator = _TrajectoryEvaluator()
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    with torch.no_grad():
        model.weight.zero_()
    batch_plan = OnlineBatchPlan(
        prompts_per_batch=1,
        n_samples_per_prompt=2,
        gradient_accumulation_steps=1 if streaming else 0,
        replay_samples_per_chunk=0,
    )
    trainer = OnlineTrainer(
        algorithm=_Algorithm(),
        collector=_Collector(),
        evaluator=evaluator,
        model=model,
        config=TrainerConfig(
            batch_plan=batch_plan,
            timestep_fraction=0.25,
            total_epochs=1,
            drop_zero_advantage=False,
            output_dir="outputs/",
            optim=OptimConfig(lr=0.0),
            ema=EMAConfig(),
            debug=DebugConfig(),
            precision_drift_guard=PrecisionDriftGuardConfig(mode="off"),
        ),
        device="cpu",
    )

    if streaming:
        asyncio.run(
            _run_streaming_optimizer_update(
                trainer,
                ["prompt"],
                batch_plan=batch_plan,
            ),
        )
    else:
        asyncio.run(trainer.step(["prompt"]))

    assert evaluator.calls == [0]


def test_unknown_replay_granularity_fails_fast() -> None:
    trainer = object.__new__(OnlineTrainer)
    trainer.evaluator = type("Evaluator", (), {"replay_granularity": "chunk"})()

    with pytest.raises(ValueError, match="replay_granularity"):
        trainer._train_replay_indices(4, 1.0)
