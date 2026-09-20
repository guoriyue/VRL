"""Several optimizer steps on one collected batch (``actor.optimizer_steps_per_batch``).

Flash-GRPO's reference collects a whole epoch, computes advantages once (global
std over the epoch), shuffles the samples and trains them as two accumulation
windows, each ending in an optimizer step. The trainer reproduces that by
dealing every group's samples across the requested number of updates after ONE
advantage computation; each update prepares its own objective state and steps
the optimizer, and the samples of the updates are disjoint and complete.
"""

from __future__ import annotations

import asyncio

import pytest
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
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import OnlineTrainer

_GROUP = 4


class _Algorithm(_EvaluatorAlgorithmFake):
    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        kl_coef = 0.0

    config = _Config()

    def __init__(self) -> None:
        self.advantage_calls = 0
        self.prepared_rewards: list[torch.Tensor] = []
        self.losses = 0

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        self.advantage_calls += 1
        return rewards - rewards.mean()

    def prepare_update(self, timesteps, *, scheduler, noise_level, sde_type):
        del scheduler, noise_level, sde_type
        # One entry per (sample, trained step); the trajectory stores the
        # sample's reward as its timestep so the update's sample set is readable.
        self.prepared_rewards.append(timesteps.clone())

    def compute_loss(self, inputs):
        signals, _advantages, _old = _algorithm_inputs(inputs)
        self.losses += 1
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


class _Collector(CollectorControlFake):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def evaluate_rollout(self, pendings):
        return list(pendings)

    async def generate_rollout(self, request):
        group_size = int(request.options["group_size"])
        # Distinct rewards per sample, stored as the trajectory timestep so an
        # update's sample set is readable from what it prepares.
        rewards = torch.arange(group_size, dtype=torch.float32) + 10.0 * self.calls
        self.calls += 1
        return _diffusion_rollout_batch(
            rewards=rewards,
            group_ids=torch.zeros(group_size, dtype=torch.long),
            num_steps=1,
            timesteps=rewards.view(-1, 1).clone(),
        )


class _Evaluator(Evaluator):
    scheduler = object()

    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw
        return _trajectory_signals(
            batch,
            model.weight.view(1).expand(batch.rewards.shape[0]),
            timestep_idx,
        )


def _trainer(tmp_path, *, optimizer_steps_per_batch: int) -> tuple[OnlineTrainer, _Algorithm]:
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    algorithm = _Algorithm()
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(
                prompts_per_batch=2,
                n_samples_per_prompt=_GROUP,
                optimizer_steps_per_batch=optimizer_steps_per_batch,
            ),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
    )
    return trainer, algorithm


def test_two_steps_partition_the_batch_after_one_advantage_pass(tmp_path) -> None:
    trainer, algorithm = _trainer(tmp_path, optimizer_steps_per_batch=2)

    async def _run():
        batch = await trainer.collect_training_batch(["p1", "p2"])
        await trainer.train_on_rollout_batch(batch)
        return batch

    batch = asyncio.run(_run())

    assert algorithm.advantage_calls == 1
    assert trainer.state.global_step == 2
    assert trainer.state.step == 1
    # Two updates, each prepared with its own sample set; together they are
    # exactly the collected samples, none repeated.
    assert len(algorithm.prepared_rewards) == 2
    collected = sorted(torch.cat([b.rewards for b in batch.batches]).tolist())
    union = torch.cat(algorithm.prepared_rewards)
    assert sorted(union.tolist()) == collected
    # Every group is dealt across both updates (not cut group by group).
    for group in batch.batches:
        members = set(group.rewards.tolist())
        for prepared in algorithm.prepared_rewards:
            assert members & set(prepared.tolist())


def test_one_step_keeps_the_single_update(tmp_path) -> None:
    trainer, algorithm = _trainer(tmp_path, optimizer_steps_per_batch=1)

    asyncio.run(trainer.step(["p1", "p2"]))

    assert trainer.state.global_step == 1
    assert len(algorithm.prepared_rewards) == 1


def test_dealing_is_deterministic_for_a_resumed_counter(tmp_path) -> None:
    first, first_algorithm = _trainer(tmp_path / "a", optimizer_steps_per_batch=2)
    second, second_algorithm = _trainer(tmp_path / "b", optimizer_steps_per_batch=2)

    asyncio.run(first.step(["p1", "p2"]))
    asyncio.run(second.step(["p1", "p2"]))

    for a, b in zip(
        first_algorithm.prepared_rewards, second_algorithm.prepared_rewards, strict=True
    ):
        assert torch.equal(a, b)


def test_streaming_rejects_several_steps_per_batch() -> None:
    with pytest.raises(ValueError, match="optimizer_steps_per_batch>1"):
        OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            prompts_per_collection=2,
            optimizer_steps_per_batch=2,
        )
