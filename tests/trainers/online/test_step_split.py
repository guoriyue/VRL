"""P4 collect/train split: the decomposition must not change behavior.

``OnlineTrainer.step`` now runs ``collect_training_batch`` then
``train_on_rollout_batch``. These lock that the split reproduces the previous
single-method behavior: identical metrics and state, the collected batch carries
the data the train half needs, and rollout weight sync still happens in the train
half (not the collect half).
"""

from __future__ import annotations

import asyncio

import torch
import torch.nn as nn

from tests.trainers.online._helpers import _algorithm_inputs, _trajectory_signals
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig, TrainerConfig
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.trainer import TrainingBatch


class _Algorithm:
    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        init_kl_coef = 0.0

    config = _Config()

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def compute_loss(self, inputs):
        signals, _advantages, _old_log_probs = _algorithm_inputs(inputs)
        loss = signals.log_prob.mean()
        return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())


class _Collector:
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        group_size = int(kwargs.get("group_size", 1))
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


def _build_trainer(tmp_path) -> OnlineTrainer:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return OnlineTrainer(
        algorithm=_Algorithm(),
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            rollout_batch_size=1,
            timestep_fraction=1.0,
            total_epochs=1,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            n_samples_per_prompt=2,
            bf16=False,
            mixed_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
    )


def _metric_fields(m: TrainStepMetrics) -> tuple:
    return (
        m.loss,
        m.policy_loss,
        m.reward_mean,
        m.reward_std,
        m.advantage_mean,
        m.grad_norm,
        m.group_size,
        m.trained_prompt_num,
        m.adv_zero_rate,
        m.adv_saturation,
    )


def test_step_equals_collect_then_train(tmp_path) -> None:
    """step() and an explicit collect+train produce identical metrics + state."""
    mono = _build_trainer(tmp_path / "mono")
    mono_metrics = asyncio.run(mono.step(["p"]))

    split = _build_trainer(tmp_path / "split")

    async def _run_split() -> TrainStepMetrics:
        batch = await split.collect_training_batch(["p"])
        return await split.train_on_rollout_batch(batch)

    split_metrics = asyncio.run(_run_split())

    assert _metric_fields(split_metrics) == _metric_fields(mono_metrics)
    assert split.state.step == mono.state.step
    assert split.state.global_step == mono.state.global_step
    assert torch.equal(split.model.weight, mono.model.weight)


def test_collect_returns_training_batch_with_data(tmp_path) -> None:
    trainer = _build_trainer(tmp_path)
    batch = asyncio.run(trainer.collect_training_batch(["p"]))

    assert isinstance(batch, TrainingBatch)
    assert batch.batches and batch.advantages
    assert len(batch.batches) == len(batch.advantages)
    # collect must not advance training state — that is the train half's job.
    assert trainer.state.step == 0
    assert trainer.state.global_step == 0


def test_weight_sync_happens_in_train_half_not_collect(tmp_path) -> None:
    """rollout_schedule.after_train_step fires in train, never during collect."""
    trainer = _build_trainer(tmp_path)
    calls: list[int] = []
    real_after = trainer.rollout_schedule.after_train_step

    async def _counting_after():
        calls.append(1)
        return await real_after()

    trainer.rollout_schedule.after_train_step = _counting_after  # type: ignore[method-assign]

    batch = asyncio.run(trainer.collect_training_batch(["p"]))
    assert calls == []  # collect half does not sync

    asyncio.run(trainer.train_on_rollout_batch(batch))
    assert calls == [1]  # train half syncs exactly once
