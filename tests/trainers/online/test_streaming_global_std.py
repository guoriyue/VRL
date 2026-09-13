"""Streaming must retain optimizer-batch normalization before clipping/filtering."""

import asyncio

import pytest
import torch

from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _diffusion_rollout_batch,
    _trajectory_signals,
)
from tests.trainers.online.test_step_split import _Algorithm, _build_trainer
from vrl.algorithms.advantages import GroupAdvantageEstimator, group_relative_advantages
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.online.config import OnlineBatchPlan


@pytest.mark.parametrize("drop_zero", [False, True])
@pytest.mark.parametrize("micro", [1, 2])
@pytest.mark.parametrize("normalized_components", [False, True])
def test_streaming_matches_full_batch_advantages_gradients_and_adam(
    tmp_path, drop_zero, micro, normalized_components
):
    rewards = {"a": [0.0, 1.0], "b": [0.0, 100.0], "c": [3.0, 3.0], "d": [-2.0, 5.0]}

    class Algorithm(_Algorithm):
        class _Config(_Algorithm._Config):
            global_std = True
            adv_clip_max = 0.5

        config = _Config()

        def compute_advantages_from_tensors(self, rewards, group_ids):
            return group_relative_advantages(
                rewards, group_ids, eps=1e-8, adv_clip_max=0.5, global_std=True
            )

        def compute_loss(self, inputs):
            signals, advantages, _old = _algorithm_inputs(inputs)
            loss = -(signals.log_prob * advantages).mean()
            return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

    async def run(streaming):
        trainer = _build_trainer(tmp_path / str(streaming))
        trainer.algorithm = Algorithm()
        if normalized_components:
            estimator = GroupAdvantageEstimator(
                eps=1e-8,
                adv_clip_max=0.5,
                global_std=True,
                strategy="normalized_sum",
                component_weights={"ocr": 0.7, "quality": 0.3},
            )
            trainer.algorithm.compute_advantages_from_components = (
                lambda values, components, groups: estimator.compute(
                    values, groups, component_rewards=components
                )
            )
        trainer.config.drop_zero_advantage = drop_zero
        plan = OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            prompts_per_collection=micro if streaming else 0,
        )
        trainer.config.batch_plan = plan
        seen = []
        syncs = []

        async def collect(prompts, **kwargs):
            iteration = RolloutIteration(
                batches=[
                    _diffusion_rollout_batch(
                        rewards=torch.tensor(rewards[prompt]),
                        group_ids=torch.full((2,), index, dtype=torch.long),
                        num_steps=2,
                    )
                    for index, prompt in enumerate(prompts)
                ]
            )
            if normalized_components:
                for batch in iteration.batches:
                    batch.extras["reward_components"] = {
                        "ocr": batch.rewards.tolist(),
                        "quality": batch.rewards.square().tolist(),
                    }
            return iteration

        def evaluate(model, batch, timestep_idx, **kwargs):
            return _trajectory_signals(
                batch,
                model.weight.reshape(()) * batch.rewards,
                timestep_idx,
                old_log_prob=batch.rewards,
            )

        original_loss = trainer.algorithm.compute_loss

        def record(inputs):
            seen.append(inputs.advantages.detach().clone())
            return original_loss(inputs)

        trainer.algorithm.compute_loss = record
        trainer.rollout_schedule.next_iteration = collect
        trainer.evaluator.evaluate = evaluate
        original_sync = trainer.rollout_schedule.after_train_step

        async def sync():
            syncs.append(trainer.model.weight.detach().clone())
            return await original_sync()

        trainer.rollout_schedule.after_train_step = sync
        if streaming:
            metrics = await _run_streaming_optimizer_update(
                trainer, list(rewards), batch_plan=plan
            )
        else:
            metrics = await trainer.step(list(rewards))
        assert not list((tmp_path / str(streaming)).glob(".advantage-spool-*"))
        return trainer, metrics, seen, syncs

    full, fm, fa, fs = asyncio.run(run(False))
    streamed, sm, sa, ss = asyncio.run(run(True))
    assert len(fs) == len(ss) == 1
    torch.testing.assert_close(fs[0], ss[0], rtol=0, atol=0)
    assert len(fa) == len(sa)
    for left, right in zip(fa, sa, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert sm.grad_norm == pytest.approx(fm.grad_norm, rel=1e-6, abs=1e-8)
    assert sm.reward_std == pytest.approx(fm.reward_std)
    torch.testing.assert_close(full.model.weight, streamed.model.weight, rtol=0, atol=0)
    left = full._optimizer.state_dict()
    right = streamed._optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    for key, state in left["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, right["state"][key][name], rtol=0, atol=0)
    assert full.state.global_step == streamed.state.global_step == 1


@pytest.mark.parametrize("stage", ["collect", "replay"])
def test_spool_cleanup_on_failure(tmp_path, stage):
    trainer = _build_trainer(tmp_path)
    trainer.algorithm.config = type("Config", (), {"global_std": True, "adv_clip_max": 5.0})()
    plan = OnlineBatchPlan(prompts_per_batch=2, n_samples_per_prompt=2, prompts_per_collection=1)
    trainer.config.batch_plan = plan
    calls = []

    async def collect(prompts, **kwargs):
        calls.append(prompts)
        if len(calls) == 2 and stage == "collect":
            assert list(tmp_path.glob(".advantage-spool-*/*.pt"))
            raise RuntimeError("injected collection failure")
        return RolloutIteration(
            batches=[
                _diffusion_rollout_batch(
                    rewards=torch.tensor([0.0, 1.0]),
                    group_ids=torch.zeros(2, dtype=torch.long),
                    num_steps=2,
                )
            ]
        )

    def backward(*args, **kwargs):
        assert len(calls) == 2
        raise RuntimeError("injected replay failure")

    trainer.rollout_schedule.next_iteration = collect
    trainer.backward_on_training_batch = backward
    with pytest.raises(RuntimeError, match="injected"):
        asyncio.run(_run_streaming_optimizer_update(trainer, ["a", "b"], batch_plan=plan))
    assert not list(tmp_path.glob(".advantage-spool-*"))
    assert trainer.state.global_step == 0
