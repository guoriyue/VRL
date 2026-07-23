"""OnlineTrainer advantage normalization, KL propagation, and zero-advantage gradient behavior."""

from __future__ import annotations

import pytest

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    DEFAULT_PRECISION,
    _algorithm_inputs,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.rollouts.evaluators.base import Evaluator


class TestAdvantageAndMetrics:
    """Groups tests for advantage and metrics."""

    def _make_cea_trainer(
        self,
        rewards: list[float],
        *,
        ppo_epochs: int = 1,
        emit_diagnostics: bool = False,
        gradient_accumulation_steps: int = 0,
    ):
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                kl_coef = 0.0

            config = _Config()

            def __init__(self) -> None:
                self.loss_calls = 0

            def compute_advantages_from_tensors(self, rewards, group_ids):
                advantages = torch.zeros_like(rewards)
                for gid in torch.unique(group_ids):
                    mask = group_ids == gid
                    gr = rewards[mask]
                    if gr.numel() <= 1:
                        continue
                    mean = gr.mean()
                    std = gr.std().clamp(min=1e-8)
                    advantages[mask] = (gr - mean) / std
                return advantages

            def compute_loss(self, inputs):
                signals, _advantages, old_log_probs = _algorithm_inputs(inputs)
                self.loss_calls += 1
                loss = signals.log_prob.mean()
                metrics = TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    update=PolicyUpdateStats(
                        approx_kl=float(old_log_probs.mean().item()),
                    ),
                )
                if emit_diagnostics:
                    call = float(self.loss_calls)
                    metrics.update.clip_fraction = call
                    metrics.update.active_clip_fraction = call / 10.0
                    metrics.weighted_kl_loss = call / 100.0
                    metrics.update.tis_clip_fraction = call / 20.0
                    metrics.update.rs_seq_masked_fraction = call / 40.0
                return loss, metrics

        class _Collector(CollectorControlFake):
            def __init__(self, reward_values: list[float]) -> None:
                self._reward_values = reward_values
                self._cursor = 0

            async def score_rollouts(self, pendings):

                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
                group_size = int(kwargs["group_size"])
                rewards = []
                for _ in range(group_size):
                    rewards.append(self._reward_values[self._cursor])
                    self._cursor += 1
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.tensor(rewards, dtype=torch.float32),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={
                        "reward_components": {
                            "observer": torch.tensor(rewards, dtype=torch.float32) + 10.0,
                        },
                    },
                    context={},
                )

        class _Evaluator(Evaluator):
            def __init__(self) -> None:
                self.calls = 0

            def evaluate(
                self,
                model,
                batch,
                timestep_idx,
                ref_model=None,
                signal_request=None,
            ):
                assert model.precision is DEFAULT_PRECISION
                assert model.precision.outer_autocast is False
                batch_size = batch.rewards.shape[0]
                self.calls += 1
                old = torch.full(
                    (batch_size,),
                    float(timestep_idx),
                    device=model.weight.device,
                )
                log_prob = old + self.calls / 1000.0 + model.weight.view(1) * 0.0
                return _trajectory_signals(batch, log_prob, timestep_idx)

        model = nn.Linear(1, 1, bias=False)
        _stamp_model_precision(model)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(rewards),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(
                    prompts_per_batch=1,
                    n_samples_per_prompt=2,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                ),
                timestep_fraction=1.0,
                ppo_epochs=ppo_epochs,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
            ),
            device="cpu",
        )
        return trainer

    def test_cea_step_advantages_independent_across_steps(self) -> None:
        """Second-step advantages should be normalized against the current group only,
        with no state leaking from previous steps."""
        import asyncio

        trainer = self._make_cea_trainer([0.0, 0.0, 0.0, 1.0])

        asyncio.run(trainer.step(["prompt-a"]))
        second_step = asyncio.run(trainer.step(["prompt-a"]))

        # Advantages are computed purely from current group — no stale history
        assert second_step.advantage_mean == pytest.approx(0.0, abs=1e-3)
        # Component metrics belong to the consumed second batch only. The first
        # step's observations must not accumulate on a shared reward object.
        assert second_step.reward_components == {"observer": pytest.approx(10.5)}

    def test_cea_metrics_propagate_approx_kl(self) -> None:
        """CEA aggregation should not silently drop approx_kl."""
        import asyncio

        trainer = self._make_cea_trainer([0.0, 1.0])
        metrics = asyncio.run(trainer.step(["prompt-a"]))

        assert metrics.update.approx_kl == pytest.approx(0.5)

    def test_cea_metrics_capture_the_complete_first_optimizer_update(self) -> None:
        """Initial replay covers every replay unit in the first optimizer update."""
        import asyncio
        from dataclasses import replace

        trainer = self._make_cea_trainer(
            [0.0, 1.0],
            emit_diagnostics=True,
            gradient_accumulation_steps=1,
        )
        batch = asyncio.run(trainer.collect_training_batch(["prompt-a"]))
        two_boundaries = replace(
            batch,
            batches=batch.batches * 2,
            advantages=batch.advantages * 2,
        )
        metrics = asyncio.run(trainer.train_on_rollout_batch(two_boundaries))

        # One optimizer target evaluates four sample chunks x two timesteps.
        assert metrics.update.clip_fraction == pytest.approx(4.5)
        assert metrics.initial_replay.clip_fraction == pytest.approx(4.5)
        assert metrics.update.active_clip_fraction == pytest.approx(0.45)
        assert metrics.initial_replay.active_clip_fraction == pytest.approx(0.45)
        assert metrics.logprob_mismatch.logprob_abs_diff_max == pytest.approx(
            0.008,
            abs=1e-6,
        )
        assert metrics.initial_replay.logprob_abs_diff_max == pytest.approx(
            0.008,
            abs=1e-6,
        )
        assert metrics.initial_replay.finite is True
        assert metrics.weighted_kl_loss == pytest.approx(0.045)
        assert metrics.update.tis_clip_fraction == pytest.approx(0.225)
        assert metrics.update.rs_seq_masked_fraction == pytest.approx(0.1125)

    def test_replay_metrics_follow_uneven_sample_chunk_weights(self) -> None:
        """An 8+2 split represents 80%+20% of the optimized group, not 50%+50%."""
        import torch

        from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
        from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.online.trainer import _ReplayMetrics, _training_sample_chunks

        batch = RolloutBatch(
            observations=torch.zeros(10, 1, 1),
            actions=torch.zeros(10, 1, 1),
            rewards=torch.zeros(10),
            group_ids=torch.zeros(10, dtype=torch.long),
        )
        chunks = _training_sample_chunks(batch, torch.ones(10), samples_per_chunk=8)
        assert [chunk.loss_weight for chunk in chunks] == pytest.approx([0.8, 0.2])

        aggregate = _ReplayMetrics()
        for value, chunk in zip((1.0, 9.0), chunks, strict=True):
            aggregate.add(
                TrainStepMetrics(
                    loss=value,
                    policy_loss=value,
                    update=PolicyUpdateStats(clip_fraction=value),
                    logprob_mismatch=LogprobMismatchStats(
                        logprob_abs_diff_mean=value,
                        logprob_abs_diff_max=value,
                    ),
                ),
                weight=chunk.loss_weight,
                capture_initial_replay=True,
            )

        initial_replay, initial_weight = aggregate.initial_replay_snapshot()
        metrics = aggregate.build(
            reward_mean=0.0,
            reward_std=0.0,
            reward_components={},
            advantage_mean=0.0,
            adv_saturation=0.0,
            adv_zero_rate=0.0,
            group_size=10.0,
            trained_prompt_num=1,
            phase_times={},
            initial_replay=initial_replay,
        )

        assert initial_weight == pytest.approx(1.0)
        assert metrics.loss == pytest.approx(2.6)
        assert metrics.update.clip_fraction == pytest.approx(2.6)
        assert metrics.logprob_mismatch.logprob_abs_diff_mean == pytest.approx(2.6)
        assert metrics.logprob_mismatch.logprob_abs_diff_max == pytest.approx(9.0)
        assert metrics.initial_replay.clip_fraction == pytest.approx(2.6)
        assert metrics.initial_replay.logprob_abs_diff_max == pytest.approx(9.0)

    def test_zero_advantage_samples_do_not_get_epsilon_gradient(self) -> None:
        """All-zero advantages should skip backward instead of inventing gradients."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.generation import GenerationRequest, GenerationSampleRow
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
        from vrl.trajectory import build_ar_discrete_trajectory

        class _Algorithm:
            class _Config:
                global_std = True
                eps = 1e-8
                adv_clip_max = 5.0
                kl_coef = 0.0

            config = _Config()

            def __init__(self) -> None:
                self.loss_calls = 0

            def compute_advantages_from_tensors(self, rewards, group_ids):
                del group_ids
                return torch.zeros_like(rewards)

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
                del signals, advantages, old_log_probs
                self.loss_calls += 1
                return torch.tensor(0.0, requires_grad=True), TrainStepMetrics()

        class _Collector(CollectorControlFake):
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
                group_size = int(kwargs["group_size"])
                prompts = list(prompts)
                request = GenerationRequest(
                    request_id="zero-adv",
                    family="janus_pro",
                    task="ar_t2i",
                    inputs=prompts,
                    samples_per_prompt=group_size,
                )
                sample_rows = [
                    GenerationSampleRow(
                        prompt_index=index // group_size,
                        sample_index=index % group_size,
                        prompt=prompts[index // group_size],
                        group_id=f"group-{index // group_size}",
                        sample_id=f"sample-{index}",
                        trajectory_id=f"trajectory-{index}",
                        seed=None,
                    )
                    for index in range(len(prompts) * group_size)
                ]
                batch_size = len(sample_rows)
                token_ids = torch.arange(batch_size * 2).view(batch_size, 2)
                trajectory = build_ar_discrete_trajectory(
                    request=request,
                    sample_rows=sample_rows,
                    token_ids=token_ids,
                    token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
                    token_mask=torch.ones_like(token_ids, dtype=torch.float32),
                    prompt_input_ids=torch.ones(len(sample_rows), 3, dtype=torch.long),
                    prompt_attention_mask=torch.ones(len(sample_rows), 3, dtype=torch.long),
                    uncond_input_ids=torch.zeros(len(sample_rows), 3, dtype=torch.long),
                    uncond_attention_mask=torch.ones(len(sample_rows), 3, dtype=torch.long),
                    context={"model_family": "janus_pro"},
                )
                return RolloutBatch(
                    observations=torch.zeros(batch_size, 1, 1),
                    actions=torch.zeros(batch_size, 1, 1),
                    rewards=torch.ones(batch_size, dtype=torch.float32),
                    group_ids=torch.zeros(batch_size, dtype=torch.long),
                    context={},
                    trajectory=trajectory,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del model, kw
                return _trajectory_signals(batch, torch.zeros(1), timestep_idx)

        algorithm = _Algorithm()
        model = nn.Linear(1, 1, bias=False)
        _stamp_model_precision(model)
        trainer = OnlineTrainer(
            algorithm=algorithm,
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
                timestep_fraction=1.0,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                drop_zero_advantage=True,
            ),
            device="cpu",
        )

        metrics = asyncio.run(trainer.step(["prompt-a"]))

        assert algorithm.loss_calls == 0
        assert trainer.state.step == 1
        assert trainer.state.global_step == 0
        assert metrics.grad_norm == 0.0
        assert metrics.group_size == 2.0
        assert metrics.trained_prompt_num == 1
