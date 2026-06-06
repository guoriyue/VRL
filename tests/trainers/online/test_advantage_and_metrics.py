"""OnlineTrainer advantage normalization, KL propagation, and zero-advantage gradient behavior."""

from __future__ import annotations

import pytest

from tests.trainers.online._helpers import _algorithm_inputs, _trajectory_signals
from vrl.rollouts.evaluators.base import Evaluator


class TestAdvantageAndMetrics:
    """Groups tests for advantage and metrics."""
    def _make_cea_trainer(self, rewards: list[float]):
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

            config = _Config()

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
                loss = signals.log_prob.mean()
                metrics = TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=float(old_log_probs.mean().item()),
                )
                return loss, metrics

        class _Collector:
            def __init__(self, reward_values: list[float]) -> None:
                self._reward_values = reward_values
                self._cursor = 0

            async def collect(self, prompts, **kwargs):
                group_size = int(kwargs.get("group_size", 1))
                rewards = []
                for _ in range(group_size):
                    rewards.append(self._reward_values[self._cursor])
                    self._cursor += 1
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.tensor(rewards, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(
                self,
                model,
                batch,
                timestep_idx,
                ref_model=None,
                signal_request=None,
            ):
                batch_size = batch.rewards.shape[0]
                log_prob = model.weight.view(1).expand(batch_size)
                return _trajectory_signals(batch, log_prob, timestep_idx)

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(rewards),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
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

    def test_cea_metrics_propagate_approx_kl(self) -> None:
        """CEA aggregation should not silently drop approx_kl."""
        import asyncio

        trainer = self._make_cea_trainer([0.0, 1.0])
        metrics = asyncio.run(trainer.step(["prompt-a"]))

        assert metrics.approx_kl == pytest.approx(0.5)

    def test_zero_advantage_samples_do_not_get_epsilon_gradient(self) -> None:
        """All-zero advantages should skip backward instead of inventing gradients."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.generation import GenerationRequest, GenerationSampleRow
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer
        from vrl.trajectory import build_ar_discrete_trajectory, build_training_view

        class _Algorithm:
            class _Config:
                global_std = True
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

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

        class _Collector:
            async def collect(self, prompts, **kwargs):
                group_size = int(kwargs.get("group_size", 1))
                prompts = list(prompts)
                request = GenerationRequest(
                    request_id="zero-adv",
                    family="janus_pro",
                    task="ar_t2i",
                    prompts=prompts,
                    samples_per_prompt=group_size,
                )
                sample_rows = [
                    GenerationSampleRow(
                        prompt_index=index // group_size,
                        sample_index=index % group_size,
                        prompt=prompts[index // group_size],
                        prompt_id=f"prompt-{index // group_size}",
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
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=torch.zeros(batch_size, dtype=torch.long),
                    prompts=prompts * group_size,
                    trajectory=trajectory,
                    training_view=build_training_view(trajectory),
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del model, kw
                return _trajectory_signals(batch, torch.zeros(1), timestep_idx)

        algorithm = _Algorithm()
        model = nn.Linear(1, 1, bias=False)
        trainer = OnlineTrainer(
            algorithm=algorithm,
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
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

