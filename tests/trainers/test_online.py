"""Tests for vrl.trainers.online (OnlineTrainer CEA pipeline)."""

from __future__ import annotations

import pytest

from vrl.rollouts.collector.base import Collector
from vrl.rollouts.evaluators.base import Evaluator


class TestOnlineTrainerCeaRegressions:
    def _make_cea_trainer(self, rewards: list[float]):
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

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

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean()
                metrics = TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=float(old_log_probs.mean().item()),
                )
                return loss, metrics

        class _Collector(Collector):
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
                    extras={
                        "log_probs": torch.tensor(
                            [[0.0, 1.0]] * group_size,
                            dtype=torch.float32,
                        ),
                    },
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
                return SignalBatch(log_prob=log_prob)

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

    def test_cea_step_forwards_prompt_example_kwargs(self) -> None:
        """PromptExample fields should be forwarded as kwargs to collector.collect()."""
        import asyncio

        import torch

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.data import PromptExample
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        captured_kwargs: list[dict] = []

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

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean()
                return loss, TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=0.0,
                )

        class _CapturingCollector(Collector):
            async def collect(self, prompts, **kwargs):
                captured_kwargs.append(dict(kwargs))
                group_size = int(kwargs.get("group_size", 1))
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.ones(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={
                        "log_probs": torch.zeros(group_size, 2, dtype=torch.float32),
                    },
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                batch_size = batch.rewards.shape[0]
                return SignalBatch(log_prob=model.weight.view(1).expand(batch_size))

        import torch.nn as nn

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_CapturingCollector(),
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

        example = PromptExample(
            prompt="sign says HELLO",
            target_text="HELLO",
            task_type="text_to_video",
            metadata={"difficulty": "easy"},
        )
        asyncio.run(trainer.step([example]))

        # Group-batched collect: one call per prompt, carrying group_size=2.
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["group_size"] == 2
        assert kw["target_text"] == "HELLO"
        assert kw["task_type"] == "text_to_video"
        assert kw["sample_metadata"]["difficulty"] == "easy"

    def test_cea_batches_plain_prompts_for_rollout_but_splits_training(self) -> None:
        """Plain prompts should collect together, then train as group-local batches."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        collect_calls: list[list[str]] = []
        evaluate_batch_sizes: list[int] = []
        evaluate_group_ids: list[list[int]] = []

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
                    advantages[mask] = gr - gr.mean()
                return advantages

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean() + advantages.mean() * 0.0
                return loss, TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=float(old_log_probs.mean().item()),
                )

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                prompts = list(prompts)
                collect_calls.append(prompts)
                group_size = int(kwargs.get("group_size", 1))
                batch_size = len(prompts) * group_size
                group_ids = torch.tensor(
                    [prompt_idx for prompt_idx in range(len(prompts)) for _ in range(group_size)],
                    dtype=torch.long,
                )
                rewards = torch.tensor(
                    [float(i % group_size) for i in range(batch_size)],
                    dtype=torch.float32,
                )
                return RolloutBatch(
                    observations=torch.zeros(batch_size, 2, 1),
                    actions=torch.zeros(batch_size, 2, 1),
                    rewards=rewards,
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=group_ids,
                    extras={"log_probs": torch.zeros(batch_size, 2)},
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                evaluate_batch_sizes.append(int(batch.rewards.shape[0]))
                evaluate_group_ids.append(
                    [int(x) for x in batch.group_ids.detach().cpu().tolist()]
                )
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
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

        asyncio.run(trainer.step(["prompt-a", "prompt-b"]))

        assert collect_calls == [["prompt-a", "prompt-b"]]
        assert evaluate_batch_sizes == [2, 2, 2, 2]
        assert evaluate_group_ids == [[0, 0], [0, 0], [1, 1], [1, 1]]

    def test_initial_rollout_weight_sync_happens_before_collect(self) -> None:
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        collect_seen_sync_counts: list[int] = []

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

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean() + advantages.mean() * 0.0
                return loss, TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=float(old_log_probs.mean().item()),
                )

        class _Syncer:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def push(self, state_dict):
                self.calls.append(dict(state_dict))

            async def pull(self):
                return dict(self.calls[-1])

        syncer = _Syncer()

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                collect_seen_sync_counts.append(len(syncer.calls))
                group_size = int(kwargs.get("group_size", 1))
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.arange(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={"log_probs": torch.zeros(group_size, 2)},
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            weight_syncer=syncer,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a"]))

        assert collect_seen_sync_counts == [1]
        assert len(syncer.calls) == 2

    def test_first_step_debug_writes_training_debug_jsonl(self, tmp_path) -> None:
        import asyncio
        import json

        import pytest
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

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

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                del advantages, old_log_probs
                loss = signals.log_prob.mean()
                return loss, TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                )

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                assert kwargs["runtime_debug"] is True
                group_size = int(kwargs.get("group_size", 1))
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.arange(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={"log_probs": torch.zeros(group_size, 2)},
                    context={
                        "runtime_debug": {
                            "ray_chunks": [
                                {
                                    "worker_id": "rollout-0",
                                    "policy_version": 1,
                                },
                            ],
                        },
                    },
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(first_step=True),
                n=2,
                bf16=False,
                mixed_precision="no",
                output_dir=str(tmp_path),
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a"]))

        debug_path = tmp_path / "training_debug.jsonl"
        records = [json.loads(line) for line in debug_path.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["event"] == "first_step_logprob_parity"
        assert record["mixed_precision"] == "no"
        assert record["abs_diff"]["mean"] == pytest.approx(1.0)
        assert record["ratio"]["mean"] == pytest.approx(torch.exp(torch.tensor(1.0)).item())
        assert record["driver_trainable_before_step"]["tensor_count"] == 1
        assert record["driver_trainable_after_step"]["tensor_count"] == 1
        assert record["runtime_debug"]["ray_chunks"][0]["worker_id"] == "rollout-0"

    def test_algorithm_diagnostic_tensors_are_cleared_after_backward(self) -> None:
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

            config = _Config()

            def __init__(self) -> None:
                self._last_policy_loss_tensor = None
                self._last_kl_term_tensor = None

            def compute_advantages_from_tensors(self, rewards, group_ids):
                del group_ids
                return rewards - rewards.mean()

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                del old_log_probs
                policy_loss = signals.log_prob.mean() + advantages.mean() * 0.0
                self._last_policy_loss_tensor = policy_loss
                self._last_kl_term_tensor = policy_loss * 0.0
                return policy_loss, TrainStepMetrics(
                    loss=policy_loss.item(),
                    policy_loss=policy_loss.item(),
                )

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                group_size = int(kwargs.get("group_size", 1))
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.arange(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={"log_probs": torch.zeros(group_size, 2)},
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

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
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a"]))

        assert algorithm._last_policy_loss_tensor is None
        assert algorithm._last_kl_term_tensor is None

    def test_gradient_accumulation_steps_control_optimizer_updates(self) -> None:
        """Positive gradient_accumulation_steps splits one rollout step into multiple updates."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

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

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                del advantages, old_log_probs
                loss = signals.log_prob.mean()
                return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                prompts = list(prompts)
                group_size = int(kwargs.get("group_size", 1))
                batch_size = len(prompts) * group_size
                group_ids = torch.tensor(
                    [prompt_idx for prompt_idx in range(len(prompts)) for _ in range(group_size)],
                    dtype=torch.long,
                )
                return RolloutBatch(
                    observations=torch.zeros(batch_size, 1, 1),
                    actions=torch.zeros(batch_size, 1, 1),
                    rewards=torch.tensor(
                        [float(i % group_size) for i in range(batch_size)],
                        dtype=torch.float32,
                    ),
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=group_ids,
                    extras={"log_probs": torch.zeros(batch_size, 1)},
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
                gradient_accumulation_steps=2,
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a", "prompt-b", "prompt-c", "prompt-d"]))

        assert trainer.state.step == 1
        assert trainer.state.global_step == 2

    def test_flow_grpo_loss_scaling_includes_timesteps(self) -> None:
        """Flow-GRPO accumulation scales loss by microbatches * train timesteps."""
        import asyncio

        import pytest
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        recorded_grads: list[float] = []

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

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                del advantages, old_log_probs
                loss = signals.log_prob.mean()
                return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                prompts = list(prompts)
                group_size = int(kwargs.get("group_size", 1))
                batch_size = len(prompts) * group_size
                group_ids = torch.tensor(
                    [prompt_idx for prompt_idx in range(len(prompts)) for _ in range(group_size)],
                    dtype=torch.long,
                )
                return RolloutBatch(
                    observations=torch.zeros(batch_size, 3, 1),
                    actions=torch.zeros(batch_size, 3, 1),
                    rewards=torch.tensor(
                        [float(i % group_size) for i in range(batch_size)],
                        dtype=torch.float32,
                    ),
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=group_ids,
                    extras={"log_probs": torch.zeros(batch_size, 3)},
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.1, weight_decay=0.0),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
                gradient_accumulation_steps=2,
            ),
            device="cpu",
        )

        original_step = trainer._clip_and_step

        def _recording_step(optimizer):
            assert model.weight.grad is not None
            recorded_grads.append(float(model.weight.grad.detach().item()))
            return original_step(optimizer)

        trainer._clip_and_step = _recording_step  # type: ignore[method-assign]

        asyncio.run(trainer.step(["prompt-a", "prompt-b", "prompt-c", "prompt-d"]))

        # Each update accumulates 2 rollout microbatches * 3 timesteps.
        # Without timestep-aware scaling these gradients would be 3.0.
        assert recorded_grads == pytest.approx([1.0, 1.0])
        assert trainer.state.global_step == 2

    def test_flow_grpo_zero_advantage_padding_rebatches_evenly(self) -> None:
        """Flow-style stat-tracker filtering pads zero-adv rows before rebatching."""
        import asyncio

        import numpy as np
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        seen_batch_sizes: list[int] = []

        class _Algorithm:
            class _Config:
                global_std = True
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

            config = _Config()

            def __init__(self) -> None:
                self.advantages_seen: list[float] = []

            def compute_advantages_from_tensors(self, rewards, group_ids):
                del rewards, group_ids
                raise AssertionError("stat_tracker should own advantages")

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                del old_log_probs
                self.advantages_seen.extend(
                    float(x) for x in advantages.detach().cpu().reshape(-1).tolist()
                )
                loss = signals.log_prob.mean() + advantages.mean() * 0.0
                return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

        class _SparseTracker:
            def update(self, prompts, rewards):
                del prompts
                out = np.zeros_like(rewards.detach().cpu().numpy())
                out[0] = 1.0
                return out

            def get_stats(self):
                return 2.0, 3

            def clear(self):
                pass

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                prompts = list(prompts)
                group_size = int(kwargs.get("group_size", 1))
                batch_size = len(prompts) * group_size
                group_ids = torch.tensor(
                    [i for i in range(len(prompts)) for _ in range(group_size)],
                    dtype=torch.long,
                )
                return RolloutBatch(
                    observations=torch.zeros(batch_size, 1, 1),
                    actions=torch.zeros(batch_size, 1, 1),
                    rewards=torch.arange(batch_size, dtype=torch.float32),
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=group_ids,
                    extras={"log_probs": torch.zeros(batch_size, 1)},
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                seen_batch_sizes.append(int(batch.rewards.shape[0]))
                return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))

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
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
                gradient_accumulation_steps=2,
            ),
            device="cpu",
            stat_tracker=_SparseTracker(),
        )

        asyncio.run(trainer.step(["prompt-a", "prompt-b", "prompt-c"]))

        assert seen_batch_sizes == [1, 1, 1]
        assert len(algorithm.advantages_seen) == 3
        assert algorithm.advantages_seen.count(1.0) == 1
        assert algorithm.advantages_seen.count(0.0) == 2
        assert trainer.state.global_step == 2

    def test_zero_advantage_stat_tracker_samples_do_not_get_epsilon_gradient(self) -> None:
        """All-zero stat-tracker advantages should skip backward instead of inventing gradients."""
        import asyncio

        import numpy as np
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

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
                del rewards, group_ids
                raise AssertionError("stat_tracker should own advantages")

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                del signals, advantages, old_log_probs
                self.loss_calls += 1
                return torch.tensor(0.0, requires_grad=True), TrainStepMetrics()

        class _ZeroTracker:
            def update(self, prompts, rewards):
                del prompts
                return np.zeros_like(rewards.detach().cpu().numpy())

            def get_stats(self):
                return 2.0, 1

            def clear(self):
                pass

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                group_size = int(kwargs.get("group_size", 1))
                return RolloutBatch(
                    observations=torch.zeros(group_size, 1, 1),
                    actions=torch.zeros(group_size, 1, 1),
                    rewards=torch.ones(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={"log_probs": torch.zeros(group_size, 1)},
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del model, batch, timestep_idx, kw
                return SignalBatch(log_prob=torch.zeros(1))

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
            ),
            device="cpu",
            stat_tracker=_ZeroTracker(),
        )

        metrics = asyncio.run(trainer.step(["prompt-a"]))

        assert algorithm.loss_calls == 0
        assert trainer.state.step == 1
        assert trainer.state.global_step == 0
        assert metrics.grad_norm == 0.0
        assert metrics.group_size == 2.0
        assert metrics.trained_prompt_num == 1


class TestOnlineTrainerResumeState:
    def test_load_state_dict_initializes_and_restores_optimizer_state(self) -> None:
        import torch

        source = _make_resume_trainer()
        optimizer = source._ensure_optimizer()
        loss = source.model(torch.ones(1, 1)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        source.state.step = 3
        source.state.global_step = 5
        state = source.state_dict()

        restored = _make_resume_trainer()
        restored.load_state_dict(state, strict=True)

        assert restored.state.step == 3
        assert restored.state.global_step == 5
        assert restored._optimizer is not None
        assert _adam_exp_avg_values(restored._optimizer) == pytest.approx(
            _adam_exp_avg_values(optimizer),
        )

    def test_load_state_dict_initializes_and_restores_ema_state(self) -> None:
        import torch

        source = _make_resume_trainer(ema=True)
        ema = source._ensure_ema()
        assert ema is not None
        ema.ema_parameters[0].fill_(7.0)
        state = source.state_dict()

        restored = _make_resume_trainer(ema=True)
        restored.load_state_dict(state, strict=True)

        assert restored._ema is not None
        assert torch.equal(
            restored._ema.ema_parameters[0],
            torch.full_like(restored._ema.ema_parameters[0], 7.0),
        )

    def test_load_state_dict_rejects_ema_state_when_ema_is_disabled(self) -> None:
        source = _make_resume_trainer(ema=True)
        source._ensure_ema()
        state = source.state_dict()

        restored = _make_resume_trainer(ema=False)
        with pytest.raises(ValueError, match="EMA state"):
            restored.load_state_dict(state, strict=True)

    def test_load_state_dict_resets_rollout_weight_initialization(self) -> None:
        trainer = _make_resume_trainer()
        trainer._rollout_weights_initialized = True

        trainer.load_state_dict({"step": 9, "global_step": 9}, strict=True)

        assert trainer._rollout_weights_initialized is False

    def test_resume_pushes_restored_driver_weights_before_next_collect(self) -> None:
        import asyncio

        syncer = _Syncer()
        collect_seen_sync_counts: list[int] = []
        trainer = _make_resume_trainer(
            weight_syncer=syncer,
            collector=_SyncCountingCollector(syncer, collect_seen_sync_counts),
        )
        trainer._rollout_weights_initialized = True
        trainer.load_state_dict({"step": 4, "global_step": 4}, strict=True)

        asyncio.run(trainer.step(["prompt-a"]))

        assert collect_seen_sync_counts == [1]


class _ResumeAlgorithm:
    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        init_kl_coef = 0.0

    config = _Config()

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def compute_signal_loss(self, signals, advantages, old_log_probs):
        from vrl.algorithms.types import TrainStepMetrics

        loss = signals.log_prob.mean() + advantages.mean() * 0.0
        return loss, TrainStepMetrics(
            loss=loss.item(),
            policy_loss=loss.item(),
            approx_kl=float(old_log_probs.mean().item()),
        )


class _ResumeCollector:
    async def collect(self, prompts, **kwargs):
        import torch

        from vrl.rollouts.batch import RolloutBatch

        group_size = int(kwargs.get("group_size", 1))
        return RolloutBatch(
            observations=torch.zeros(group_size, 2, 1),
            actions=torch.zeros(group_size, 2, 1),
            rewards=torch.arange(group_size, dtype=torch.float32),
            dones=torch.ones(group_size, dtype=torch.bool),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            extras={"log_probs": torch.zeros(group_size, 2)},
            prompts=list(prompts) * group_size,
        )


class _SyncCountingCollector(_ResumeCollector):
    def __init__(self, syncer: _Syncer, seen_counts: list[int]) -> None:
        self.syncer = syncer
        self.seen_counts = seen_counts

    async def collect(self, prompts, **kwargs):
        self.seen_counts.append(len(self.syncer.calls))
        return await super().collect(prompts, **kwargs)


class _ResumeEvaluator:
    def evaluate(self, model, batch, timestep_idx, **kw):
        del timestep_idx, kw

        from vrl.rollouts.evaluators.types import SignalBatch

        return SignalBatch(log_prob=model.weight.view(1).expand(batch.rewards.shape[0]))


class _Syncer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def push(self, state_dict):
        self.calls.append(dict(state_dict))

    async def pull(self):
        return dict(self.calls[-1])


def _make_resume_trainer(
    *,
    ema: bool = False,
    weight_syncer=None,
    collector=None,
):
    import torch
    import torch.nn as nn

    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return OnlineTrainer(
        algorithm=_ResumeAlgorithm(),
        collector=collector or _ResumeCollector(),
        evaluator=_ResumeEvaluator(),
        model=model,
        weight_syncer=weight_syncer,
        config=TrainerConfig(
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(enable=ema),
            debug=DebugConfig(),
            n=2,
            bf16=False,
        ),
        device="cpu",
    )


def _adam_exp_avg_values(optimizer) -> list[float]:
    values: list[float] = []
    for slot in optimizer.state.values():
        exp_avg = slot.get("exp_avg")
        if exp_avg is not None:
            values.extend(float(v) for v in exp_avg.reshape(-1).detach().cpu().tolist())
    return values
