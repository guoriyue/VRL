"""Tests for vrl.trainers.online (OnlineTrainer CEA pipeline)."""

from __future__ import annotations

import pytest

from vrl.rollouts.collector.base import Collector
from vrl.rollouts.evaluators.base import Evaluator


def _algorithm_inputs(inputs):
    if inputs.signals is None:
        raise AssertionError("test algorithm requires trajectory signals")
    signals = inputs.signals.primary
    return signals, inputs.advantages, signals.old_log_prob


def _trajectory_signals(batch, log_prob, timestep_idx: int = 0):
    import torch

    from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

    old_log_prob = torch.full_like(log_prob, float(timestep_idx))
    mask = torch.ones_like(log_prob)
    axes = ("sample",) if int(getattr(log_prob, "ndim", 1)) <= 1 else ("sample", "timestep")
    return TrajectorySignalBatch(
        segments={
            "default": SegmentSignal(
                name="default",
                segment="default",
                axis="timestep",
                axes=axes,
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=mask,
            ),
        },
        group_ids=batch.group_ids,
        primary_segment="default",
    )


class TestOnlineTrainerCeaRegressions:
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

    def test_cea_step_forwards_prompt_example_kwargs(self) -> None:
        """PromptExample fields should be forwarded as kwargs to collector.collect()."""
        import asyncio

        import torch

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.data import PromptExample
        from vrl.trainers.online import OnlineTrainer

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

            def compute_loss(self, inputs):
                signals, _advantages, _old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                batch_size = batch.rewards.shape[0]
                return _trajectory_signals(batch, model.weight.view(1).expand(batch_size), timestep_idx)

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
            reference_image="/tmp/reference.png",
            task_type="text_to_video",
            metadata={"difficulty": "easy"},
        )
        asyncio.run(trainer.step([example]))

        # Group-batched collect: one call per prompt, carrying group_size=2.
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["group_size"] == 2
        assert kw["target_text"] == "HELLO"
        assert kw["reference_image"] == "/tmp/reference.png"
        assert kw["task_type"] == "text_to_video"
        assert kw["sample_metadata"]["difficulty"] == "easy"

    def test_cea_batches_plain_prompts_for_rollout_but_splits_training(self) -> None:
        """Plain prompts should collect together, then train as group-local batches."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

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

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                evaluate_batch_sizes.append(int(batch.rewards.shape[0]))
                evaluate_group_ids.append(
                    [int(x) for x in batch.group_ids.detach().cpu().tolist()]
                )
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

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
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

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

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            weight_syncer=syncer,
            sync_state_getter=lambda: {"linear.weight": model.weight.detach().clone()},
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

    def test_weight_sync_requires_explicit_trainable_state_getter(self) -> None:
        import pytest
        import torch.nn as nn

        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

            config = _Config()

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                del prompts, kwargs
                raise AssertionError("constructor guard should run before collect")

        class _Evaluator(Evaluator):
            pass

        class _Syncer:
            async def push(self, state_dict):
                del state_dict

            async def pull(self):
                return {}

        with pytest.raises(ValueError, match="trainable-state getter"):
            OnlineTrainer(
                algorithm=_Algorithm(),
                collector=_Collector(),
                evaluator=_Evaluator(),
                model=nn.Linear(1, 1),
                weight_syncer=_Syncer(),
                config=TrainerConfig(
                    optim=OptimConfig(lr=0.01),
                    ema=EMAConfig(),
                    debug=DebugConfig(),
                    n=2,
                    bf16=False,
                ),
                device="cpu",
            )

    def test_first_step_debug_writes_training_debug_jsonl(self, tmp_path) -> None:
        import asyncio
        import json

        import pytest
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
                del group_ids
                return rewards - rewards.mean()

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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

        grad_enabled: list[bool] = []

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                grad_enabled.append(torch.is_grad_enabled())
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

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
        assert grad_enabled[0] is False
        assert any(grad_enabled[1:])

    def test_algorithm_diagnostic_tensors_are_cleared_after_backward(self) -> None:
        import asyncio

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

            def __init__(self) -> None:
                self._last_policy_loss_tensor = None
                self._last_kl_term_tensor = None

            def compute_advantages_from_tensors(self, rewards, group_ids):
                del group_ids
                return rewards - rewards.mean()

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

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
                del group_ids
                return rewards - rewards.mean()

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

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
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

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

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

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
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

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

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                seen_batch_sizes.append(int(batch.rewards.shape[0]))
                return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)

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
        from vrl.engine import GenerationRequest, GenerationSampleSpec
        from vrl.engine.trajectory import build_ar_discrete_trajectory, build_training_view
        from vrl.rollouts.batch import RolloutBatch
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

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

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
                prompts = list(prompts)
                request = GenerationRequest(
                    request_id="zero-adv",
                    family="janus_pro",
                    task="ar_t2i",
                    prompts=prompts,
                    samples_per_prompt=group_size,
                )
                sample_specs = [
                    GenerationSampleSpec(
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
                batch_size = len(sample_specs)
                token_ids = torch.arange(batch_size * 2).view(batch_size, 2)
                trajectory = build_ar_discrete_trajectory(
                    request=request,
                    sample_specs=sample_specs,
                    token_ids=token_ids,
                    token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
                    token_mask=torch.ones_like(token_ids, dtype=torch.float32),
                    prompt_input_ids=torch.ones(len(sample_specs), 3, dtype=torch.long),
                    prompt_attention_mask=torch.ones(len(sample_specs), 3, dtype=torch.long),
                    uncond_input_ids=torch.zeros(len(sample_specs), 3, dtype=torch.long),
                    uncond_attention_mask=torch.ones(len(sample_specs), 3, dtype=torch.long),
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

    def compute_loss(self, inputs):
        signals, advantages, old_log_probs = _algorithm_inputs(inputs)
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
        del kw

        return _trajectory_signals(batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx)


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

    from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
    from vrl.trainers.online import OnlineTrainer

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return OnlineTrainer(
        algorithm=_ResumeAlgorithm(),
        collector=collector or _ResumeCollector(),
        evaluator=_ResumeEvaluator(),
        model=model,
        weight_syncer=weight_syncer,
        sync_state_getter=(
            lambda: {"linear.weight": model.weight.detach().clone()}
            if weight_syncer is not None
            else None
        ),
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


def test_select_move_and_remap_preserve_rollout_trajectory_fields() -> None:
    import torch

    from vrl.engine import GenerationRequest, GenerationSampleSpec
    from vrl.engine.trajectory import build_ar_discrete_trajectory, build_training_view
    from vrl.rollouts.batch import RolloutBatch
    from vrl.trainers.online import (
        _move_training_batch_to_device,
        _remap_group_ids_,
        _select_batch,
    )

    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["a", "b"],
        samples_per_prompt=2,
    )
    sample_specs = [
        GenerationSampleSpec(
            prompt_index=index // 2,
            sample_index=index % 2,
            prompt=request.prompts[index // 2],
            prompt_id=f"p{index // 2}",
            group_id=f"g{index // 2}",
            sample_id=f"s{index}",
            trajectory_id=f"t{index}",
            seed=None,
        )
        for index in range(4)
    ]
    token_ids = torch.arange(8).view(4, 2)
    trajectory = build_ar_discrete_trajectory(
        request=request,
        sample_specs=sample_specs,
        token_ids=token_ids,
        token_log_probs=torch.zeros(4, 2),
        token_mask=torch.ones(4, 2),
        prompt_input_ids=torch.ones(4, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(4, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(4, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(4, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )
    batch = RolloutBatch(
        observations=torch.ones(4, 1, 3, dtype=torch.long),
        actions=token_ids,
        rewards=torch.arange(4, dtype=torch.float32),
        dones=torch.ones(4, dtype=torch.bool),
        group_ids=torch.tensor([0, 0, 1, 1]),
        trajectory=trajectory,
        training_view=build_training_view(trajectory),
    )

    selected = _select_batch(batch, torch.tensor([True, False, True, False]))

    assert selected.trajectory is not None
    assert selected.training_view == batch.training_view
    assert selected.trajectory.axes["sample"].length == 2
    assert torch.equal(selected.trajectory.group_ids, torch.tensor([0, 1]))
    assert torch.equal(
        selected.trajectory.segments["image_tokens"].tensors["token_ids"].value,
        torch.tensor([[0, 1], [4, 5]]),
    )

    moved = _move_training_batch_to_device(selected, torch.device("cpu"))
    assert moved.trajectory is not None
    assert moved.trajectory.group_ids.device.type == "cpu"

    _remap_group_ids_(moved, [10, 11])
    assert torch.equal(moved.group_ids, torch.tensor([10, 11]))
    assert moved.trajectory is not None
    assert torch.equal(moved.trajectory.group_ids, torch.tensor([10, 11]))
