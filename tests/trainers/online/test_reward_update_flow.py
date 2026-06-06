"""OnlineTrainer rollout consume + update loop: prompt-kwarg forwarding, batching, gradient accumulation, loss scaling, zero-advantage rebatching, and batch-op field preservation."""

from __future__ import annotations

from tests.trainers.online._helpers import _algorithm_inputs, _trajectory_signals
from vrl.rollouts.evaluators.base import Evaluator


class TestRewardUpdateFlow:
    """Groups tests for reward update flow."""
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

        class _CapturingCollector:
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

        class _Collector:
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

        class _Collector:
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

        class _Collector:
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
        """Flow-style zero-advantage filtering pads rows before rebatching."""
        import asyncio

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
                del group_ids
                out = torch.zeros_like(rewards)
                out[0] = 1.0
                return out

            def compute_loss(self, inputs):
                signals, advantages, old_log_probs = _algorithm_inputs(inputs)
                del old_log_probs
                self.advantages_seen.extend(
                    float(x) for x in advantages.detach().cpu().reshape(-1).tolist()
                )
                loss = signals.log_prob.mean() + advantages.mean() * 0.0
                return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

        class _Collector:
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
                drop_zero_advantage=True,
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a", "prompt-b", "prompt-c"]))

        assert seen_batch_sizes == [1, 1, 1]
        assert len(algorithm.advantages_seen) == 3
        assert algorithm.advantages_seen.count(1.0) == 1
        assert algorithm.advantages_seen.count(0.0) == 2
        assert trainer.state.global_step == 2


def test_select_move_and_remap_preserve_rollout_trajectory_fields() -> None:
    """Checks select move and remap preserve rollout trajectory fields."""
    import torch

    from vrl.generation import GenerationRequest, GenerationSampleRow
    from vrl.rollouts.batch import RolloutBatch
    from vrl.rollouts.batch.ops import (
        move_training_batch_to_device,
        remap_group_ids_,
        select_batch,
    )
    from vrl.trajectory import build_ar_discrete_trajectory, build_training_view

    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["a", "b"],
        samples_per_prompt=2,
    )
    sample_rows = [
        GenerationSampleRow(
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
        sample_rows=sample_rows,
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

    selected = select_batch(batch, torch.tensor([True, False, True, False]))

    assert selected.trajectory is not None
    assert selected.training_view == batch.training_view
    assert selected.trajectory.axes["sample"].length == 2
    assert torch.equal(selected.trajectory.group_ids, torch.tensor([0, 1]))
    assert torch.equal(
        selected.trajectory.segments["image_tokens"].tensors["token_ids"].value,
        torch.tensor([[0, 1], [4, 5]]),
    )

    moved = move_training_batch_to_device(selected, torch.device("cpu"))
    assert moved.trajectory is not None
    assert moved.trajectory.group_ids.device.type == "cpu"

    remap_group_ids_(moved, [10, 11])
    assert torch.equal(moved.group_ids, torch.tensor([10, 11]))
    assert moved.trajectory is not None
    assert torch.equal(moved.trajectory.group_ids, torch.tensor([10, 11]))
