"""OnlineTrainer rollout consume + update loop: prompt-kwarg forwarding, batching, gradient accumulation, loss scaling, zero-advantage rebatching, and batch-op field preservation."""

from __future__ import annotations

from tests.trainers.online._collector_control import CollectorControlFake
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
        captured_inputs: list = []

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                kl_coef = 0.0

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

        class _CapturingCollector(CollectorControlFake):
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, inputs, **kwargs):
                captured_inputs.extend(inputs)
                captured_kwargs.append(dict(kwargs))
                group_size = int(kwargs["group_size"])
                prompts = [item.prompt for item in inputs]
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.ones(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    prompts=prompts * group_size,
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                batch_size = batch.rewards.shape[0]
                return _trajectory_signals(
                    batch, model.weight.view(1).expand(batch_size), timestep_idx
                )

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
                prompts_per_batch=1,
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n_samples_per_prompt=2,
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
        # Conditioning rides the typed GenerationInput; reward targets ride the
        # group metadata dict (PromptExample owns that mapping).
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["group_size"] == 2
        assert kw["metadata"]["target_text"] == "HELLO"
        assert kw["metadata"]["difficulty"] == "easy"
        assert len(captured_inputs) == 1
        assert captured_inputs[0].reference_image == "/tmp/reference.png"
        assert captured_inputs[0].task_type == "text_to_video"

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
                kl_coef = 0.0

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

        class _Collector(CollectorControlFake):
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, inputs, **kwargs):
                prompts = [getattr(item, "prompt", item) for item in inputs]
                collect_calls.append(prompts)
                group_size = int(kwargs["group_size"])
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
                return _trajectory_signals(
                    batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx
                )

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                prompts_per_batch=1,
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n_samples_per_prompt=2,
                replay_samples_per_chunk=0,
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a", "prompt-b"]))

        assert collect_calls == [["prompt-a", "prompt-b"]]
        assert evaluate_batch_sizes == [2, 2, 2, 2]
        assert evaluate_group_ids == [[0, 0], [0, 0], [1, 1], [1, 1]]

    def test_streaming_accumulation_runs_one_optimizer_step(self) -> None:
        """gradient_accumulation_steps>0 streams microbatches into ONE optimizer update."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.scripts.common.online import _run_streaming_optimizer_update
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

        collect_calls: list[list[str]] = []
        after_step_calls: list[int] = []

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

            def after_optimizer_step(self, model, global_step):
                del model
                after_step_calls.append(global_step)

        class _Collector(CollectorControlFake):
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, inputs, **kwargs):
                prompts = [getattr(item, "prompt", item) for item in inputs]
                collect_calls.append(list(prompts))
                group_size = int(kwargs["group_size"])
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
                return _trajectory_signals(
                    batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx
                )

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                prompts_per_batch=4,
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n_samples_per_prompt=2,
                gradient_accumulation_steps=4,
            ),
            device="cpu",
        )

        asyncio.run(
            _run_streaming_optimizer_update(
                trainer,
                ["prompt-a", "prompt-b", "prompt-c", "prompt-d"],
                gradient_accumulation_steps=4,
                prompts_per_batch=4,
                n_samples_per_prompt=2,
            ),
        )

        # 4 microbatches of 1 prompt each, collected/trained/released separately.
        assert collect_calls == [["prompt-a"], ["prompt-b"], ["prompt-c"], ["prompt-d"]]
        # ...but ONE optimizer update: step/global_step advance once, NFT sync once.
        assert trainer.state.step == 1
        assert trainer.state.global_step == 1
        assert after_step_calls == [0]

    def test_streaming_releases_microbatch_before_next_collect(self) -> None:
        """Streaming should not retain the previous rollout batch while collecting the next."""
        import asyncio
        import gc
        import weakref

        from vrl.scripts.common.online import _run_streaming_optimizer_update

        class _Batch:
            __slots__ = (
                "__weakref__",
                "adv_saturation",
                "adv_zero_rate",
                "group_size",
                "iteration",
                "pre_filter_adv_mean",
                "pre_filter_reward_mean",
                "pre_filter_reward_std",
                "timer",
                "trained_prompt_num",
            )

            def __init__(self) -> None:
                self.iteration = object()
                self.timer = object()
                self.pre_filter_reward_mean = 1.0
                self.pre_filter_reward_std = 0.0
                self.pre_filter_adv_mean = 0.0
                self.adv_zero_rate = 0.0
                self.adv_saturation = 0.0
                self.trained_prompt_num = 1
                self.group_size = 2

        class _Stats:
            def as_phase_dict(self):
                return {}

        class _Trainer:
            def __init__(self) -> None:
                self.batch_refs = []

            def begin_optimizer_update(self):
                pass

            async def collect_training_batch(self, prompts):
                del prompts
                if self.batch_refs:
                    gc.collect()
                    assert self.batch_refs[-1]() is None
                batch = _Batch()
                self.batch_refs.append(weakref.ref(batch))
                return batch

            async def backward_on_training_batch(self, batch, *, total_groups):
                del batch, total_groups

            def _step_stats(self, iteration, timer):
                del iteration, timer
                return _Stats()

            async def finish_optimizer_update(self, **kwargs):
                return kwargs

        trainer = _Trainer()
        asyncio.run(
            _run_streaming_optimizer_update(
                trainer,
                ["prompt-a", "prompt-b"],
                gradient_accumulation_steps=2,
                prompts_per_batch=2,
                n_samples_per_prompt=2,
            ),
        )
        gc.collect()
        assert trainer.batch_refs[-1]() is None

    def test_flow_grpo_loss_scaling_includes_timesteps(self) -> None:
        """Flow-GRPO accumulation scales loss by microbatches * train timesteps."""
        import asyncio

        import pytest
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.scripts.common.online import _run_streaming_optimizer_update
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

        recorded_grads: list[float] = []

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

            async def collect_unscored(self, inputs, **kwargs):
                prompts = [getattr(item, "prompt", item) for item in inputs]
                group_size = int(kwargs["group_size"])
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
                return _trajectory_signals(
                    batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx
                )

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                prompts_per_batch=4,
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.1, weight_decay=0.0),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n_samples_per_prompt=2,
                gradient_accumulation_steps=4,
            ),
            device="cpu",
        )

        original_step = trainer._clip_and_step

        def _recording_step(optimizer):
            assert model.weight.grad is not None
            recorded_grads.append(float(model.weight.grad.detach().item()))
            return original_step(optimizer)

        trainer._clip_and_step = _recording_step  # type: ignore[method-assign]

        asyncio.run(
            _run_streaming_optimizer_update(
                trainer,
                ["prompt-a", "prompt-b", "prompt-c", "prompt-d"],
                gradient_accumulation_steps=4,
                prompts_per_batch=4,
                n_samples_per_prompt=2,
            ),
        )

        # One optimizer update over 4 microbatches (1 group each) * 3 timesteps:
        # loss_scale = total_groups(4) * train_timesteps(3) = 12, so the single
        # accumulated gradient is 1.0 (not 3.0 without timestep scaling, not 12.0).
        assert recorded_grads == pytest.approx([1.0])
        assert trainer.state.global_step == 1

    def test_streaming_matches_full_batch_gradient(self) -> None:
        """Streaming microbatch path is gradient-equivalent to the full-batch path."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.batch import RolloutBatch
        from vrl.scripts.common.online import _run_streaming_optimizer_update
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

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

            async def collect_unscored(self, inputs, **kwargs):
                prompts = [getattr(item, "prompt", item) for item in inputs]
                group_size = int(kwargs["group_size"])
                batch_size = len(prompts) * group_size
                group_ids = torch.tensor(
                    [i for i in range(len(prompts)) for _ in range(group_size)],
                    dtype=torch.long,
                )
                return RolloutBatch(
                    observations=torch.zeros(batch_size, 2, 1),
                    actions=torch.zeros(batch_size, 2, 1),
                    rewards=torch.arange(batch_size, dtype=torch.float32),
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=group_ids,
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                return _trajectory_signals(
                    batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx
                )

        def _make_trainer(gas: int) -> OnlineTrainer:
            model = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1.0)
            return OnlineTrainer(
                algorithm=_Algorithm(),
                collector=_Collector(),
                evaluator=_Evaluator(),
                model=model,
                config=TrainerConfig(
                    prompts_per_batch=4,
                    timestep_fraction=1.0,
                    total_epochs=1,
                    drop_zero_advantage=False,
                    output_dir="outputs/",
                    optim=OptimConfig(lr=0.1),
                    ema=EMAConfig(),
                    debug=DebugConfig(),
                    n_samples_per_prompt=2,
                    gradient_accumulation_steps=gas,
                ),
                device="cpu",
            )

        prompts = ["p0", "p1", "p2", "p3"]
        trainer_full = _make_trainer(gas=0)
        asyncio.run(trainer_full.step(list(prompts)))

        trainer_stream = _make_trainer(gas=4)
        asyncio.run(
            _run_streaming_optimizer_update(
                trainer_stream,
                list(prompts),
                gradient_accumulation_steps=4,
                prompts_per_batch=4,
                n_samples_per_prompt=2,
            ),
        )

        assert trainer_full.state.global_step == 1
        assert trainer_stream.state.global_step == 1
        # Same accumulated gradient + one SGD step from identical init weights.
        assert torch.allclose(
            trainer_full.model.weight,
            trainer_stream.model.weight,
            atol=1e-6,
        )


def test_replay_samples_per_chunk_splits_backward_and_preserves_gradient(monkeypatch) -> None:
    """The replay-only chunk integer changes call shape without changing gradients."""
    import asyncio

    import pytest
    import torch
    import torch.nn as nn

    from vrl.algorithms.types import TrainStepMetrics
    from vrl.rollouts.batch import RolloutBatch
    from vrl.scripts.common.online import _run_streaming_optimizer_update
    from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.online import trainer as trainer_module

    device_move_sizes: list[int] = []
    original_move_training_batch_to_device = trainer_module.move_training_batch_to_device

    def _recording_move_training_batch_to_device(batch, *args, **kwargs):
        device_move_sizes.append(int(batch.rewards.shape[0]))
        return original_move_training_batch_to_device(batch, *args, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "move_training_batch_to_device",
        _recording_move_training_batch_to_device,
    )

    class _Algorithm:
        class _Config:
            global_std = False
            eps = 1e-8
            adv_clip_max = 5.0
            kl_coef = 0.0

        config = _Config()

        def compute_advantages_from_tensors(self, rewards, group_ids):
            del group_ids
            return torch.ones_like(rewards)

        def compute_loss(self, inputs):
            signals, _advantages, _old_log_probs = _algorithm_inputs(inputs)
            loss = signals.log_prob.mean()
            return loss, TrainStepMetrics(loss=loss.item(), policy_loss=loss.item())

    class _Collector(CollectorControlFake):
        async def score_rollouts(self, pendings):
            return list(pendings)

        async def collect_unscored(self, prompts, **kwargs):
            prompts = list(prompts)
            group_size = int(kwargs["group_size"])
            batch_size = len(prompts) * group_size
            return RolloutBatch(
                observations=torch.zeros(batch_size, 1, 1),
                actions=torch.zeros(batch_size, 1, 1),
                rewards=torch.arange(batch_size, dtype=torch.float32),
                dones=torch.ones(batch_size, dtype=torch.bool),
                group_ids=torch.zeros(batch_size, dtype=torch.long),
                prompts=[p for p in prompts for _ in range(group_size)],
            )

    class _Evaluator(Evaluator):
        def __init__(self, calls: list[int]) -> None:
            self.calls = calls

        def evaluate(self, model, batch, timestep_idx, **kw):
            del kw
            self.calls.append(int(batch.rewards.shape[0]))
            log_prob = model.weight.reshape(()) * batch.rewards
            return _trajectory_signals(batch, log_prob, timestep_idx)

    def _make_trainer(
        replay_samples_per_chunk: int,
        *,
        streaming: bool,
    ) -> tuple[OnlineTrainer, list[int]]:
        replay_calls: list[int] = []
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(replay_calls),
            model=model,
            config=TrainerConfig(
                prompts_per_batch=1,
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.0),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n_samples_per_prompt=4,
                gradient_accumulation_steps=1 if streaming else 0,
                replay_samples_per_chunk=replay_samples_per_chunk,
            ),
            device="cpu",
        )
        return trainer, replay_calls

    def _run(
        replay_samples_per_chunk: int,
        *,
        streaming: bool,
    ) -> tuple[float, list[int], list[int]]:
        device_move_sizes.clear()
        trainer, replay_calls = _make_trainer(
            replay_samples_per_chunk,
            streaming=streaming,
        )
        recorded_grads: list[float] = []
        original_step = trainer._clip_and_step

        def _recording_step(optimizer):
            assert trainer.model.weight.grad is not None
            recorded_grads.append(float(trainer.model.weight.grad.detach().item()))
            return original_step(optimizer)

        trainer._clip_and_step = _recording_step  # type: ignore[method-assign]

        if streaming:
            asyncio.run(
                _run_streaming_optimizer_update(
                    trainer,
                    ["prompt"],
                    gradient_accumulation_steps=1,
                    prompts_per_batch=1,
                    n_samples_per_prompt=4,
                ),
            )
        else:
            asyncio.run(trainer.step(["prompt"]))

        assert len(recorded_grads) == 1
        return recorded_grads[0], replay_calls, list(device_move_sizes)

    full_grad, full_calls, full_device_moves = _run(
        replay_samples_per_chunk=0,
        streaming=False,
    )
    legacy_split_grad, legacy_split_calls, legacy_split_device_moves = _run(
        replay_samples_per_chunk=2,
        streaming=False,
    )
    streaming_split_grad, streaming_split_calls, streaming_split_device_moves = _run(
        replay_samples_per_chunk=2,
        streaming=True,
    )

    assert full_calls == [4]
    assert 4 in full_device_moves
    assert legacy_split_calls == [2, 2]
    assert streaming_split_calls == [2, 2]
    assert max(legacy_split_device_moves) == 2
    assert max(streaming_split_device_moves) == 2
    assert full_grad == pytest.approx(1.5)
    assert legacy_split_grad == pytest.approx(full_grad)
    assert streaming_split_grad == pytest.approx(full_grad)


def test_gradient_accumulation_steps_validation() -> None:
    """gradient_accumulation_steps must evenly divide prompts_per_batch when > 0."""
    import pytest

    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    def _cfg(rbs: int, gas: int, ppo: int = 1) -> TrainerConfig:
        return TrainerConfig(
            optim=OptimConfig(lr=1e-4),
            n_samples_per_prompt=2,
            prompts_per_batch=rbs,
            timestep_fraction=0.5,
            total_epochs=1,
            output_dir="x",
            drop_zero_advantage=False,
            gradient_accumulation_steps=gas,
            ppo_epochs=ppo,
        )

    # Valid: divisible (streaming) or 0 (legacy full-batch, any ppo_epochs).
    for rbs, gas, ppo in [(32, 32, 1), (8, 4, 1), (4, 4, 1), (4, 0, 1), (4, 0, 3)]:
        _cfg(rbs, gas, ppo)

    with pytest.raises(ValueError, match="evenly divide"):
        _cfg(6, 4)
    with pytest.raises(ValueError, match="evenly divide"):
        _cfg(1, 2)
    with pytest.raises(ValueError, match="ppo_epochs"):
        _cfg(4, 2, ppo=3)
    with pytest.raises(ValueError, match=">= 0"):
        _cfg(4, -1)


def test_microbatch_size_reconciles_with_gradient_accumulation_steps() -> None:
    """microbatch_size (size) and gradient_accumulation_steps (count) derive each other."""
    import pytest

    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    def _cfg(**kw: object) -> TrainerConfig:
        base = dict(
            optim=OptimConfig(lr=1e-4),
            n_samples_per_prompt=2,
            prompts_per_batch=32,
            timestep_fraction=0.5,
            total_epochs=1,
            output_dir="x",
            drop_zero_advantage=False,
        )
        base.update(kw)
        return TrainerConfig(**base)  # type: ignore[arg-type]

    # Declare the SIZE -> the microstep count derives.
    c = _cfg(microbatch_size=4)
    assert c.gradient_accumulation_steps == 8
    assert c.microbatch_size == 4
    # Declare the COUNT (legacy) -> the slice size derives.
    c = _cfg(gradient_accumulation_steps=8)
    assert c.microbatch_size == 4
    assert c.gradient_accumulation_steps == 8
    # Both declared + consistent is allowed.
    _cfg(microbatch_size=4, gradient_accumulation_steps=8)
    # Neither -> legacy full-batch (no streaming).
    c = _cfg()
    assert c.gradient_accumulation_steps == 0
    assert c.microbatch_size == 0

    with pytest.raises(ValueError, match="evenly divide"):
        _cfg(microbatch_size=5)
    with pytest.raises(ValueError, match="set only one"):
        _cfg(microbatch_size=4, gradient_accumulation_steps=4)
    with pytest.raises(ValueError, match="ppo_epochs"):
        _cfg(microbatch_size=4, ppo_epochs=2)
    with pytest.raises(ValueError, match=">= 0"):
        _cfg(microbatch_size=-1)
    with pytest.raises(ValueError, match="samples_per_chunk"):
        _cfg(samples_per_chunk=-1)


def test_replay_samples_per_chunk_is_an_independent_fixed_integer() -> None:
    """Replay defaults to one and never inherits generation's chunk declaration."""
    import pytest

    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    def _cfg(**kw: object) -> TrainerConfig:
        base = dict(
            optim=OptimConfig(lr=1e-4),
            n_samples_per_prompt=2,
            prompts_per_batch=32,
            timestep_fraction=0.5,
            total_epochs=1,
            output_dir="x",
            drop_zero_advantage=False,
        )
        base.update(kw)
        return TrainerConfig(**base)  # type: ignore[arg-type]

    assert _cfg().replay_samples_per_chunk == 1
    assert _cfg(samples_per_chunk=4).replay_samples_per_chunk == 1
    assert _cfg(samples_per_chunk="auto").samples_per_chunk == "auto"
    assert _cfg(samples_per_chunk="auto").replay_samples_per_chunk == 1
    c = _cfg(samples_per_chunk=4, replay_samples_per_chunk=2)
    assert c.samples_per_chunk == 4
    assert c.replay_samples_per_chunk == 2
    assert _cfg(replay_samples_per_chunk=0).replay_samples_per_chunk == 0

    with pytest.raises(ValueError, match="replay_samples_per_chunk"):
        _cfg(replay_samples_per_chunk=-1)
    with pytest.raises(ValueError, match="replay_samples_per_chunk"):
        _cfg(replay_samples_per_chunk="auto")
    with pytest.raises(ValueError, match="samples_per_chunk"):
        _cfg(samples_per_chunk="largest")


def test_fixed_replay_chunk_remains_available_to_distributed_strategies() -> None:
    """DDP/FSDP accept the same explicit replay chunk configuration."""
    from types import SimpleNamespace

    import torch.nn as nn

    from vrl.trainers.core.types import OptimConfig, TrainerConfig
    from vrl.trainers.online import OnlineTrainer

    class _Algorithm:
        class _Config:
            sft_weight = 0.0

        config = _Config()

    for strategy_name in ("ddp", "fsdp"):
        strategy = SimpleNamespace(
            context=SimpleNamespace(strategy=strategy_name),
            prepare_model=lambda model: model,
        )
        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=CollectorControlFake(),
            evaluator=None,
            model=nn.Linear(1, 1),
            config=TrainerConfig(
                optim=OptimConfig(lr=1e-4),
                n_samples_per_prompt=2,
                prompts_per_batch=1,
                timestep_fraction=1.0,
                total_epochs=1,
                output_dir="x",
                drop_zero_advantage=False,
                replay_samples_per_chunk=1,
            ),
            strategy=strategy,  # type: ignore[arg-type]
            device="cpu",
        )
        assert trainer.config.replay_samples_per_chunk == 1


def test_rollout_memory_plan_logs_streaming_and_legacy_warning(caplog) -> None:
    """Startup logs should make rollout microbatch residency visible."""
    import logging

    from vrl.scripts.common.online import _log_rollout_memory_plan
    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    def _cfg(rbs: int, gas: int) -> TrainerConfig:
        return TrainerConfig(
            optim=OptimConfig(lr=1e-4),
            n_samples_per_prompt=2,
            prompts_per_batch=rbs,
            timestep_fraction=1.0,
            total_epochs=1,
            output_dir="x",
            drop_zero_advantage=False,
            gradient_accumulation_steps=gas,
            samples_per_chunk=2,
        )

    logger_name = "vrl.scripts.common.online"
    with caplog.at_level(logging.INFO, logger=logger_name):
        _log_rollout_memory_plan(_cfg(4, 4))
    streaming_messages = [record.getMessage() for record in caplog.records]
    assert any("streaming accumulation enabled" in msg for msg in streaming_messages)
    assert any("microbatch_prompts=1" in msg for msg in streaming_messages)
    assert any("generation_samples_per_chunk=2" in msg for msg in streaming_messages)
    assert any("replay_samples_per_chunk=1" in msg for msg in streaming_messages)
    assert any("target_samples_per_update=8" in msg for msg in streaming_messages)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger_name):
        _log_rollout_memory_plan(_cfg(4, 0))
    legacy_messages = [record.getMessage() for record in caplog.records]
    assert any("legacy full-batch accumulation" in msg for msg in legacy_messages)
    assert any("generation_samples_per_chunk=2" in msg for msg in legacy_messages)
    assert any("replay_samples_per_chunk=1" in msg for msg in legacy_messages)
    # The legacy path must emit a host-RAM residency WARNING. Assert the warning
    # level fired (the behavioral contract) rather than pinning its exact prose,
    # which a benign reword would redden with no real regression.
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_global_std_streaming_divergence_warning(caplog) -> None:
    """global_std=true + streaming with >1 group/microbatch warns; exempt cases don't."""
    import logging

    from omegaconf import OmegaConf

    from vrl.scripts.common.online import _warn_global_std_streaming_divergence
    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    def _tc(rbs: int, gas: int) -> TrainerConfig:
        return TrainerConfig(
            optim=OptimConfig(lr=1e-4),
            n_samples_per_prompt=2,
            prompts_per_batch=rbs,
            timestep_fraction=0.5,
            total_epochs=1,
            output_dir="x",
            drop_zero_advantage=False,
            gradient_accumulation_steps=gas,
        )

    logger_name = "vrl.scripts.common.online"
    cfg_true = OmegaConf.create({"algorithm": {"global_std": True}})

    def _warns(cfg, tc) -> bool:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=logger_name):
            _warn_global_std_streaming_divergence(cfg, tc)
        return any("global_std=true with streaming" in r.getMessage() for r in caplog.records)

    # global_std=true + 2 groups/microbatch (rbs=8, gas=4) -> warn (the sd3 case).
    assert _warns(cfg_true, _tc(8, 4))
    # Exempt: microbatch_size=1 (gas=8 -> 1 group/microbatch; per-group == global).
    assert not _warns(cfg_true, _tc(8, 8))
    # Exempt: global_std=false (per-group std is streaming-equivalent).
    assert not _warns(OmegaConf.create({"algorithm": {"global_std": False}}), _tc(8, 4))
    # Exempt: legacy full-batch (gas=0, no streaming).
    assert not _warns(cfg_true, _tc(8, 0))


def test_host_memory_budget_fail_fast(monkeypatch) -> None:
    """The host-RAM guard raises over budget and passes under it (injected RSS)."""
    import pytest

    from vrl.scripts.common import online
    from vrl.utils.memory import HostMemorySnapshot

    def _inject(used_fraction: float) -> None:
        # total=100GiB; available carved so used_fraction comes out as asked.
        total = 100_000.0
        snap = HostMemorySnapshot(
            rss_mb=total * used_fraction,
            available_mb=total * (1.0 - used_fraction),
            total_mb=total,
        )
        monkeypatch.setattr(online, "capture_host_memory", lambda: snap)

    # Over budget -> fail fast with an actionable message.
    _inject(0.95)
    with pytest.raises(MemoryError, match=r"rollout\.microbatch_size"):
        online._check_host_memory_budget(0.9, microbatch_prompts=1, n_samples_per_prompt=8)

    # Exactly at / under budget -> pass (<= budget does not trip).
    _inject(0.90)
    online._check_host_memory_budget(0.9, microbatch_prompts=1, n_samples_per_prompt=8)
    _inject(0.50)
    online._check_host_memory_budget(0.9, microbatch_prompts=1, n_samples_per_prompt=8)

    # Unreadable host memory (used_fraction None) -> never raises (no false kill).
    monkeypatch.setattr(
        online,
        "capture_host_memory",
        lambda: HostMemorySnapshot(rss_mb=None, available_mb=None, total_mb=None),
    )
    online._check_host_memory_budget(0.9, microbatch_prompts=1, n_samples_per_prompt=8)


def test_host_memory_budget_fraction_bounds() -> None:
    """host_memory_budget_fraction in [0.0,1.0); >0 requires streaming (no footgun)."""
    import pytest

    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    def _cfg(fraction: float, *, microbatch_size: int = 0) -> TrainerConfig:
        return TrainerConfig(
            optim=OptimConfig(lr=1e-4),
            n_samples_per_prompt=2,
            prompts_per_batch=4,
            timestep_fraction=0.5,
            total_epochs=1,
            output_dir="x",
            drop_zero_advantage=False,
            microbatch_size=microbatch_size,
            host_memory_budget_fraction=fraction,
        )

    # >0 budget is valid when streaming is on (microbatch_size>0 -> gas>0).
    for ok in (0.5, 0.9, 0.999):
        assert _cfg(ok, microbatch_size=1).host_memory_budget_fraction == ok
    # 0.0 (guard off) is valid with or without streaming.
    assert _cfg(0.0).host_memory_budget_fraction == 0.0
    # Out-of-range always rejected.
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="host_memory_budget_fraction"):
            _cfg(bad, microbatch_size=1)
    # Footgun: >0 budget with no streaming is a config error, not a silent no-op.
    with pytest.raises(ValueError, match="requires streaming"):
        _cfg(0.9)


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
        inputs=["a", "b"],
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
