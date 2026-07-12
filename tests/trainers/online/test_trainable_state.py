"""OnlineTrainer trainable-state / weight-sync wiring: pre-collect sync ordering and getter requirement."""

from __future__ import annotations

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import _algorithm_inputs, _trajectory_signals
from vrl.rollouts.evaluators.base import Evaluator


class TestTrainableState:
    """Groups tests for trainable state."""

    def test_initial_rollout_weight_sync_happens_before_collect(self) -> None:
        """Checks initial rollout weight sync happens before collect."""
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
                kl_coef = 0.0

            config = _Config()

            def compute_advantages_from_tensors(self, rewards, group_ids):
                del group_ids
                return rewards - rewards.mean()

            def compute_loss(self, inputs):
                signals, advantages, _old_log_probs = _algorithm_inputs(inputs)
                loss = signals.log_prob.mean() + advantages.mean() * 0.0
                return loss, TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                )

        class _Syncer:
            current_policy_version = None  # PolicyVersionProvider: no version tracked

            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def push(self, state_dict):
                self.calls.append(dict(state_dict))

            async def pull(self):
                return dict(self.calls[-1])

        syncer = _Syncer()

        class _Collector(CollectorControlFake):
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
                collect_seen_sync_counts.append(len(syncer.calls))
                group_size = int(kwargs["group_size"])
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
            weight_syncer=syncer,
            sync_state_getter=lambda: {"linear.weight": model.weight.detach().clone()},
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

        asyncio.run(trainer.step(["prompt-a"]))

        assert collect_seen_sync_counts == [1]
        assert len(syncer.calls) == 2

    def test_weight_sync_requires_explicit_trainable_state_getter(self) -> None:
        """Checks weight sync requires explicit trainable state getter."""
        import pytest
        import torch.nn as nn

        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig
        from vrl.trainers.online import OnlineTrainer

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                kl_coef = 0.0

            config = _Config()

        class _Collector(CollectorControlFake):
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
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
