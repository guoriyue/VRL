"""OnlineTrainer diagnostics: first-step debug jsonl and post-backward diagnostic-tensor clearing."""

from __future__ import annotations

from tests.trainers.online._helpers import _algorithm_inputs, _trajectory_signals
from vrl.rollouts.evaluators.base import Evaluator


class TestDiagnostics:
    """Groups tests for diagnostics."""
    def test_first_step_debug_writes_training_debug_jsonl(self, tmp_path) -> None:
        """Checks first-step debug writes training debug JSONL."""
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
                kl_coef = 0.0

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

        class _Collector:
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
                assert kwargs["runtime_debug"] is True
                group_size = int(kwargs["group_size"])
                return RolloutBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.arange(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    context={
                        "rollout_transformer_dtype": "float32",
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
                prompts_per_batch=1,
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(first_step=True),
                n_samples_per_prompt=2,
                train_precision="no",
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
        assert record["precision_policy"]["train_precision"] == "fp32"
        assert record["precision_policy"]["rollout_precision"] == "fp32"
        assert record["precision_policy"]["math_precision"] == "fp32"
        assert record["precision_policy"]["trainer_autocast_enabled"] is False
        assert record["precision_policy"]["trainer_transformer_dtype"] == "float32"
        assert record["precision_policy"]["rollout_transformer_dtype"] == "float32"
        assert record["abs_diff"]["mean"] == pytest.approx(1.0)
        assert record["ratio"]["mean"] == pytest.approx(torch.exp(torch.tensor(1.0)).item())
        assert record["driver_trainable_before_step"]["tensor_count"] == 1
        assert record["driver_trainable_after_step"]["tensor_count"] == 1
        assert record["rollout_context"]["rollout_transformer_dtype"] == "float32"
        assert record["runtime_debug"]["ray_chunks"][0]["worker_id"] == "rollout-0"
        assert grad_enabled[0] is False
        assert any(grad_enabled[1:])

    def test_algorithm_diagnostic_tensors_are_cleared_after_backward(self) -> None:
        """Checks algorithm diagnostic tensors are cleared after backward."""
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
                kl_coef = 0.0

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

        class _Collector:
            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
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

        assert algorithm._last_policy_loss_tensor is None
        assert algorithm._last_kl_term_tensor is None
