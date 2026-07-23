"""OnlineTrainer diagnostics: first-step debug jsonl and post-backward diagnostic-tensor clearing."""

from __future__ import annotations

import pytest

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.rollouts.evaluators.base import Evaluator


class TestDiagnostics:
    """Groups tests for diagnostics."""

    @pytest.mark.parametrize(
        ("model_value", "late_offset", "expected_finite", "failure_pattern"),
        [
            (0.0, 0.0, True, None),
            (0.02, 0.0, True, "first-step log-prob parity failed"),
            (float("nan"), 0.0, False, "first-step log-prob parity failed"),
            (0.0, 0.02, True, "full first-step log-prob parity failed"),
        ],
    )
    def test_first_step_debug_enforces_parity_before_optimizer(
        self,
        tmp_path,
        model_value: float,
        late_offset: float,
        expected_finite: bool,
        failure_pattern: str | None,
    ) -> None:
        """The debug record is persisted and bad parity aborts before training."""
        import asyncio
        import json

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
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

        class _Collector(CollectorControlFake):
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
                log_prob = model.weight.view(1).expand(batch.rewards.shape[0]) + float(
                    timestep_idx,
                )
                if timestep_idx > 0:
                    log_prob = log_prob + late_offset
                return _trajectory_signals(
                    batch,
                    log_prob,
                    timestep_idx,
                )

        model = nn.Linear(1, 1, bias=False)
        _stamp_model_precision(model)
        with torch.no_grad():
            model.weight.fill_(model_value)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=False,
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(first_step=True),
                train_precision="no",
                output_dir=str(tmp_path),
            ),
            device="cpu",
        )

        before = model.weight.detach().clone()
        if failure_pattern is not None:
            with pytest.raises(RuntimeError, match=failure_pattern):
                asyncio.run(trainer.step(["prompt-a"]))
        else:
            asyncio.run(trainer.step(["prompt-a"]))

        debug_path = tmp_path / "training_debug.jsonl"
        records = [json.loads(line) for line in debug_path.read_text().splitlines()]
        by_event = {record["event"]: record for record in records}
        if failure_pattern == "first-step log-prob parity failed":
            assert list(by_event) == ["first_step_logprob_parity"]
            assert by_event["first_step_logprob_parity"]["passed"] is False
            assert by_event["first_step_logprob_parity"]["finite"] is expected_finite
        elif failure_pattern == "full first-step log-prob parity failed":
            assert list(by_event) == ["first_step_full_logprob_parity"]
            full_record = by_event["first_step_full_logprob_parity"]
            assert full_record["passed"] is False
            assert full_record["finite"] is True
            assert full_record["max_abs_diff"] == pytest.approx(0.02)
        else:
            assert set(by_event) == {
                "first_step_full_logprob_parity",
                "first_step_logprob_parity",
            }
            assert all(record["passed"] for record in records)
            assert by_event["first_step_full_logprob_parity"]["max_abs_diff"] == 0.0

        if failure_pattern is not None:
            assert trainer.state.step == 0
            assert trainer.state.global_step == 0
            torch.testing.assert_close(model.weight, before, equal_nan=True)
            return

        record = by_event["first_step_logprob_parity"]
        assert record["finite"] is expected_finite
        assert record["precision_policy"]["training_precision"] == "fp32"
        assert record["precision_policy"]["rollout_precision"] == "fp32"
        assert record["precision_policy"]["math_precision"] == "fp32"
        assert record["precision_policy"]["effective_float32_precision"] == {
            "matmul": "ieee",
            "cudnn": "ieee",
        }
        assert record["precision_policy"]["trainer_transformer_dtype"] == "float32"
        assert record["abs_diff"]["mean"] == pytest.approx(0.0)
        assert record["ratio"]["mean"] == pytest.approx(1.0)
        assert record["driver_trainable_before_step"]["tensor_count"] == 1
        assert record["driver_trainable_after_step"]["tensor_count"] == 1
        assert record["runtime_debug"]["ray_chunks"][0]["worker_id"] == "rollout-0"
        assert grad_enabled[0] is False
        assert any(grad_enabled[1:])

    def test_full_parity_gate_is_neutral_when_first_update_has_no_evaluations(
        self,
        tmp_path,
    ) -> None:
        """A fully filtered first update skips the gate instead of failing it.

        Streaming reaches finish_optimizer_update with zero pass-zero
        evaluations when every rank's first collection is dropped by the
        zero-advantage filter; distributed all-dummy ranks hit the same shape.
        An empty snapshot must be neutral, not a parity failure.
        """
        import torch.nn as nn

        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
        from vrl.trainers.online.trainer import _ReplayMetrics

        class _Algorithm:
            class _Config:
                kl_coef = 0.0

            config = _Config()

            def compute_advantages_from_tensors(self, rewards, group_ids):
                raise AssertionError("not exercised")

            def compute_loss(self, inputs):
                raise AssertionError("not exercised")

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                raise AssertionError("the gate must not evaluate anything")

        model = nn.Linear(1, 1, bias=False)
        _stamp_model_precision(model)
        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=CollectorControlFake(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
                timestep_fraction=1.0,
                total_epochs=1,
                drop_zero_advantage=True,
                output_dir=str(tmp_path),
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(first_step=True),
            ),
            device="cpu",
        )

        local, local_weight = _ReplayMetrics().initial_replay_snapshot()
        assert local.finite is False
        assert local.logprob_abs_diff_max == float("inf")

        resolved = trainer._validate_first_update_parity(local, local_weight=local_weight)

        assert resolved.finite is True
        assert resolved.logprob_abs_diff_max == 0.0
        assert not (tmp_path / "training_debug.jsonl").exists()
