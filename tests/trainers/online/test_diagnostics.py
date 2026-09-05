"""OnlineTrainer diagnostics: first-step debug jsonl and post-backward diagnostic-tensor clearing."""

from __future__ import annotations

import pytest

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _diffusion_rollout_batch,
    _EvaluatorAlgorithmFake,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.rollouts.evaluators.base import Evaluator


def _make_parity_boundary_trainer(
    tmp_path,
    *,
    drop_zero_advantage: bool,
    precision_correction=None,
):
    import torch.nn as nn

    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.core.types import EMAConfig, OptimConfig
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    class _Algorithm(_EvaluatorAlgorithmFake):
        precision_correction = PrecisionCorrectionConfig()

        class _Config:
            kl_coef = 0.0

        config = _Config()

    class _Evaluator(Evaluator):
        def evaluate(self, model, batch, timestep_idx, **kw):
            raise AssertionError("the boundary test must not evaluate")

    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    return OnlineTrainer(
        algorithm=_Algorithm(),
        collector=CollectorControlFake(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
            timestep_fraction=1.0,
            drop_zero_advantage=drop_zero_advantage,
            output_dir=str(tmp_path),
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            precision_correction=precision_correction or PrecisionCorrectionConfig(),
        ),
        device="cpu",
    )


class TestDiagnostics:
    """Groups tests for diagnostics."""

    @pytest.mark.parametrize(
        (
            "model_value",
            "late_offset",
            "debug_enabled",
            "expected_finite",
            "failure_pattern",
        ),
        [
            (0.0, 0.0, True, True, None),
            (0.02, 0.0, True, True, "replay parity failed"),
            (float("nan"), 0.0, True, False, "replay parity failed"),
            (0.0, 0.02, True, True, "replay parity failed"),
            (0.02, 0.0, False, True, "replay parity failed"),
        ],
    )
    def test_replay_parity_is_mandatory_while_debug_only_adds_diagnostics(
        self,
        tmp_path,
        model_value: float,
        late_offset: float,
        debug_enabled: bool,
        expected_finite: bool,
        failure_pattern: str | None,
    ) -> None:
        """Full replay parity aborts before training independently of debug."""
        import asyncio
        import json

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

        class _Algorithm(_EvaluatorAlgorithmFake):
            required_signal_keys = ("log_prob",)
            required_data_keys: tuple[str, ...] = ()

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
                assert kwargs["runtime_debug"] is debug_enabled
                group_size = int(kwargs["group_size"])
                return _diffusion_rollout_batch(
                    rewards=torch.arange(group_size, dtype=torch.float32),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    num_steps=2,
                    context={
                        "guidance_scale": 4.5,
                        "runtime_debug": {
                            "ray_chunks": [
                                {
                                    "worker_id": "rollout-0",
                                    "policy_version": 1,
                                },
                            ],
                        },
                    },
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
                    old_log_prob=torch.full_like(log_prob, float(timestep_idx)),
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
                drop_zero_advantage=False,
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(first_step=debug_enabled),
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
        if failure_pattern is not None:
            expected_events = {"replay_parity_gate"}
            if debug_enabled and model_value != 0.0:
                expected_events.add("first_step_logprob_parity")
            assert set(by_event) == expected_events
            full_record = by_event["replay_parity_gate"]
            assert full_record["passed"] is False
            assert full_record["finite"] is expected_finite
            if "first_step_logprob_parity" in by_event:
                assert by_event["first_step_logprob_parity"]["passed"] is False
                assert by_event["first_step_logprob_parity"]["finite"] is expected_finite
        else:
            assert set(by_event) == {
                "replay_parity_gate",
                "first_step_logprob_parity",
            }
            assert all(record["passed"] for record in records)
            assert by_event["replay_parity_gate"]["max_abs_diff"] == 0.0

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
        assert record["rollout_context"]["guidance_scale"] == 4.5
        assert "runtime_debug" not in record["rollout_context"]
        assert record["runtime_debug"]["ray_chunks"][0]["worker_id"] == "rollout-0"
        assert grad_enabled[0] is False
        assert any(grad_enabled[1:])

    def test_replay_parity_passes_only_after_first_measured_update(
        self,
        tmp_path,
    ) -> None:
        """A fully filtered first update skips the gate instead of failing it.

        Streaming reaches finish_optimizer_update with zero pass-zero
        evaluations when every rank's first collection is dropped by the
        zero-advantage filter; distributed all-dummy ranks hit the same shape.
        An empty snapshot must be neutral, not a parity failure.
        """
        import json

        from vrl.algorithms.types import InitialReplayStats
        from vrl.trainers.online.trainer import _ReplayMetrics

        trainer = _make_parity_boundary_trainer(
            tmp_path,
            drop_zero_advantage=True,
        )

        local, local_weight = _ReplayMetrics().initial_replay_snapshot()
        assert local.finite is False
        assert local.logprob_abs_diff_max == float("inf")

        resolved = trainer._validate_first_update_parity(local, local_weight=local_weight)

        assert resolved.finite is True
        assert resolved.logprob_abs_diff_max == 0.0
        assert trainer._replay_parity_passed is False
        assert not (tmp_path / "training_debug.jsonl").exists()

        at_limit = InitialReplayStats(logprob_abs_diff_max=0.01, finite=True)
        resolved = trainer._validate_first_update_parity(at_limit, local_weight=1.0)

        assert resolved.logprob_abs_diff_max == pytest.approx(0.01)
        assert trainer._replay_parity_passed is True
        record = json.loads((tmp_path / "training_debug.jsonl").read_text().strip())
        assert record["event"] == "replay_parity_gate"
        assert record["passed"] is True

        # The proof belongs to this process/backend, so later updates do not
        # repeat it until load_state_dict resets the lifecycle.
        later_mismatch = InitialReplayStats(logprob_abs_diff_max=1.0, finite=True)
        trainer._validate_first_update_parity(later_mismatch, local_weight=1.0)

    def test_precision_drift_guard_stays_pending_until_first_trainable_update(
        self,
        tmp_path,
    ) -> None:
        """A filtered first rollout must not consume the precision guard."""
        import asyncio
        import json

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.trainers.core.types import (
            EMAConfig,
            OptimConfig,
            PrecisionDriftGuardConfig,
        )
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

        class _Algorithm(_EvaluatorAlgorithmFake):
            required_signal_keys = ("log_prob",)
            required_data_keys: tuple[str, ...] = ()

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
            def __init__(self) -> None:
                super().__init__()
                self.collections = 0

            async def score_rollouts(self, pendings):
                return list(pendings)

            async def collect_unscored(self, prompts, **kwargs):
                del prompts
                group_size = int(kwargs["group_size"])
                self.collections += 1
                rewards = (
                    torch.zeros(group_size, dtype=torch.float32)
                    if self.collections == 1
                    else torch.arange(group_size, dtype=torch.float32)
                )
                return _diffusion_rollout_batch(
                    rewards=rewards,
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    num_steps=2,
                )

        evaluator_calls: list[int] = []

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                evaluator_calls.append(timestep_idx)
                return _trajectory_signals(
                    batch,
                    model.weight.view(1).expand(batch.rewards.shape[0]),
                    timestep_idx,
                )

        model = nn.Linear(1, 1, bias=False)
        _stamp_model_precision(model)
        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
                timestep_fraction=1.0,
                drop_zero_advantage=True,
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                precision_drift_guard=PrecisionDriftGuardConfig(mode="fail"),
                train_precision="no",
                output_dir=str(tmp_path),
            ),
            device="cpu",
        )
        assert trainer.config.batch_plan.streaming is False

        filtered_metrics = asyncio.run(trainer.step(["filtered-prompt"]))

        assert filtered_metrics.adv_zero_rate == 1.0
        assert trainer._precision_drift_guard_pending is True
        assert trainer.state.step == 1
        assert trainer.state.global_step == 0
        assert evaluator_calls == []
        assert not (tmp_path / "training_debug.jsonl").exists()

        asyncio.run(trainer.step(["trainable-prompt"]))

        assert trainer._precision_drift_guard_pending is False
        assert trainer.state.step == 2
        assert trainer.state.global_step == 1
        records = [
            json.loads(line)
            for line in (tmp_path / "training_debug.jsonl").read_text().splitlines()
        ]
        precision_records = [
            record for record in records if record["event"] == "precision_drift_guard"
        ]
        assert len(precision_records) == 1
        assert precision_records[0]["violated"] is False
        assert precision_records[0]["worst_stats"]["logprob_abs_diff_max"] == 0.0
        assert all(record["event"] != "first_step_logprob_parity" for record in records)

    def test_fully_filtered_update_still_serializes_metrics_row(
        self,
        tmp_path,
    ) -> None:
        """A no-work streaming update must produce a CSV-serializable row.

        When every streamed microbatch is filtered (e.g. the reward signal is
        exhausted and all groups have zero advantage), finish_optimizer_update
        skips the optimizer but the recipe still writes a metrics row; the
        step result must carry a real InitialReplayStats, not None.
        """
        import asyncio

        from vrl.algorithms.types import InitialReplayStats
        from vrl.trainers.metrics_io import OnlineMetricRow
        from vrl.trainers.online.trainer import RolloutStats

        trainer = _make_parity_boundary_trainer(
            tmp_path,
            drop_zero_advantage=True,
        )

        trainer.begin_optimizer_update()
        metrics = asyncio.run(
            trainer.finish_optimizer_update(
                stats=RolloutStats(),
                reward_mean=0.0,
                reward_std=0.0,
                adv_mean=0.0,
                adv_zero_rate=1.0,
                adv_saturation=0.0,
                group_size=2.0,
                trained_prompt_num=0,
                reward_components={"nsfw_safety": 0.0},
            ),
        )

        assert isinstance(metrics.initial_replay, InitialReplayStats)
        assert trainer._replay_parity_passed is False
        row = OnlineMetricRow.from_step_metrics(0, metrics, ("nsfw_safety",))
        assert row.pre_update_clip_fraction == 0.0
        assert trainer.state.global_step == 0

    def test_streaming_finish_enforces_parity_before_optimizer(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio

        from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
        from vrl.algorithms.types import TrainStepMetrics
        from vrl.trainers.online.trainer import RolloutStats

        trainer = _make_parity_boundary_trainer(
            tmp_path,
            drop_zero_advantage=False,
        )
        trainer.begin_optimizer_update()
        trainer._update_had_training_work = True
        trainer._update_agg_metrics.add(
            TrainStepMetrics(
                logprob_mismatch=LogprobMismatchStats(logprob_abs_diff_max=0.02),
            ),
            weight=1.0,
            capture_initial_replay=True,
        )
        optimizer_called = False

        def fail_if_called(optimizer):
            nonlocal optimizer_called
            del optimizer
            optimizer_called = True
            raise AssertionError("optimizer must remain behind the parity gate")

        monkeypatch.setattr(trainer, "_clip_and_step", fail_if_called)

        with pytest.raises(RuntimeError, match="replay parity failed"):
            asyncio.run(
                trainer.finish_optimizer_update(
                    stats=RolloutStats(),
                    reward_mean=0.0,
                    reward_std=0.0,
                    adv_mean=0.0,
                    adv_zero_rate=0.0,
                    adv_saturation=0.0,
                    group_size=2.0,
                    trained_prompt_num=1,
                    reward_components={},
                ),
            )

        assert optimizer_called is False
        assert trainer._replay_parity_passed is False
        assert trainer.state.step == 0
        assert trainer.state.global_step == 0

    def test_intentional_precision_correction_uses_its_bounded_drift_contract(
        self,
        tmp_path,
    ) -> None:
        from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
        from vrl.algorithms.types import InitialReplayStats

        trainer = _make_parity_boundary_trainer(
            tmp_path,
            drop_zero_advantage=False,
            precision_correction=PrecisionCorrectionConfig(tis_mode="truncate"),
        )

        drift = InitialReplayStats(logprob_abs_diff_max=1.0, finite=True)
        resolved = trainer._validate_first_update_parity(drift, local_weight=1.0)

        assert resolved.logprob_abs_diff_max == pytest.approx(1.0)
        assert trainer._replay_parity_passed is False
        assert not (tmp_path / "training_debug.jsonl").exists()
