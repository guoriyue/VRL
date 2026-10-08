"""OnlineTrainer replay-parity gate and first-step diagnostics on the real trainer.

Every trainer is ``real_trainer`` (tiny SANA, real GRPO and SDE evaluator). The
fp32 tiny stack replays bit-exactly, so a drift the gate sees is one the test
put there: a rollout log-prob that disagrees with the replay at a chosen step,
or rollout weights that differ from the trainer's.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.algorithms.types import InitialReplayStats
from vrl.trainers.online.trainer import TrainingBatch


def _trainer(monkeypatch, tmp_path, *overrides: str) -> TrainerBench:
    return real_trainer(monkeypatch, tmp_path, overrides=tuple(overrides))


def _debug_records(tb: TrainerBench) -> list[dict]:
    path = Path(tb.trainer.config.output_dir) / "training_debug.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _debug_path(tb: TrainerBench) -> Path:
    return Path(tb.trainer.config.output_dir) / "training_debug.jsonl"


def _snapshot(tb: TrainerBench) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in tb.trainable_parameters().items()}


def _unchanged(tb: TrainerBench, before: dict[str, torch.Tensor]) -> bool:
    return all(
        torch.equal(before[name], value) for name, value in tb.trainable_parameters().items()
    )


def _shift_rollout_log_prob(batch: TrainingBatch, step: int, delta: float) -> None:
    """The rollout recorded a log-prob at ``step`` the replay will not reproduce."""

    for rollout in batch.batches:
        old_log_prob = rollout.trajectory.segments["denoise"].tensors["old_log_prob"].value
        old_log_prob[:, step] += delta


class TestDiagnostics:
    """Groups tests for diagnostics."""

    @pytest.mark.parametrize(
        ("drift_step", "delta", "debug_enabled", "expected_finite", "failure_pattern"),
        [
            (None, 0.0, True, True, None),
            (0, 0.02, True, True, "replay parity failed"),
            (0, float("nan"), True, False, "replay parity failed"),
            (1, 0.02, True, True, "replay parity failed"),
            (0, 0.02, False, True, "replay parity failed"),
        ],
        ids=["parity", "early_drift", "non_finite", "late_drift", "drift_without_debug"],
    )
    def test_replay_parity_is_mandatory_while_debug_only_adds_diagnostics(
        self,
        monkeypatch,
        tmp_path,
        drift_step: int | None,
        delta: float,
        debug_enabled: bool,
        expected_finite: bool,
        failure_pattern: str | None,
    ) -> None:
        """Full replay parity aborts before training independently of debug."""

        tb = _trainer(
            monkeypatch,
            tmp_path,
            f"trainer.debug.first_step={str(debug_enabled).lower()}",
            # Three denoise steps, all inside the SDE window and all replayed:
            # the flow-GRPO evaluator skips only the near-deterministic last
            # step, so steps 0 and 1 are both measured by the gate.
            "sampling.num_steps=3",
            "rollout.sde.window_range=[0,3]",
            "actor.timestep_fraction=1.0",
        )
        batch = asyncio.run(tb.trainer.collect_training_batch(["a cat"]))
        assert tb.collector.trace.requests[0].runtime_debug is debug_enabled
        if drift_step is not None:
            _shift_rollout_log_prob(batch, drift_step, delta)
        before = _snapshot(tb)

        if failure_pattern is not None:
            with pytest.raises(RuntimeError, match=failure_pattern):
                asyncio.run(tb.trainer.train_on_rollout_batch(batch))
        else:
            asyncio.run(tb.trainer.train_on_rollout_batch(batch))

        records = _debug_records(tb)
        by_event = {record["event"]: record for record in records}
        assert set(by_event) == {"replay_parity_gate"}
        gate = by_event["replay_parity_gate"]
        if failure_pattern is None:
            assert gate["passed"] is True
            assert gate["max_abs_diff"] == 0.0
            return
        assert gate["passed"] is False
        assert gate["finite"] is expected_finite
        assert tb.trainer.state.step == 0
        assert tb.trainer.state.global_step == 0
        assert _unchanged(tb, before)

    @pytest.mark.parametrize("difference", [0.0, 2.980232238769531e-7, 0.00230485200881958])
    def test_zero_parity_limit_rejects_even_small_finite_drift(
        self, monkeypatch, tmp_path, difference
    ):
        """The four-L40S gate must not inherit the default 0.01 tolerance."""

        trainer = _trainer(
            monkeypatch, tmp_path, "trainer.replay_parity.max_abs_logprob_diff=0.0"
        ).trainer
        stats = InitialReplayStats(logprob_abs_diff_max=difference, finite=True)
        if difference:
            with pytest.raises(RuntimeError, match="replay parity failed"):
                trainer._validate_first_update_parity(stats, local_weight=1.0)
            assert trainer._replay_parity_passed is False
        else:
            trainer._validate_first_update_parity(stats, local_weight=1.0)
            assert trainer._replay_parity_passed is True

    def test_every_update_parity_rejects_drift_after_initial_success(self, monkeypatch, tmp_path):
        trainer = _trainer(
            monkeypatch,
            tmp_path,
            "trainer.replay_parity.max_abs_logprob_diff=0.0",
            "trainer.replay_parity.every_update=true",
        ).trainer
        trainer._validate_first_update_parity(
            InitialReplayStats(logprob_abs_diff_max=0.0, finite=True),
            local_weight=1.0,
        )
        assert trainer._replay_parity_passed
        with pytest.raises(RuntimeError, match="replay parity failed before optimizer update"):
            trainer._validate_first_update_parity(
                InitialReplayStats(logprob_abs_diff_max=1e-12, finite=True),
                local_weight=1.0,
            )

    def test_replay_parity_passes_only_after_first_measured_update(
        self, monkeypatch, tmp_path
    ) -> None:
        """A fully filtered first update skips the gate instead of failing it.

        Streaming reaches finish_optimizer_update with zero pass-zero
        evaluations when every rank's first collection is dropped by the
        zero-advantage filter; distributed all-dummy ranks hit the same shape.
        An empty snapshot must be neutral, not a parity failure.
        """

        from vrl.trainers.online.trainer import _ReplayMetrics

        tb = _trainer(monkeypatch, tmp_path, "trainer.debug.first_step=false")
        trainer = tb.trainer
        assert trainer.config.drop_zero_advantage is True

        local, local_weight = _ReplayMetrics().initial_replay_snapshot()
        assert local.finite is False
        assert local.logprob_abs_diff_max == float("inf")

        resolved = trainer._validate_first_update_parity(local, local_weight=local_weight)

        assert resolved.finite is True
        assert resolved.logprob_abs_diff_max == 0.0
        assert trainer._replay_parity_passed is False
        assert not _debug_path(tb).exists()

        at_limit = InitialReplayStats(logprob_abs_diff_max=0.01, finite=True)
        resolved = trainer._validate_first_update_parity(at_limit, local_weight=1.0)

        assert resolved.logprob_abs_diff_max == pytest.approx(0.01)
        assert trainer._replay_parity_passed is True
        record = json.loads(_debug_path(tb).read_text().strip())
        assert record["event"] == "replay_parity_gate"
        assert record["passed"] is True

        # The proof belongs to this process/backend, so later updates do not
        # repeat it until load_state_dict resets the lifecycle.
        later_mismatch = InitialReplayStats(logprob_abs_diff_max=1.0, finite=True)
        trainer._validate_first_update_parity(later_mismatch, local_weight=1.0)

    def test_fully_filtered_update_still_serializes_metrics_row(
        self, monkeypatch, tmp_path
    ) -> None:
        """A no-work streaming update must produce a CSV-serializable row.

        When every streamed microbatch is filtered (e.g. the reward signal is
        exhausted and all groups have zero advantage), finish_optimizer_update
        skips the optimizer but the recipe still writes a metrics row; the
        step result must carry a real InitialReplayStats, not None.
        """

        from vrl.trainers.metrics_io import OnlineMetricRow
        from vrl.trainers.online.trainer import RolloutStats

        trainer = _trainer(monkeypatch, tmp_path).trainer

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
        self, monkeypatch, tmp_path
    ) -> None:
        """A streamed update whose replay drifted never reaches the optimizer.

        The rollout serves weights that differ from the trainer's (a broken
        weight transport), so every streamed replay disagrees with its rollout.
        """

        from vrl.scripts.common.online import _run_streaming_optimizer_update

        tb = _trainer(
            monkeypatch, tmp_path, "actor.prompts_per_collection=1", "actor.ppo_epochs=1"
        )
        runtime = tb.collector.runtime
        real_update = runtime.update_weights
        generator = torch.Generator().manual_seed(1)

        async def corrupted_update(trainable_state, policy_version):
            return await real_update(
                {
                    name: value + torch.randn(value.shape, generator=generator) * 0.05
                    for name, value in trainable_state.items()
                },
                policy_version,
            )

        monkeypatch.setattr(runtime, "update_weights", corrupted_update)
        tb.collector.trace.watch(tb.trainer, "_clip_and_step", "optimizer_step")
        before = _snapshot(tb)

        with pytest.raises(RuntimeError, match="replay parity failed"):
            asyncio.run(
                _run_streaming_optimizer_update(
                    tb.trainer, ["a cat"], batch_plan=tb.trainer.config.batch_plan
                ),
            )

        assert "optimizer_step" not in tb.collector.trace.events
        assert tb.trainer._replay_parity_passed is False
        assert tb.trainer.state.step == 0
        assert tb.trainer.state.global_step == 0
        assert _unchanged(tb, before)

    def test_intentional_precision_correction_uses_its_bounded_drift_contract(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        trainer = _trainer(
            monkeypatch, tmp_path, "trainer.precision_correction.tis_mode=truncate"
        ).trainer

        drift = InitialReplayStats(logprob_abs_diff_max=1.0, finite=True)
        with caplog.at_level("WARNING", logger="vrl.trainers.online.trainer"):
            resolved = trainer._validate_first_update_parity(drift, local_weight=1.0)

        # Correction modes bound or bypass the drift inside the loss, so the
        # gate must not stop the run, but it still measures and reports it.
        assert resolved.logprob_abs_diff_max == pytest.approx(1.0)
        assert trainer._replay_parity_passed is False
        record = json.loads(
            (Path(trainer.config.output_dir) / "training_debug.jsonl").read_text().strip()
        )
        assert record["event"] == "replay_parity_gate"
        assert record["passed"] is False
        assert record["enforced"] is False
        assert "driver_trainable_before_step" in record
        assert any("precision_correction is enabled" in m for m in caplog.messages)

    def test_every_update_parity_records_the_digest_only_on_first_proof_and_failure(
        self, monkeypatch, tmp_path
    ) -> None:
        tb = _trainer(monkeypatch, tmp_path, "trainer.replay_parity.every_update=true")
        trainer = tb.trainer
        for _ in range(2):
            trainer._validate_first_update_parity(
                InitialReplayStats(logprob_abs_diff_max=0.0, finite=True), local_weight=1.0
            )
        with pytest.raises(RuntimeError, match="replay parity failed"):
            trainer._validate_first_update_parity(
                InitialReplayStats(logprob_abs_diff_max=0.5, finite=True), local_weight=1.0
            )

        records = _debug_records(tb)
        assert [r["passed"] for r in records] == [True, True, False]
        assert ["driver_trainable_before_step" in r for r in records] == [True, False, True]
