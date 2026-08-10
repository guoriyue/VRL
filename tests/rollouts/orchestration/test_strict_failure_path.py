"""Failure-path tests for the strict on-policy rollout schedule.

The risk these guard: if collection raises mid-rollout, the driver model may be
restored only after rollout runtime memory was released successfully. Otherwise
both models can become resident after an incomplete GPU handoff. We also pin
that weight sync happens in after_train_step, never before collect.
"""

from __future__ import annotations

import pytest

from vrl.rollouts.orchestration.strict_on_policy import (
    RolloutPhaseCleanupError,
    StrictOnPolicyRolloutSchedule,
)
from vrl.rollouts.stats import RolloutStats


class _RaisingCollector:
    """Collector whose generation always raises, like a mid-rollout crash."""

    requires_generation_offload_before_reward = True
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False

    def __init__(self) -> None:
        self.generation_runtime = None

    async def collect_unscored(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("collect blew up")


class _RecordingLifecycle:
    """Minimal RolloutLifecycle stand-in that records call order."""

    def __init__(self) -> None:
        self.collector = _RaisingCollector()
        self.calls: list[str] = []
        self._parked = True
        self.fail_rollout_offload = False
        self.fail_restore = False

    def validate_training_state_parking(self) -> None:
        self.calls.append("validate_training_state_parking")

    async def ensure_initial_weights(self, _stats: RolloutStats) -> None:
        self.calls.append("ensure_initial_weights")

    def current_policy_version(self) -> int | None:
        return 0

    def park_training_state_for_rollout(self, _stats: RolloutStats) -> bool:
        self.calls.append("park_training_state_for_rollout")
        return self._parked

    async def activate_rollout_runtime(self, _stats: RolloutStats) -> None:
        self.calls.append("activate_rollout_runtime")

    async def offload_rollout_runtime_memory(self, _stats: RolloutStats) -> None:
        self.calls.append("offload_rollout_runtime_memory")
        if self.fail_rollout_offload:
            raise RuntimeError("rollout offload blew up")

    def restore_training_state_after_rollout(self, _stats: RolloutStats) -> None:
        self.calls.append("restore_training_state_after_rollout")
        if self.fail_restore:
            raise RuntimeError("trainer restore blew up")

    async def sync_weights_after_train(self, _stats: RolloutStats) -> None:
        self.calls.append("sync_weights_after_train")


@pytest.mark.asyncio
async def test_cleanup_runs_when_collect_raises() -> None:
    """Checks cleanup runs when collect raises."""
    lifecycle = _RecordingLifecycle()
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    # Both cleanup hooks ran despite the collect failure, in finally order.
    assert "activate_rollout_runtime" in lifecycle.calls
    assert "offload_rollout_runtime_memory" in lifecycle.calls
    assert "restore_training_state_after_rollout" in lifecycle.calls
    assert lifecycle.calls.index("offload_rollout_runtime_memory") < lifecycle.calls.index(
        "restore_training_state_after_rollout"
    )
    # Weights were never synced as part of a failed collection.
    assert "sync_weights_after_train" not in lifecycle.calls


@pytest.mark.asyncio
async def test_driver_not_restored_when_not_offloaded() -> None:
    """If offload was skipped, the failed rollout must not restore the driver."""
    lifecycle = _RecordingLifecycle()
    lifecycle._parked = False
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert "offload_rollout_runtime_memory" in lifecycle.calls
    assert "restore_training_state_after_rollout" not in lifecycle.calls


@pytest.mark.asyncio
async def test_failed_rollout_offload_keeps_trainer_parked() -> None:
    lifecycle = _RecordingLifecycle()
    lifecycle.fail_rollout_offload = True
    # A restore failure would be observable if strict scheduling incorrectly
    # attempted to wake the trainer after an incomplete rollout handoff.
    lifecycle.fail_restore = True
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "rollout offload blew up"
    assert raised.value.__cause__ is raised.value.root_cause
    assert "restore_training_state_after_rollout" not in lifecycle.calls


@pytest.mark.asyncio
async def test_offload_failure_after_collection_keeps_trainer_parked() -> None:
    lifecycle = _RecordingLifecycle()
    lifecycle.collector = _OkCollector()
    lifecycle.fail_rollout_offload = True
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    with pytest.raises(RuntimeError, match="rollout offload blew up"):
        await schedule.next_iteration([], group_size=2)

    assert "offload_rollout_runtime_memory" in lifecycle.calls
    assert "restore_training_state_after_rollout" not in lifecycle.calls


@pytest.mark.asyncio
async def test_restore_failure_is_reported_after_successful_rollout_offload() -> None:
    lifecycle = _RecordingLifecycle()
    lifecycle.fail_restore = True
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "trainer restore blew up"
    assert lifecycle.calls.index("offload_rollout_runtime_memory") < lifecycle.calls.index(
        "restore_training_state_after_rollout"
    )


@pytest.mark.asyncio
async def test_weight_sync_only_in_after_train_step() -> None:
    """Weight sync belongs to after_train_step, not the collect path."""
    lifecycle = _RecordingLifecycle()
    lifecycle.collector = _OkCollector()
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    await schedule.next_iteration([], group_size=2)
    assert "sync_weights_after_train" not in lifecycle.calls

    await schedule.after_train_step()
    assert lifecycle.calls.count("sync_weights_after_train") == 1


class _OkCollector:
    """Collector that returns no batches (empty prompt list path)."""

    def __init__(self) -> None:
        self.generation_runtime = None

    async def collect_unscored(
        self, *_args: object, **_kwargs: object
    ) -> object:  # pragma: no cover
        raise AssertionError("should not be called for empty prompts")
