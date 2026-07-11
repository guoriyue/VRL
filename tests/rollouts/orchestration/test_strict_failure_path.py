"""Failure-path tests for the strict on-policy rollout schedule.

The risk these guard: if collection raises mid-rollout, the driver model must
still be restored and rollout runtime memory released. Otherwise the next step
trains on a half-initialized / offloaded model. We also pin that weight sync
happens in after_train_step, never before collect.
"""

from __future__ import annotations

import pytest

from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule


class _RaisingCollector:
    """Collector whose generation always raises, like a mid-rollout crash."""

    def __init__(self) -> None:
        self.runtime = None

    async def collect_unscored(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("collect blew up")


class _RecordingLifecycle:
    """Minimal RolloutLifecycle stand-in that records call order."""

    def __init__(self) -> None:
        self.collector = _RaisingCollector()
        self.calls: list[str] = []
        self._offloaded = True

    async def ensure_initial_weights(self, _phase_times: dict[str, float]) -> None:
        self.calls.append("ensure_initial_weights")

    def current_policy_version(self) -> int | None:
        return 0

    def offload_driver_model_for_rollout(self, _phase_times: dict[str, float]) -> bool:
        self.calls.append("offload")
        return self._offloaded

    async def activate_rollout_runtime(self, _phase_times: dict[str, float]) -> None:
        self.calls.append("activate_rollout_runtime")

    async def offload_rollout_runtime_memory(self, _phase_times: dict[str, float]) -> None:
        self.calls.append("offload_rollout_runtime_memory")

    def restore_driver_model_after_rollout(self, _phase_times: dict[str, float]) -> None:
        self.calls.append("restore_driver_model_after_rollout")

    async def sync_weights_after_train(self, _phase_times: dict[str, float]) -> int | None:
        self.calls.append("sync_weights_after_train")
        return 1

    @property
    def state(self) -> object:  # pragma: no cover - not used by these tests
        raise AttributeError


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
    assert "restore_driver_model_after_rollout" in lifecycle.calls
    assert lifecycle.calls.index("offload_rollout_runtime_memory") < lifecycle.calls.index(
        "restore_driver_model_after_rollout"
    )
    # Weights were never synced as part of a failed collection.
    assert "sync_weights_after_train" not in lifecycle.calls


@pytest.mark.asyncio
async def test_driver_not_restored_when_not_offloaded() -> None:
    """If offload was skipped, the failed rollout must not restore the driver."""
    lifecycle = _RecordingLifecycle()
    lifecycle._offloaded = False
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert "offload_rollout_runtime_memory" in lifecycle.calls
    assert "restore_driver_model_after_rollout" not in lifecycle.calls


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
        self.runtime = None

    async def collect_unscored(
        self, *_args: object, **_kwargs: object
    ) -> object:  # pragma: no cover
        raise AssertionError("should not be called for empty prompts")
