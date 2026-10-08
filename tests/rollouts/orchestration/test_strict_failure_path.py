"""Failure-path tests for the strict on-policy rollout schedule.

The risk these guard: if collection raises mid-rollout, the driver model may be
restored only after rollout runtime memory was released successfully. Otherwise
both models can become resident after an incomplete GPU handoff. We also pin
that weight sync happens in after_train_step, never before collect.

These drive the real RolloutRuntimeCoordinator (the phase-transition ordering
under test lives in its ``rollout_phase`` manager) over the real collector and
trainer strategy; a ``Trace`` records the transitions and injects the one-shot
failures, never the ordering.
"""

from __future__ import annotations

import pytest

from tests.rollouts.collector._helpers import CollectorBench, Trace, real_collector, trainer_side
from vrl.ray.resources import RayLifecyclePlan
from vrl.rollouts.orchestration.rollout_runtime import (
    RolloutPhaseCleanupError,
)
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule


def _schedule(
    monkeypatch,
    tmp_path,
    *,
    trainer_shares_gpu: bool = True,
    with_syncer: bool = False,
) -> tuple[StrictOnPolicyRolloutSchedule, CollectorBench, Trace]:
    # The lifecycle plan drives whether the coordinator parks the trainer at
    # phase entry: a rollout on the trainer GPU is the shared-GPU fact.
    lifecycle = (
        RayLifecyclePlan(trainer=(0,), rollout=(0,), reward=()) if trainer_shares_gpu else None
    )
    bench = real_collector(monkeypatch, tmp_path, lifecycle=lifecycle)
    trainer = trainer_side(bench, initialized=True)
    phase = Trace(monkeypatch)
    phase.watch(trainer.strategy, "park_training_state", "park_trainer")
    phase.watch(trainer.strategy, "restore_training_state", "restore_trainer")
    phase.watch(bench.collector, "activate_generation_runtime", "activate_rollout")
    phase.watch(bench.collector, "generate_rollout", "collect")
    phase.watch(bench.collector, "offload_generation_runtime_memory", "offload_rollout")
    coordinator = trainer.coordinator(bench, syncer=with_syncer)
    return StrictOnPolicyRolloutSchedule(lifecycle=coordinator), bench, phase


@pytest.mark.asyncio
async def test_cleanup_runs_when_collect_raises(monkeypatch, tmp_path) -> None:
    schedule, bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("collect", "collect blew up")

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    # Both cleanup hooks ran despite the collect failure, in handoff order.
    calls = phase.events
    assert "activate_rollout" in calls
    assert "offload_rollout" in calls
    assert "restore_trainer" in calls
    assert calls.index("park_trainer") < calls.index("activate_rollout")
    assert calls.index("offload_rollout") < calls.index("restore_trainer")
    # Weights were never synced as part of a failed collection.
    assert "update_weights" not in bench.trace.events


@pytest.mark.asyncio
async def test_driver_not_restored_when_not_parked(monkeypatch, tmp_path) -> None:
    """A topology that never parked the trainer must not restore it either."""

    schedule, _bench, phase = _schedule(monkeypatch, tmp_path, trainer_shares_gpu=False)
    phase.fail("collect", "collect blew up")

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert "park_trainer" not in phase.events
    assert "offload_rollout" in phase.events
    assert "restore_trainer" not in phase.events


@pytest.mark.asyncio
async def test_failed_rollout_offload_keeps_trainer_parked(monkeypatch, tmp_path) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("collect", "collect blew up")
    phase.fail("offload_rollout", "rollout offload blew up")
    # A restore failure would be observable if the phase manager incorrectly
    # attempted to wake the trainer after an incomplete rollout handoff.
    phase.fail("restore_trainer", "trainer restore blew up")

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "rollout offload blew up"
    assert raised.value.__cause__ is raised.value.root_cause
    assert "restore_trainer" not in phase.events


@pytest.mark.asyncio
async def test_offload_failure_after_collection_keeps_trainer_parked(
    monkeypatch, tmp_path
) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("offload_rollout", "rollout offload blew up")

    with pytest.raises(RuntimeError, match="rollout offload blew up"):
        await schedule.next_iteration([], group_size=2)

    assert "offload_rollout" in phase.events
    assert "restore_trainer" not in phase.events


@pytest.mark.asyncio
async def test_restore_failure_is_reported_after_successful_rollout_offload(
    monkeypatch, tmp_path
) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("collect", "collect blew up")
    phase.fail("restore_trainer", "trainer restore blew up")

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "trainer restore blew up"
    assert phase.events.index("offload_rollout") < phase.events.index("restore_trainer")


@pytest.mark.asyncio
async def test_weight_sync_only_in_after_train_step(monkeypatch, tmp_path) -> None:
    """Weight sync belongs to after_train_step, not the collect path."""

    schedule, bench, _phase = _schedule(monkeypatch, tmp_path, with_syncer=True)

    await schedule.next_iteration([], group_size=2)
    assert "update_weights" not in bench.trace.events

    await schedule.after_train_step()
    assert bench.trace.events.count("update_weights") == 1
