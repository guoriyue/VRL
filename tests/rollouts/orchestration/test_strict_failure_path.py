"""The strict on-policy rollout schedule: GPU handoff order, failure paths and weight sync.

The risk these guard: if collection raises mid-rollout, the driver model may be
restored only after rollout runtime memory was released successfully. Otherwise
both models can become resident after an incomplete GPU handoff. We also pin
that weight sync happens in after_train_step, never before collect, and that a
prepared weight push installs exactly the snapshot taken at prepare time.

These drive the real RolloutRuntimeCoordinator (the phase-transition ordering
under test lives in its ``rollout_phase`` manager) over the real collector and
trainer strategy; a ``Trace`` records the transitions and injects the one-shot
failures, never the ordering.
"""

from __future__ import annotations

import pytest
import torch

from tests.rollouts.collector._helpers import CollectorBench, Trace, real_collector, trainer_side
from vrl.ray.resources import RoleLifecycle
from vrl.rollouts.orchestration.rollout_runtime import (
    RolloutPhaseCleanupError,
)
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.weight_sync import flatten_trainable_module_state

# Either the rollout or the reward sits on the trainer GPU; both shapes park
# the trainer at phase entry.
_ROLLOUT_SHARES_TRAINER_GPU = RoleLifecycle(trainer=(0,), rollout=(0,), reward=())
_REWARD_SHARES_TRAINER_GPU = RoleLifecycle(trainer=(0,), rollout=(1,), reward=(0,))


def _schedule(
    monkeypatch,
    tmp_path,
    *,
    lifecycle: RoleLifecycle | None = _ROLLOUT_SHARES_TRAINER_GPU,
    with_syncer: bool = False,
) -> tuple[StrictOnPolicyRolloutSchedule, CollectorBench, Trace]:
    """A schedule over an initialized trainer; ``lifecycle=None`` is the tiny
    run's own disjoint plan, which never parks the trainer."""

    bench = real_collector(monkeypatch, tmp_path, lifecycle=lifecycle)
    trainer = trainer_side(bench, initialized=True)
    phase = Trace(monkeypatch)
    phase.watch(trainer.strategy, "validate_training_state_parking", "trainer.validate")
    phase.watch(trainer, "training_state", "trainer.get_live_state")
    phase.watch(trainer.strategy, "park_training_state", "trainer.park")
    phase.watch(trainer.strategy, "restore_training_state", "trainer.restore")
    phase.watch(bench.collector, "activate_generation_runtime", "rollout.activate")
    phase.watch(bench.collector, "generate_rollout", "rollout.collect")
    phase.watch(bench.collector, "offload_generation_runtime_memory", "rollout.offload")
    phase.watch(bench.collector, "shutdown", "collector.shutdown")
    coordinator = trainer.coordinator(bench, syncer=with_syncer)
    return StrictOnPolicyRolloutSchedule(lifecycle=coordinator), bench, phase


# The strategy validates inside its own park; the schedule does not re-ask.
_SHARED_PHASE = [
    "trainer.get_live_state",
    "trainer.park",
    "trainer.validate",
    "rollout.activate",
    "rollout.collect",
    "rollout.offload",
    "trainer.get_live_state",
    "trainer.restore",
]


@pytest.mark.asyncio
async def test_strict_schedule_collects_and_syncs(monkeypatch, tmp_path) -> None:
    """One trainer-ready iteration: initial weights pushed, prompts collected under
    that version, weights pushed again after the train step."""

    bench = real_collector(monkeypatch, tmp_path)
    trainer = trainer_side(bench)
    schedule = StrictOnPolicyRolloutSchedule(lifecycle=trainer.coordinator(bench))

    iteration = await schedule.next_iteration(["p0", "p1"], group_size=2, runtime_debug=True)
    await schedule.after_train_step()

    request = bench.trace.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.policy_version == 1
    assert request.runtime_debug is True
    assert len(iteration.batches) == 2
    assert sum(batch.rewards.numel() for batch in iteration.batches) == 4
    assert bench.trace.events.count("update_weights") == 2
    assert bench.runtime.current_policy_version == 2
    assert bench.trace.events.count("activate") == 1
    assert bench.trace.events.count("offload") == 1


@pytest.mark.asyncio
async def test_strict_shared_phase_keeps_handoff_order_in_next_iteration(
    monkeypatch, tmp_path
) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)

    await schedule.next_iteration(["p0"], group_size=1)

    assert phase.events == _SHARED_PHASE


@pytest.mark.asyncio
async def test_strict_parks_trainer_when_only_reward_shares_its_gpu(monkeypatch, tmp_path) -> None:
    schedule, _bench, phase = _schedule(
        monkeypatch, tmp_path, lifecycle=_REWARD_SHARES_TRAINER_GPU
    )

    await schedule.next_iteration(["p0"], group_size=1)

    assert phase.events == _SHARED_PHASE


@pytest.mark.asyncio
async def test_strict_terminal_shutdown_parks_before_shared_pipeline_cleanup(
    monkeypatch, tmp_path
) -> None:
    schedule, _bench, phase = _schedule(
        monkeypatch, tmp_path, lifecycle=_REWARD_SHARES_TRAINER_GPU
    )

    await schedule.shutdown()

    assert phase.events == [
        "trainer.get_live_state",
        "trainer.park",
        "trainer.validate",
        "collector.shutdown",
    ]


@pytest.mark.asyncio
async def test_cleanup_runs_when_collect_raises(monkeypatch, tmp_path) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("rollout.collect", "collect blew up")

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    # Both cleanup hooks ran despite the collect failure, in handoff order.
    calls = phase.events
    assert "rollout.activate" in calls
    assert "rollout.offload" in calls
    assert "trainer.restore" in calls
    assert calls.index("trainer.park") < calls.index("rollout.activate")
    assert calls.index("rollout.offload") < calls.index("trainer.restore")


@pytest.mark.asyncio
async def test_driver_not_restored_when_not_parked(monkeypatch, tmp_path) -> None:
    """A topology that never parked the trainer must not restore it either."""

    schedule, _bench, phase = _schedule(monkeypatch, tmp_path, lifecycle=None)
    phase.fail("rollout.collect", "collect blew up")

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert "trainer.park" not in phase.events
    assert "rollout.offload" in phase.events
    assert "trainer.restore" not in phase.events


@pytest.mark.asyncio
async def test_failed_rollout_offload_keeps_trainer_parked(monkeypatch, tmp_path) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("rollout.collect", "collect blew up")
    phase.fail("rollout.offload", "rollout offload blew up")
    # A restore failure would be observable if the phase manager incorrectly
    # attempted to wake the trainer after an incomplete rollout handoff.
    phase.fail("trainer.restore", "trainer restore blew up")

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "rollout offload blew up"
    assert raised.value.__cause__ is raised.value.root_cause
    assert "trainer.restore" not in phase.events


@pytest.mark.asyncio
async def test_offload_failure_after_collection_keeps_trainer_parked(
    monkeypatch, tmp_path
) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("rollout.offload", "rollout offload blew up")

    with pytest.raises(RuntimeError, match="rollout offload blew up"):
        await schedule.next_iteration([], group_size=2)

    assert "rollout.offload" in phase.events
    assert "trainer.restore" not in phase.events


@pytest.mark.asyncio
async def test_restore_failure_is_reported_after_successful_rollout_offload(
    monkeypatch, tmp_path
) -> None:
    schedule, _bench, phase = _schedule(monkeypatch, tmp_path)
    phase.fail("rollout.collect", "collect blew up")
    phase.fail("trainer.restore", "trainer restore blew up")

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "trainer restore blew up"
    assert phase.events.index("rollout.offload") < phase.events.index("trainer.restore")


@pytest.mark.asyncio
async def test_weight_sync_only_in_after_train_step(monkeypatch, tmp_path) -> None:
    """Weight sync belongs to after_train_step, not the collect path."""

    schedule, bench, _phase = _schedule(monkeypatch, tmp_path, with_syncer=True)

    await schedule.next_iteration([], group_size=2)
    assert "update_weights" not in bench.trace.events

    await schedule.after_train_step()
    assert bench.trace.events.count("update_weights") == 1


@pytest.mark.asyncio
async def test_prepared_weight_push_never_reenters_live_getter(monkeypatch, tmp_path) -> None:
    """The snapshot taken at prepare time is what the rollout receives, however
    the live policy changes in between, and preparing again exports nothing."""

    bench = real_collector(monkeypatch, tmp_path)
    trainer = trainer_side(bench)
    exports = Trace(monkeypatch)
    exports.watch(trainer, "export", "trainer.export")
    coordinator = trainer.coordinator(bench)

    prepared = coordinator.prepare_initial_weight_sync_state()
    assert prepared is not None
    key = next(iter(prepared))
    live = dict(trainer.bundle.model.trainable_modules["transformer"].named_parameters())[
        key.removeprefix("transformer.")
    ]
    with torch.no_grad():
        live.fill_(9.0)
    await coordinator.push_prepared_weights(prepared, RolloutStats())

    assert exports.events == ["trainer.export"]
    assert coordinator.weights_initialized is True
    installed = flatten_trainable_module_state(
        bench.runtime.worker.executor.model.trainable_modules
    )[key]
    assert torch.equal(installed, prepared[key])
    assert not torch.equal(installed, live.detach())
    assert coordinator.prepare_initial_weight_sync_state() is None
    assert exports.events == ["trainer.export"]


@pytest.mark.asyncio
async def test_failed_prepared_weight_push_does_not_publish_initialized_state(
    monkeypatch, tmp_path
) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    trainer = trainer_side(bench)
    coordinator = trainer.coordinator(bench)
    bench.trace.fail("update_weights", "push failed")
    prepared = coordinator.prepare_initial_weight_sync_state()

    with pytest.raises(RuntimeError, match="push failed"):
        await coordinator.push_prepared_weights(prepared, RolloutStats())

    assert coordinator.weights_initialized is False
    assert bench.runtime.current_policy_version == 0
