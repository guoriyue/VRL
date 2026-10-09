"""The strict on-policy schedule over the real coordinator, collector and trainer strategy."""

from __future__ import annotations

import pytest
import torch

from tests.rollouts.collector._helpers import Trace, real_collector, trainer_side
from vrl.ray.resources import RayLifecyclePlan
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.weight_sync import flatten_trainable_module_state


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


def _phase_trace(monkeypatch, bench, trainer) -> Trace:
    """Order of the trainer- and rollout-side transitions the coordinator drives."""

    phase = Trace(monkeypatch)
    phase.watch(trainer.strategy, "validate_training_state_parking", "trainer.validate")
    phase.watch(trainer, "training_state", "trainer.get_live_state")
    phase.watch(trainer.strategy, "park_training_state", "trainer.park")
    phase.watch(trainer.strategy, "restore_training_state", "trainer.restore")
    phase.watch(bench.collector, "activate_generation_runtime", "rollout.activate")
    phase.watch(bench.collector, "generate_rollout", "rollout.collect")
    phase.watch(bench.collector, "offload_generation_runtime_memory", "rollout.offload")
    phase.watch(bench.collector, "shutdown", "collector.shutdown")
    return phase


def _parking_schedule(
    monkeypatch, tmp_path, *, reward_uses_trainer: bool = False
) -> tuple[StrictOnPolicyRolloutSchedule, Trace]:
    # Either the rollout or the reward sits on the trainer GPU; both shapes park
    # the trainer at phase entry.
    lifecycle = (
        RayLifecyclePlan(trainer=(0,), rollout=(1,), reward=(0,))
        if reward_uses_trainer
        else RayLifecyclePlan(trainer=(0,), rollout=(0,), reward=(2,))
    )
    bench = real_collector(monkeypatch, tmp_path, lifecycle=lifecycle)
    trainer = trainer_side(bench, initialized=True)
    phase = _phase_trace(monkeypatch, bench, trainer)
    coordinator = trainer.coordinator(bench, syncer=False)
    return StrictOnPolicyRolloutSchedule(lifecycle=coordinator), phase


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
async def test_strict_shared_phase_keeps_handoff_order_in_next_iteration(
    monkeypatch, tmp_path
) -> None:
    schedule, phase = _parking_schedule(monkeypatch, tmp_path)

    await schedule.next_iteration(["p0"], group_size=1)

    assert phase.events == _SHARED_PHASE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing_event", "message", "expect_restore"),
    [
        ("rollout.collect", "collect failed", True),
        ("rollout.offload", "offload failed", False),
        ("trainer.restore", "restore failed", True),
    ],
)
async def test_strict_shared_phase_restores_only_after_rollout_offload(
    monkeypatch, tmp_path, failing_event: str, message: str, expect_restore: bool
) -> None:
    schedule, phase = _parking_schedule(monkeypatch, tmp_path)
    phase.fail(failing_event, message)

    with pytest.raises(RuntimeError, match=message):
        await schedule.next_iteration(["p0"], group_size=1)

    assert ("trainer.restore" in phase.events) is expect_restore
    if expect_restore:
        assert phase.events.index("trainer.restore") > phase.events.index("rollout.offload")


@pytest.mark.asyncio
async def test_strict_parks_trainer_when_only_reward_shares_its_gpu(monkeypatch, tmp_path) -> None:
    schedule, phase = _parking_schedule(monkeypatch, tmp_path, reward_uses_trainer=True)

    await schedule.next_iteration(["p0"], group_size=1)

    assert phase.events == _SHARED_PHASE


@pytest.mark.asyncio
async def test_strict_terminal_shutdown_parks_before_shared_pipeline_cleanup(
    monkeypatch, tmp_path
) -> None:
    schedule, phase = _parking_schedule(monkeypatch, tmp_path, reward_uses_trainer=True)

    await schedule.shutdown()

    assert phase.events == [
        "trainer.get_live_state",
        "trainer.park",
        "trainer.validate",
        "collector.shutdown",
    ]


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
