"""Integration tests for the continuous rollout schedule."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.generation.execution.types import StaleSlotDiscard
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration import (
    ContinuousRolloutSchedule,
    RolloutScheduleMode,
    build_rollout_schedule,
)
from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState


def _batch(prompts: list[str], group_size: int) -> RolloutBatch:
    batch_size = len(prompts) * group_size
    group_ids = torch.tensor(
        [idx for idx in range(len(prompts)) for _ in range(group_size)],
        dtype=torch.long,
    )
    return RolloutBatch(
        observations=torch.zeros(batch_size, 1, 1),
        actions=torch.zeros(batch_size, 1, 1),
        rewards=torch.arange(batch_size, dtype=torch.float32),
        dones=torch.ones(batch_size, dtype=torch.bool),
        group_ids=group_ids,
        prompts=[prompt for prompt in prompts for _ in range(group_size)],
    )


class _Runtime:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.requires_driver_model_offload = False
        self.colocated = False
        # Default False keeps every existing test on the draining barrier; the
        # non-draining tests flip it True to exercise the slot-backed path.
        self.supports_non_draining_weight_sync = False

    def is_colocated(self) -> bool:
        # Fakes implement the GenerationRuntime protocol method directly
        # instead of mimicking the runtime's internal config layout.
        return self.colocated


class _Syncer:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    async def push(self, state_dict: dict[str, Any]) -> None:
        self.calls.append(dict(state_dict))
        self.runtime.current_policy_version += 1

    async def pull(self) -> dict[str, Any]:
        return dict(self.calls[-1])

    @property
    def current_policy_version(self) -> int | None:
        # Mirrors RayRuntimeWeightSyncer's PolicyVersionProvider contract.
        return self.runtime.current_policy_version


class _FailingPostTrainSyncer(_Syncer):
    def __init__(self, runtime: _Runtime) -> None:
        super().__init__(runtime)
        self.push_attempts = 0

    async def push(self, state_dict: dict[str, Any]) -> None:
        self.push_attempts += 1
        if self.push_attempts == 2:
            raise RuntimeError("worker install ACK mismatch")
        await super().push(state_dict)


class _Collector:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.requires_runtime_offload_before_reward = False
        self.requires_driver_model_offload_for_reward = False
        self.supports_reward_generation_overlap = False
        self.calls: list[dict[str, Any]] = []
        self.activation_calls = 0
        self.offload_calls = 0
        self.shutdown_calls = 0
        self.shutdown_failures = 0

    async def collect_unscored(self, inputs: Any, **kwargs: Any) -> RolloutBatch:
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.calls.append({"prompts": prompts, **dict(kwargs)})
        return _batch(prompts, int(kwargs["group_size"]))

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        return list(pendings)

    async def activate_runtime(self) -> None:
        self.activation_calls += 1

    async def offload_runtime_memory(self) -> None:
        self.offload_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("collector cleanup failed")


def _continuous_config(**continuous: Any) -> SimpleNamespace:
    defaults = {
        "max_inflight_groups": 1,
        "max_ready_groups": 4,
        "max_ready_bytes_mb": 8192,
        "max_stale_policy_versions": 1,
        "wait_timeout_s": 5.0,
        "queue_poll_interval_s": 0.001,
        "fail_fast_errors": 3,
    }
    defaults.update(continuous)
    return SimpleNamespace(
        schedule_mode="continuous",
        continuous=SimpleNamespace(**defaults),
    )


def _build(
    config: SimpleNamespace,
    collector: _Collector,
    syncer: _Syncer | None,
    *,
    algorithm_tolerates_off_policy_staleness: bool = True,
    initially_initialized: bool = False,
):
    initialized = {"value": initially_initialized}

    def _set(value: bool) -> None:
        initialized["value"] = bool(value)

    model = nn.Linear(1, 1)
    return build_rollout_schedule(
        config,
        collector=collector,
        strategy=SingleProcessStrategy(),
        training_state_getter=lambda: TrainingMemoryState(
            model=model,
            ref_model=None,
            optimizer=None,
            ema=None,
            grad_scaler=None,
            device=torch.device("cpu"),
        ),
        weight_syncer=syncer,
        sync_state_getter=(lambda: {"w": 1}) if syncer is not None else None,
        weights_initialized=lambda: initialized["value"],
        set_weights_initialized=_set,
        algorithm_tolerates_off_policy_staleness=(algorithm_tolerates_off_policy_staleness),
    )


async def _snapshot_when(
    schedule: ContinuousRolloutSchedule,
    condition: Callable[[Any], bool],
    *,
    timeout_s: float = 5.0,
) -> Any:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        snapshot = await schedule._debug_snapshot()
        if condition(snapshot):
            return snapshot
        if loop.time() >= deadline:
            raise AssertionError("owner snapshot condition not reached before timeout")
        await asyncio.sleep(0.001)


def test_factory_builds_continuous_schedule() -> None:
    """Checks factory builds continuous schedule."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))
    assert isinstance(schedule, ContinuousRolloutSchedule)
    assert schedule.mode is RolloutScheduleMode.CONTINUOUS


@pytest.mark.asyncio
async def test_shutdown_joins_owner_and_is_idempotent() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    schedule = _build(
        _continuous_config(),
        collector,
        _Syncer(runtime),
    )

    await schedule.next_iteration(["p0"], group_size=1)
    running = await schedule._debug_snapshot()
    assert running.producer_state is not None
    assert running.producer_state.running is True

    await asyncio.gather(schedule.shutdown(), schedule.shutdown())
    await schedule.shutdown()

    assert collector.shutdown_calls == 1
    assert schedule._owner._stopped.is_set()
    thread = schedule._owner._thread
    assert thread is not None and not thread.is_alive()


@pytest.mark.asyncio
async def test_shutdown_failure_retries_cleanup_before_closing_owner() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.shutdown_failures = 1
    schedule = _build(_continuous_config(), collector, _Syncer(runtime))

    await schedule.next_iteration(["p0"], group_size=1)

    with pytest.raises(RuntimeError, match="collector cleanup failed"):
        await schedule.shutdown()

    thread = schedule._owner._thread
    assert collector.shutdown_calls == 1
    assert schedule._owner._closed is False
    assert thread is not None and thread.is_alive()

    await schedule.shutdown()
    await schedule.shutdown()

    assert collector.shutdown_calls == 2
    assert schedule._owner._closed is True
    assert thread is not None and not thread.is_alive()


class _SlowCollector(_Collector):
    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        await asyncio.sleep(0.02)
        return await super().collect_unscored(prompts, **kwargs)


@pytest.mark.asyncio
async def test_owner_production_advances_while_trainer_loop_is_blocked() -> None:
    runtime = _Runtime()
    collector = _SlowCollector(runtime)
    schedule = _build(_continuous_config(), collector, _Syncer(runtime))

    try:
        await schedule.next_iteration(["p0"], group_size=1)
        before = await schedule._debug_snapshot()
        assert before.producer_state is not None

        # This blocks the trainer asyncio loop exactly like synchronous backward.
        time.sleep(0.12)

        after = await schedule._debug_snapshot()
        assert after.producer_state is not None
        assert after.producer_state.tick_count > before.producer_state.tick_count
        assert after.producer_state.submitted_count > before.producer_state.submitted_count
        assert after.producer_state.completed_count > before.producer_state.completed_count
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_initialized_runtime_does_not_receive_redundant_initial_push() -> None:
    runtime = _Runtime()
    runtime.current_policy_version = 7
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(),
        collector,
        syncer,
        initially_initialized=True,
    )

    try:
        iteration = await schedule.next_iteration(["p0"], group_size=1)

        assert iteration.policy_version == 7
        assert runtime.current_policy_version == 7
        assert syncer.calls == []
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_reset_reuses_committed_runtime_weights_without_version_bump() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        first = await schedule.next_iteration(["p0"], group_size=1)
        assert first.policy_version == 1
        assert len(syncer.calls) == 1

        schedule.reset()
        second = await schedule.next_iteration(["p0"], group_size=1)

        assert second.policy_version == 1
        assert len(syncer.calls) == 1
        assert collector.shutdown_calls == 0
    finally:
        await schedule.shutdown()


def test_continuous_rejects_stale_window_for_intolerant_algorithm() -> None:
    """A likelihood-free algorithm + max_stale>0 must fail fast as unsound."""
    runtime = _Runtime()
    with pytest.raises(ValueError, match="likelihood-free"):
        _build(
            _continuous_config(max_stale_policy_versions=1),
            _Collector(runtime),
            _Syncer(runtime),
            algorithm_tolerates_off_policy_staleness=False,
        )


def test_continuous_allows_stale_window_for_tolerant_algorithm() -> None:
    """A GRPO-family (tolerant) algorithm + max_stale>0 builds normally."""
    runtime = _Runtime()
    schedule = _build(
        _continuous_config(max_stale_policy_versions=1),
        _Collector(runtime),
        _Syncer(runtime),
        algorithm_tolerates_off_policy_staleness=True,
    )
    assert isinstance(schedule, ContinuousRolloutSchedule)


def test_continuous_rejects_zero_window_before_algorithm_gate() -> None:
    """Zero-window execution belongs to strict_on_policy, not continuous."""
    runtime = _Runtime()
    with pytest.raises(ValueError, match=r"max_stale_policy_versions >= 1"):
        _build(
            _continuous_config(max_stale_policy_versions=0),
            _Collector(runtime),
            _Syncer(runtime),
            algorithm_tolerates_off_policy_staleness=False,
        )


@pytest.mark.asyncio
async def test_continuous_drains_full_homogeneous_iteration() -> None:
    """Checks continuous drains full homogeneous iteration."""
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        iteration = await schedule.next_iteration(["p0", "p1"], group_size=2)

        # Full set, one fresh policy version, distinct group ids 0..1.
        assert iteration.mode is RolloutScheduleMode.CONTINUOUS
        assert iteration.policy_version == 1
        assert iteration.prompt_count == 2
        assert iteration.sample_count == 4
        assert len(iteration.batches) == 2
        group_ids = sorted(int(b.group_ids[0]) for b in iteration.batches)
        assert group_ids == [0, 1]
        assert all(b.context["rollout_policy_version"] == 1 for b in iteration.batches)
        assert "continuous.queue_wait_s" in iteration.stats.as_phase_dict()
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_weight_sync_barrier_advances_version_and_resumes() -> None:
    """Checks weight sync barrier advances version and resumes."""
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        first = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert first.policy_version == 1

        sync_calls_before = len(syncer.calls)
        await schedule.after_train_step()
        # Barrier performed exactly one post-train sync and resumed admission.
        assert len(syncer.calls) == sync_calls_before + 1
        snapshot = await schedule._debug_snapshot()
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
        assert runtime.current_policy_version == 2

        second = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert second.policy_version == 2
        assert second.rollout_id == 1
        assert second.metadata["consume_policy_version"] == 2
        assert second.metadata["stale_policy_versions"] == 0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_partial_commit_failure_closes_admission_and_runtime() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _FailingPostTrainSyncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        await schedule.next_iteration(["p0"], group_size=1)

        with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
            await schedule.after_train_step()

        calls_after_failure = len(collector.calls)
        failed = await schedule._debug_snapshot()
        assert failed.producer_state is None
        assert failed.queue_stats == {}
        assert "worker install ACK mismatch" in str(failed.terminal_error)
        assert collector.shutdown_calls == 1

        await asyncio.sleep(0.02)
        assert len(collector.calls) == calls_after_failure
        with pytest.raises(RuntimeError, match="owner has failed"):
            await schedule.next_iteration(["p0"], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_after_train_step_purges_items_outside_production_window() -> None:
    """The first lag is retained; the next commit purges version-one items."""
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(max_ready_groups=4, max_stale_policy_versions=1),
        collector,
        syncer,
    )

    try:
        first = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert first.policy_version == 1

        # Simulate optimizer time: producer keeps filling the ready queue with
        # v1 work while the trainer is still training that same v1 rollout.
        before = await _snapshot_when(
            schedule,
            lambda item: item.queue_stats.get("ready_items", 0) > 0,
        )
        queued_before_sync = before.queue_stats
        assert queued_before_sync["ready_items"] > 0
        assert queued_before_sync["dropped_stale"] == 0

        first_sync = await schedule.after_train_step()
        assert runtime.current_policy_version == 2
        assert first_sync["continuous.post_sync_dropped_stale"] == 0.0

        second_sync = await schedule.after_train_step()
        after = await schedule._debug_snapshot()
        assert runtime.current_policy_version == 3
        assert (
            second_sync["continuous.post_sync_dropped_stale"] >= queued_before_sync["ready_items"]
        )
        assert after.queue_stats["dropped_stale"] >= queued_before_sync["ready_items"]
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_queue_capacity_autosizes_to_prompt_set() -> None:
    # max_ready_groups (2) is smaller than the prompt set (3); the schedule must
    # still be able to assemble a full iteration rather than deadlock.
    """Checks queue capacity autosizes to prompt set."""
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(max_ready_groups=2), collector, syncer)

    try:
        iteration = await schedule.next_iteration(["a", "b", "c"], group_size=2)
        assert iteration.prompt_count == 3
        assert iteration.sample_count == 6
        assert sorted(int(b.group_ids[0]) for b in iteration.batches) == [0, 1, 2]
    finally:
        await schedule.shutdown()


def test_rejects_colocated_runtime() -> None:
    """Checks that rejects colocated runtime."""
    runtime = _Runtime()
    runtime.colocated = True
    with pytest.raises(RuntimeError, match="separate trainer and rollout GPU"):
        _build(_continuous_config(), _Collector(runtime), None)


def test_rejects_mid_iteration_reward_offload() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.requires_runtime_offload_before_reward = True
    with pytest.raises(RuntimeError, match=r"does not offload.*mid-iteration"):
        _build(_continuous_config(), collector, _Syncer(runtime))


def test_rejects_reward_scoring_on_trainer_gpu() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.requires_driver_model_offload_for_reward = True

    with pytest.raises(RuntimeError, match="trainer GPU while backward overlaps"):
        _build(_continuous_config(), collector, _Syncer(runtime))


class _FailingCollector(_Collector):
    def __init__(self, runtime: _Runtime, message: str = "boom") -> None:
        super().__init__(runtime)
        self.message = message

    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        raise RuntimeError(self.message)


@pytest.mark.asyncio
async def test_persistent_producer_failure_fails_fast_with_root_cause() -> None:
    # Every generation fails. The consumer must surface the producer's root
    # cause well before the (long) wait timeout, not an opaque timeout.
    """Checks persistent producer failure fails fast with root cause."""
    runtime = _Runtime()
    collector = _FailingCollector(runtime, message="reward model OOM")
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(wait_timeout_s=30.0, fail_fast_errors=2),
        collector,
        syncer,
    )

    try:
        with pytest.raises(RuntimeError, match="reward model OOM") as excinfo:
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert "failing every generation" in str(excinfo.value)
    finally:
        await schedule.shutdown()


class _RewardFailingCollector(_Collector):
    """Generation succeeds; reward scoring always fails."""

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        raise RuntimeError("reward model exploded")


@pytest.mark.asyncio
async def test_reward_failure_fails_fast_and_never_reaches_queue() -> None:
    # Reward scoring (not generation) fails persistently: the consumer must
    # surface that root cause and the ready queue must stay empty.
    """Checks reward failure fails fast and never reaches queue."""
    runtime = _Runtime()
    collector = _RewardFailingCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(wait_timeout_s=30.0, fail_fast_errors=2),
        collector,
        syncer,
    )

    try:
        with pytest.raises(RuntimeError, match="reward model exploded"):
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert (await schedule._debug_snapshot()).queue_stats == {}
    finally:
        await schedule.shutdown()


class _GatedScoreCollector(_Collector):
    """Reward scoring blocks until the test opens the gate."""

    def __init__(self, runtime: _Runtime) -> None:
        super().__init__(runtime)
        self.allow_score = threading.Event()
        self.allow_score.set()
        self.score_blocked = threading.Event()

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        if not self.allow_score.is_set():
            self.score_blocked.set()
            await asyncio.to_thread(self.allow_score.wait)
        return await super().score_rollouts(pendings)


@pytest.mark.asyncio
async def test_weight_sync_waits_for_inflight_reward() -> None:
    # after_train_step must drain in-flight generation+reward before pushing
    # weights; syncing earlier would mix two policies inside one request.
    """Checks weight sync waits for in-flight reward scoring."""
    runtime = _Runtime()
    collector = _GatedScoreCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)

        # Gate scoring, then wait for the producer's next in-flight group to
        # reach (and block inside) the reward phase.
        collector.score_blocked.clear()
        collector.allow_score.clear()
        assert await asyncio.to_thread(collector.score_blocked.wait, 5.0)

        sync_calls_before = len(syncer.calls)
        barrier = asyncio.create_task(schedule.after_train_step())
        await asyncio.sleep(0.05)
        # Reward still in flight: admission paused, sync not yet performed.
        blocked = await schedule._debug_snapshot()
        assert blocked.producer_state is not None
        assert blocked.producer_state.paused_for_weight_sync is True
        assert len(syncer.calls) == sync_calls_before
        assert not barrier.done()

        collector.allow_score.set()
        await asyncio.wait_for(barrier, 5.0)
        assert len(syncer.calls) == sync_calls_before + 1
        resumed = await schedule._debug_snapshot()
        assert resumed.producer_state is not None
        assert resumed.producer_state.paused_for_weight_sync is False
    finally:
        collector.allow_score.set()
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_draining_barrier_reports_mode_zero() -> None:
    """Default (no versioned slots): the barrier mode metric is 0 (draining)."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)
        phases = await schedule.after_train_step()
        assert phases["continuous.weight_sync_barrier_mode"] == 0.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_non_draining_sync_skips_inflight_wait() -> None:
    # The whole point of versioned slots: when the runtime advertises non-draining
    # support, after_train_step must NOT wait for in-flight generation/reward — it
    # syncs and returns while the gated reward is still blocked (the in-flight
    # request keeps its own version's slot and finishes concurrently).
    """Checks non-draining sync does not drain in-flight work."""
    runtime = _Runtime()
    runtime.supports_non_draining_weight_sync = True
    collector = _GatedScoreCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)

        # Block reward and wait for the producer's next group to be in-flight.
        collector.score_blocked.clear()
        collector.allow_score.clear()
        assert await asyncio.to_thread(collector.score_blocked.wait, 5.0)

        sync_calls_before = len(syncer.calls)
        # Must complete WITHOUT opening the reward gate (contrast with the draining
        # canary test, where this would block until allow_score.set()).
        phases = await asyncio.wait_for(schedule.after_train_step(), 5.0)

        assert phases["continuous.weight_sync_barrier_mode"] == 1.0
        assert len(syncer.calls) == sync_calls_before + 1
        snapshot = await schedule._debug_snapshot()
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
    finally:
        collector.allow_score.set()
        await schedule.shutdown()


class _StaleSlotCollector(_Collector):
    """Generation always hits an evicted trainable-state slot (non-draining sync).

    Mirrors what the Ray executor raises when a request outlives its worker's slot
    window: a typed StaleSlotDiscard, NOT a generation failure.
    """

    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        raise StaleSlotDiscard("trainable-state slot evicted for policy_version=1")


@pytest.mark.asyncio
async def test_stale_slot_discard_counts_as_discard_not_error() -> None:
    # A StaleSlotDiscard from generation is an expected graceful discard under a
    # non-draining weight sync: the producer must count it as discarded_stale, not
    # as a collect error (which would inflate error_count and trip fail-fast).
    """Checks stale-slot discard increments discarded_stale_count, not error_count."""
    runtime = _Runtime()
    runtime.supports_non_draining_weight_sync = True
    collector = _StaleSlotCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(wait_timeout_s=30.0, fail_fast_errors=2),
        collector,
        syncer,
    )

    request = asyncio.create_task(
        schedule.next_iteration(["p0", "p1"], group_size=2),
    )
    try:
        # Let the background loop submit + harvest several stale-slot groups.
        snapshot = await _snapshot_when(
            schedule,
            lambda item: (
                item.producer_state is not None and item.producer_state.discarded_stale_count >= 2
            ),
        )
        # Discards never become errors, so fail-fast (2) is never tripped.
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.error_count == 0
        assert snapshot.producer_state.last_error is None
        assert snapshot.queue_stats["ready_items"] == 0
    finally:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prompt_set_update_swaps_producer_source() -> None:
    """Checks prompt set update swaps producer source."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)
        await schedule.next_iteration(["p0", "p2"], group_size=2)
        assert (await schedule._debug_snapshot()).prompts == ("p0", "p2")
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prompt_set_update_purges_old_ready_items_before_wait() -> None:
    """Old prompt-set ready items must not block admission for the new set."""
    runtime = _Runtime()
    schedule = _build(
        _continuous_config(
            max_stale_policy_versions=1,
            max_ready_groups=4,
            max_inflight_groups=2,
            wait_timeout_s=1.0,
        ),
        _Collector(runtime),
        _Syncer(runtime),
    )

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=1)
        await _snapshot_when(
            schedule,
            lambda item: item.queue_stats.get("ready_items", 0) >= 4,
        )
        await schedule.after_train_step()
        after_sync = await schedule._debug_snapshot()
        assert after_sync.queue_stats["ready_items"] >= 4

        second = await schedule.next_iteration(["p2", "p3"], group_size=1)

        assert sorted(batch.prompts[0] for batch in second.batches) == ["p2", "p3"]
        assert second.stats.as_phase_dict()["continuous.prompt_set_dropped"] >= 4.0
        assert (await schedule._debug_snapshot()).queue_stats["dropped_prompt_set"] >= 4.0
    finally:
        await schedule.shutdown()
