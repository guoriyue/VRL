"""Integration tests for the continuous rollout schedule."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.rollouts.orchestration.continuous._helpers import _wait_until
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration import (
    ContinuousRolloutSchedule,
    RolloutScheduleMode,
    build_rollout_schedule,
)


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


class _Collector:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        prompts = list(prompts)
        self.calls.append({"prompts": prompts, **dict(kwargs)})
        return _batch(prompts, int(kwargs.get("group_size", 1)))

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        return list(pendings)

    async def release_runtime_memory(self) -> None:
        self.calls.append({"release_runtime_memory": True})


def _continuous_config(**continuous: Any) -> SimpleNamespace:
    defaults = {
        "max_inflight_groups": 1,
        "max_ready_groups": 4,
        "max_ready_bytes_mb": 8192,
        "max_stale_policy_versions": 0,
        "drop_policy": "drop_oldest_stale",
        "wait_timeout_s": 5.0,
        "queue_poll_interval_s": 0.001,
    }
    defaults.update(continuous)
    return SimpleNamespace(
        mode="continuous",
        max_pending_rollouts=2,
        require_separate_gpus=True,
        weight_sync_barrier="pause_admission_and_drain_inflight",
        continuous=SimpleNamespace(**defaults),
    )


def _build(config: SimpleNamespace, collector: _Collector, syncer: _Syncer | None):
    initialized = {"value": False}

    def _set(value: bool) -> None:
        initialized["value"] = bool(value)

    return build_rollout_schedule(
        config,
        collector=collector,
        model=nn.Linear(1, 1),
        device=torch.device("cpu"),
        weight_syncer=syncer,
        sync_state_getter=(lambda: {"w": 1}) if syncer is not None else None,
        weights_initialized=lambda: initialized["value"],
        set_weights_initialized=_set,
    )


def test_factory_builds_continuous_schedule() -> None:
    """Checks factory builds continuous schedule."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))
    assert isinstance(schedule, ContinuousRolloutSchedule)
    assert schedule.mode is RolloutScheduleMode.CONTINUOUS


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
        assert all(
            b.context["rollout_policy_version"] == 1 for b in iteration.batches
        )
        assert "continuous.queue_wait_s" in iteration.phase_times
    finally:
        await schedule.producer.stop()


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
        assert schedule.producer.state.paused_for_weight_sync is False
        assert runtime.current_policy_version == 2

        second = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert second.policy_version == 2
        assert second.rollout_id == 1
        assert second.metadata["consume_policy_version"] == 2
        assert second.metadata["stale_policy_versions"] == 0
    finally:
        await schedule.producer.stop()


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
        await schedule.producer.stop()


@pytest.mark.asyncio
async def test_rejects_colocated_runtime() -> None:
    """Checks that rejects colocated runtime."""
    runtime = _Runtime()
    runtime.colocated = True
    schedule = _build(_continuous_config(), _Collector(runtime), None)

    with pytest.raises(RuntimeError, match="separate trainer and rollout GPU"):
        await schedule.next_iteration(["p0"], group_size=1)


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
        await schedule.producer.stop()


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
        assert schedule.queue.size() == 0
    finally:
        await schedule.producer.stop()


class _GatedScoreCollector(_Collector):
    """Reward scoring blocks until the test opens the gate."""

    def __init__(self, runtime: _Runtime) -> None:
        super().__init__(runtime)
        self.allow_score = asyncio.Event()
        self.allow_score.set()

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        await self.allow_score.wait()
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
        collector.allow_score.clear()
        await _wait_until(lambda: schedule.producer.state.inflight_count > 0)

        sync_calls_before = len(syncer.calls)
        barrier = asyncio.create_task(schedule.after_train_step())
        await asyncio.sleep(0.05)
        # Reward still in flight: admission paused, sync not yet performed.
        assert schedule.producer.state.paused_for_weight_sync is True
        assert len(syncer.calls) == sync_calls_before
        assert not barrier.done()

        collector.allow_score.set()
        await asyncio.wait_for(barrier, 5.0)
        assert len(syncer.calls) == sync_calls_before + 1
        assert schedule.producer.state.paused_for_weight_sync is False
        # The drained group was harvested with its pre-sync policy version.
        assert all(
            item.rollout_policy_version == 1 for item in schedule.queue._items
        )
    finally:
        await schedule.producer.stop()


@pytest.mark.asyncio
async def test_prompt_set_update_swaps_producer_source() -> None:
    """Checks prompt set update swaps producer source."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)
        await schedule.next_iteration(["p0", "p2"], group_size=2)
        assert schedule.producer.prompts == ["p0", "p2"]
    finally:
        await schedule.producer.stop()
