"""Scheduler contracts for the continuous producer/queue/consumer.

These pin the behaviors the trainer relies on:
- the ready queue only ever holds complete, reward-scored batches;
- the weight-sync drain waits for in-flight generation AND reward;
- items carry the policy version captured at submission time;
- the consumer honors the staleness bound and aggregates per-item
  collect phase timings into the iteration.

Some cases use ``StalenessPolicy(0)`` to pin the mechanism's exact boundary.
That value is intentionally unreachable from production continuous config,
which requires at least one stale policy version; zero-staleness execution uses
the strict-on-policy schedule.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.rollouts.orchestration.continuous._helpers import _wait_until
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.consumer import ContinuousRolloutConsumer
from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.scheduler import RolloutScheduler
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem
from vrl.rollouts.orchestration.types import RolloutScheduleMode


def _batch(prompt: str, samples: int = 2) -> RolloutBatch:
    return RolloutBatch(
        observations=torch.zeros(samples, 1),
        actions=torch.zeros(samples, 1),
        rewards=torch.arange(samples, dtype=torch.float32),
        dones=torch.ones(samples, dtype=torch.bool),
        group_ids=torch.zeros(samples, dtype=torch.long),
        prompts=[prompt] * samples,
    )


@dataclass
class _Unscored:
    batch: RolloutBatch
    phases: dict[str, float]


class _GatedCollector:
    """Collector whose generation/reward phases block on explicit gates."""

    requires_runtime_offload_before_reward = False
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False

    def __init__(self) -> None:
        self.allow_generate = asyncio.Event()
        self.allow_generate.set()
        self.allow_score = asyncio.Event()
        self.allow_score.set()
        self.generation_started = asyncio.Event()
        self.events: list[str] = []

    async def collect_unscored(self, prompts: list[str], **kwargs: Any) -> _Unscored:
        self.generation_started.set()
        self.events.append("generate_start")
        await self.allow_generate.wait()
        self.events.append("generate_end")
        return _Unscored(
            batch=_batch(str(prompts[0]), int(kwargs["group_size"])),
            phases={"collect.engine_generate": 1.0},
        )

    async def score_rollouts(self, pendings: list[_Unscored]) -> list[RolloutBatch]:
        await self.allow_score.wait()
        self.events.append("score_end")
        pendings[0].phases["collect.reward_score"] = 0.5
        return [pending.batch for pending in pendings]


class _Lifecycle:
    """Minimal RolloutLifecycle stand-in for producer-level tests."""

    def __init__(self, collector: Any, version: int = 1) -> None:
        self.collector = collector
        self.version = version

    def current_policy_version(self) -> int | None:
        return self.version

    async def ensure_initial_weights(self, phase_times: dict[str, float]) -> None:
        del phase_times

    async def activate_rollout_runtime(self, phase_times: dict[str, float]) -> None:
        del phase_times


def _producer(
    collector: Any,
    queue: ContinuousRolloutQueue,
    *,
    lifecycle: _Lifecycle | None = None,
    max_stale: int = 0,
) -> ContinuousRolloutProducer:
    scheduler = RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=max_stale),
        max_inflight_groups=1,
        capacity=2,
        max_bytes=0,
        groups_per_iteration=1,
    )
    return ContinuousRolloutProducer(
        lifecycle=lifecycle or _Lifecycle(collector),
        prompts=["p0"],
        queue=queue,
        scheduler=scheduler,
        group_size=2,
        poll_interval_s=0.001,
    )


# ------------------------------------------------------------- ready queue


@pytest.mark.asyncio
async def test_ready_queue_gets_items_only_after_reward_scoring() -> None:
    """A generated-but-unscored group must never appear in the ready queue."""
    collector = _GatedCollector()
    collector.allow_score.clear()
    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(collector, queue)

    await producer.start()
    try:
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        await asyncio.sleep(0.05)
        # Generation finished, reward still pending: nothing trainer-visible.
        assert queue.size() == 0

        collector.allow_score.set()
        await _wait_until(lambda: queue.size() >= 1)
        item = queue._items[0]
        # The queued item is a complete, reward-scored, trainer-ready batch.
        assert item.batch.rewards is not None
        assert item.batch.rewards.numel() == 2
        # Collect phase timings rode along on the item, not on shared state.
        assert item.stats.as_phase_dict()["collect.engine_generate"] == 1.0
        assert item.stats.as_phase_dict()["collect.reward_score"] == 0.5
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_failed_reward_scoring_never_enqueues() -> None:
    """Reward failures surface as producer errors, not as queue items."""

    class _RewardBoom(_GatedCollector):
        async def score_rollouts(self, pendings: list[_Unscored]) -> list[RolloutBatch]:
            raise RuntimeError("reward model exploded")

    collector = _RewardBoom()
    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(collector, queue)

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.error_count >= 2)
        assert queue.size() == 0
        assert "reward model exploded" in str(producer.state.last_error)
        assert producer.state.completed_count == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_control_loop_failure_reaches_consumer_without_timeout() -> None:
    collector = _GatedCollector()
    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(collector, queue)

    def fail_admission() -> None:
        raise RuntimeError("admission invariant broke")

    producer._admit = fail_admission
    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(queue, max_stale=0)

        with pytest.raises(RuntimeError, match="producer control loop failed") as caught:
            await consumer.drain_for_iteration(
                rollout_id=0,
                min_groups=1,
                current_version=1,
                mode=RolloutScheduleMode.CONTINUOUS,
                wait_timeout_s=60.0,
                poll_interval_s=0.001,
                producer_state=producer.state,
            )

        assert caught.value.__cause__ is producer.state.fatal_error
        assert "admission invariant broke" in str(caught.value.__cause__)
    finally:
        await producer.stop()


# ------------------------------------------------------- weight-sync drain


@pytest.mark.asyncio
async def test_drain_inflight_waits_for_generation_and_reward() -> None:
    """The barrier drain returns only after gen + reward of in-flight work."""
    collector = _GatedCollector()
    collector.allow_generate.clear()
    collector.allow_score.clear()
    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(collector, queue)

    await producer.start()
    try:
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        producer.pause_admission()
        barrier: list[str] = []

        async def _drain_then_sync() -> None:
            await producer.drain_inflight()
            barrier.append("sync")

        drain_task = asyncio.create_task(_drain_then_sync())
        await asyncio.sleep(0.05)
        assert not barrier  # generation still running

        collector.allow_generate.set()
        await asyncio.sleep(0.05)
        assert not barrier  # reward still running

        collector.allow_score.set()
        await asyncio.wait_for(drain_task, 5.0)
        # Sync strictly after the full collect, and the drained group was
        # harvested into the ready queue (drained, never dropped).
        assert collector.events == ["generate_start", "generate_end", "score_end"]
        assert barrier == ["sync"]
        assert queue.size() == 1
    finally:
        producer.resume_admission()
        await producer.stop()


@pytest.mark.asyncio
async def test_late_reward_finishes_before_version_bump_under_draining() -> None:
    """OFF-POLICY INVARIANT, draining branch.

    A reward that completes *after* generation but is still in-flight at the
    weight-sync barrier must finish before the policy-version bump, so the group
    is never trained off-policy. This is the reward-late timing variant of
    ``test_drain_inflight_waits_for_generation_and_reward``: here generation is
    already done and only ``score_rollouts`` is outstanding when the barrier
    starts. ``schedule.after_train_step`` (non_draining=False) runs
    ``drain_inflight`` -> ``sync_weights_after_train``, so reward(N) must
    complete strictly before the version advances to N+1.
    """
    collector = _GatedCollector()
    # Generation finishes immediately; reward is the late phase still running
    # when the barrier opens.
    collector.allow_score.clear()
    queue = ContinuousRolloutQueue(max_items=4)
    lifecycle = _Lifecycle(collector, version=1)
    producer = _producer(collector, queue, lifecycle=lifecycle, max_stale=0)

    await producer.start()
    try:
        # Generation has completed; the group is parked in score_rollouts.
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        await _wait_until(lambda: "generate_end" in collector.events)
        assert "score_end" not in collector.events  # reward still in flight
        assert queue.size() == 0  # unscored work is never trainer-visible

        producer.pause_admission()
        order: list[str] = []

        async def _drain_then_bump() -> None:
            # Mirror schedule.after_train_step's draining branch exactly:
            # drain in-flight (gen + reward) BEFORE syncing/bumping the version.
            await producer.drain_inflight()
            order.append("drain_done")
            lifecycle.version = 2  # the weight-sync version bump
            order.append("version_bumped")

        drain_task = asyncio.create_task(_drain_then_bump())
        await asyncio.sleep(0.05)
        # The reward is what is holding the barrier open: no bump yet.
        assert order == []
        assert "score_end" not in collector.events
        assert lifecycle.version == 1

        collector.allow_score.set()
        await asyncio.wait_for(drain_task, 5.0)

        # Reward finished strictly before the version bump.
        assert collector.events == ["generate_start", "generate_end", "score_end"]
        assert order == ["drain_done", "version_bumped"]
        # The fully-scored group is in the queue, stamped at the pre-bump
        # version it was generated under -- so it is on-policy for v1.
        assert queue.size() == 1
        item = queue._items[0]
        assert item.rollout_policy_version == 1
        assert item.batch.rewards is not None
        assert item.batch.rewards.numel() == 2
        # A late reward cannot relabel the stamped version: reward is computed
        # from the rollout output by a frozen reward model, so the group's
        # policy version is fixed at generation time, never at scoring time.
        assert producer.state.discarded_stale_count == 0
    finally:
        producer.resume_admission()
        await producer.stop()


# ------------------------------------------------------- version stamping


@pytest.mark.asyncio
async def test_items_carry_policy_version_captured_at_submission() -> None:
    """A version bump mid-flight must not relabel an already-submitted group."""
    collector = _GatedCollector()
    collector.allow_generate.clear()
    queue = ContinuousRolloutQueue(max_items=4)
    lifecycle = _Lifecycle(collector, version=1)
    # max_stale=1 so the mid-flight bump to v2 keeps the group inside the
    # freshness window (staleness 1 <= 1); this test pins version *stamping*,
    # not the freshness gate, so the group must survive to be inspected.
    producer = _producer(collector, queue, lifecycle=lifecycle, max_stale=1)

    await producer.start()
    try:
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        producer.pause_admission()
        lifecycle.version = 2  # trainer syncs while the group is in flight
        collector.allow_generate.set()
        await producer.drain_inflight()

        assert queue.size() == 1
        assert queue._items[0].rollout_policy_version == 1
        assert producer.state.discarded_stale_count == 0
    finally:
        producer.resume_admission()
        await producer.stop()


# ----------------------------------------------- producer freshness gate


@pytest.mark.asyncio
async def test_producer_discards_group_too_stale_at_receipt() -> None:
    """At the mechanism-only zero bound, a superseded group drops at receipt."""
    collector = _GatedCollector()
    collector.allow_generate.clear()
    queue = ContinuousRolloutQueue(max_items=4)
    lifecycle = _Lifecycle(collector, version=1)
    producer = _producer(collector, queue, lifecycle=lifecycle, max_stale=0)

    await producer.start()
    try:
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        producer.pause_admission()
        lifecycle.version = 2  # trainer advanced while the group was in flight
        collector.allow_generate.set()
        await producer.drain_inflight()

        # Generation completed, but the v1 group is stale=1 > 0 at receipt.
        assert queue.size() == 0
        assert producer.state.discarded_stale_count == 1
        # Still counted as completed work — discarded is a subset of completed.
        assert producer.state.completed_count == 1
    finally:
        producer.resume_admission()
        await producer.stop()


@pytest.mark.asyncio
async def test_producer_discards_group_past_stale_window() -> None:
    """The gate respects the configured window, not any version change: with
    max_stale=1 a two-version-old group is still dropped."""
    collector = _GatedCollector()
    collector.allow_generate.clear()
    queue = ContinuousRolloutQueue(max_items=4)
    lifecycle = _Lifecycle(collector, version=1)
    producer = _producer(collector, queue, lifecycle=lifecycle, max_stale=1)

    await producer.start()
    try:
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        producer.pause_admission()
        lifecycle.version = 3  # two versions ahead: staleness 2 > 1
        collector.allow_generate.set()
        await producer.drain_inflight()

        assert queue.size() == 0
        assert producer.state.discarded_stale_count == 1
    finally:
        producer.resume_admission()
        await producer.stop()


# ------------------------------------------------------------- consumer


def _item(
    group_key: int,
    version: int | None,
    phase_times: dict[str, float] | None = None,
) -> ContinuousRolloutItem:
    from vrl.utils.stats import RolloutStats

    return ContinuousRolloutItem(
        group_key=group_key,
        rollout_policy_version=version,
        batch=_batch(f"p{group_key}"),
        stats=RolloutStats.from_phase_dict(phase_times),
    )


def _consumer(queue: ContinuousRolloutQueue, max_stale: int) -> ContinuousRolloutConsumer:
    scheduler = RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=max_stale),
        max_inflight_groups=1,
        capacity=max(1, queue.max_items),
        max_bytes=0,
        groups_per_iteration=1,
    )
    return ContinuousRolloutConsumer(queue=queue, scheduler=scheduler)


async def _drain(
    consumer: ContinuousRolloutConsumer,
    *,
    min_groups: int,
    current_version: int,
    timeout_s: float = 1.0,
):
    return await consumer.drain_for_iteration(
        rollout_id=0,
        min_groups=min_groups,
        current_version=current_version,
        mode=RolloutScheduleMode.CONTINUOUS,
        wait_timeout_s=timeout_s,
        poll_interval_s=0.001,
    )


@pytest.mark.asyncio
async def test_consumer_consumes_stale_items_within_bound() -> None:
    """max_stale=1 lets the trainer consume one-version-old groups."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=1))

    iteration = await _drain(
        _consumer(queue, max_stale=1),
        min_groups=2,
        current_version=2,
    )

    assert iteration.policy_version == 1
    assert iteration.metadata["stale_policy_versions"] == 1
    assert iteration.metadata["consume_policy_version"] == 2


@pytest.mark.asyncio
async def test_consumer_drops_too_stale_items_and_times_out() -> None:
    """max_stale=0 must drop pre-sync items instead of training on them."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=1))

    with pytest.raises(TimeoutError):
        await _drain(
            _consumer(queue, max_stale=0),
            min_groups=2,
            current_version=2,
            timeout_s=0.05,
        )
    assert queue.size() == 0
    assert queue.dropped_stale == 2


@pytest.mark.asyncio
async def test_late_reward_group_dropped_under_non_draining_max_stale_0() -> None:
    """OFF-POLICY INVARIANT, non-draining branch (max_stale=0).

    When the runtime advertises ``supports_non_draining_weight_sync``, the
    barrier skips ``drain_inflight`` and lets an in-flight collect (whose reward
    completes after the bump) finish concurrently with training. The group is
    version-stamped at submit time (the pre-bump version), so the post-sync
    ``drop_stale`` purge -- and ``select_iteration``'s own drop-stale -- remove
    it. It can NEVER reach a trained iteration, because reward is
    policy-independent and cannot relabel the stamped version to the new one.

    This drives the exact machinery the non-draining branch uses
    (``schedule._drop_stale_ready_items_after_sync`` -> ``scheduler.drop_stale``
    at schedule.py:252, then ``consumer.drain_for_iteration`` ->
    ``scheduler.select_iteration``).
    """
    queue = ContinuousRolloutQueue(max_items=8)
    scheduler = RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=0),
        max_inflight_groups=1,
        capacity=8,
        groups_per_iteration=1,
        max_bytes=0,
    )
    consumer = ContinuousRolloutConsumer(queue=queue, scheduler=scheduler)

    # The late-reward group: produced (and stamped) under v1, but its reward
    # only lands AFTER the trainer has bumped to v2 (the non-draining barrier
    # did not wait for it). It enters the ready queue stamped at v1.
    queue.put(_item(group_key=0, version=1))
    assert queue.size() == 1

    # Trainer is now at v2; the post-sync purge runs drop_stale at the new
    # version (exactly schedule._drop_stale_ready_items_after_sync's call).
    dropped = scheduler.drop_stale(queue, current_version=2)
    assert dropped == 1
    assert queue.size() == 0
    assert queue.dropped_stale == 1

    # Even if the purge had not run, the consumer's select would drop it: the
    # superseded v1 group never fills an iteration at current_version=2, so the
    # trainer never trains the late-reward group off-policy. With nothing fresh
    # queued, drain times out rather than yielding the stale group.
    with pytest.raises(TimeoutError):
        await _drain(consumer, min_groups=1, current_version=2, timeout_s=0.05)

    # Positive control: a FRESH v2 group (reward landed in time) is the one that
    # gets trained -- proving the timeout above is a drop, not an empty-queue
    # artifact, and that the machinery still admits in-window work.
    queue.put(_item(group_key=0, version=2))
    iteration = await _drain(consumer, min_groups=1, current_version=2)
    assert iteration.policy_version == 2
    assert queue.dropped_stale == 1  # the v1 late group, dropped exactly once


@pytest.mark.asyncio
async def test_consumer_aggregates_item_phase_times() -> None:
    """Per-item collect timings sum into iteration.stats.as_phase_dict()."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(
        _item(
            group_key=0,
            version=1,
            phase_times={"collect.engine_generate": 1.5, "collect.reward_score": 0.5},
        ),
    )
    queue.put(
        _item(
            group_key=1,
            version=1,
            phase_times={"collect.engine_generate": 2.5},
        ),
    )

    iteration = await _drain(
        _consumer(queue, max_stale=0),
        min_groups=2,
        current_version=1,
    )

    assert iteration.stats.as_phase_dict()["collect.engine_generate"] == 4.0
    assert iteration.stats.as_phase_dict()["collect.reward_score"] == 0.5
    assert "continuous.queue_wait_s" in iteration.stats.as_phase_dict()
