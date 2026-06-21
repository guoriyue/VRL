"""Scheduler contracts for the continuous producer/queue/consumer.

These pin the behaviors the trainer relies on:
- the ready queue only ever holds complete, reward-scored batches;
- the weight-sync drain waits for in-flight generation AND reward;
- items carry the policy version captured at submission time;
- the consumer honors the staleness bound and aggregates per-item
  collect phase timings into the iteration.
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
    """max_stale=0 (e.g. DiffusionNFT): a group whose policy was superseded mid
    flight is dropped at receipt, not queued for the consumer to drop later."""
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
    item_id: int,
    group_key: int,
    version: int | None,
    phase_times: dict[str, float] | None = None,
) -> ContinuousRolloutItem:
    from vrl.utils.stats import RolloutStats

    return ContinuousRolloutItem(
        item_id=item_id,
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
    queue.put(_item(0, group_key=0, version=1))
    queue.put(_item(1, group_key=1, version=1))

    iteration = await _drain(
        _consumer(queue, max_stale=1), min_groups=2, current_version=2,
    )

    assert iteration.policy_version == 1
    assert iteration.metadata["stale_policy_versions"] == 1
    assert iteration.metadata["consume_policy_version"] == 2


@pytest.mark.asyncio
async def test_consumer_drops_too_stale_items_and_times_out() -> None:
    """max_stale=0 must drop pre-sync items instead of training on them."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(0, group_key=0, version=1))
    queue.put(_item(1, group_key=1, version=1))

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
async def test_consumer_aggregates_item_phase_times() -> None:
    """Per-item collect timings sum into iteration.stats.as_phase_dict()."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(
        _item(
            0,
            group_key=0,
            version=1,
            phase_times={"collect.engine_generate": 1.5, "collect.reward_score": 0.5},
        ),
    )
    queue.put(
        _item(
            1,
            group_key=1,
            version=1,
            phase_times={"collect.engine_generate": 2.5},
        ),
    )

    iteration = await _drain(
        _consumer(queue, max_stale=0), min_groups=2, current_version=1,
    )

    assert iteration.stats.as_phase_dict()["collect.engine_generate"] == 4.0
    assert iteration.stats.as_phase_dict()["collect.reward_score"] == 0.5
    assert "continuous.queue_wait_s" in iteration.stats.as_phase_dict()
