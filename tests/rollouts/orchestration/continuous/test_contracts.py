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
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.consumer import ContinuousRolloutConsumer
from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.scheduler import RolloutScheduler
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem
from vrl.rollouts.orchestration.prompt_collection import PromptCollectionCleanupError
from vrl.rollouts.orchestration.types import RolloutScheduleMode
from vrl.rollouts.stats import RolloutStats


def _batch(prompt: str, samples: int = 2) -> RolloutBatch:
    return RolloutBatch(
        rewards=torch.arange(samples, dtype=torch.float32),
        group_ids=torch.zeros(samples, dtype=torch.long),
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

    async def ensure_initial_weights(self, stats: RolloutStats) -> None:
        del stats

    async def activate_rollout_runtime(self, stats: RolloutStats) -> None:
        del stats


def _producer(
    collector: Any,
    queue: ContinuousRolloutQueue,
    *,
    lifecycle: _Lifecycle | None = None,
    max_stale: int = 0,
    prompts: list[str] | None = None,
    group_size: int = 2,
    runtime_debug: bool = False,
    max_inflight: int = 1,
    poll_interval_s: float = 0.001,
    fail_fast_errors: int = 3,
) -> ContinuousRolloutProducer:
    prompt_list = ["p0"] if prompts is None else list(prompts)
    scheduler = RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=max_stale),
        max_inflight_groups=max_inflight,
        max_bytes=queue.max_bytes,
    )
    producer = ContinuousRolloutProducer(
        lifecycle=lifecycle or _Lifecycle(collector),
        queue=queue,
        scheduler=scheduler,
        poll_interval_s=poll_interval_s,
        fail_fast_errors=fail_fast_errors,
    )
    producer.set_prompt_batch(
        prompt_list,
        group_size=group_size,
        runtime_debug=runtime_debug,
    )
    return producer


class _FiniteCollector:
    """Records prompt-batch inputs and optionally fails one attempt per prompt."""

    requires_runtime_offload_before_reward = False
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False

    def __init__(self, *, fail_once: set[str] | None = None) -> None:
        self.fail_once = set(fail_once or ())
        self.attempts: dict[str, int] = {}
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}
        self.calls: list[tuple[str, int | None, int, bool]] = []

    async def collect_unscored(self, prompts: list[Any], **kwargs: Any) -> _Unscored:
        prompt = str(getattr(prompts[0], "prompt", prompts[0]))
        attempt = self.attempts.get(prompt, 0) + 1
        self.attempts[prompt] = attempt
        self.active[prompt] = self.active.get(prompt, 0) + 1
        self.max_active[prompt] = max(
            self.max_active.get(prompt, 0),
            self.active[prompt],
        )
        self.calls.append(
            (
                prompt,
                kwargs.get("policy_version"),
                int(kwargs["group_size"]),
                bool(kwargs["runtime_debug"]),
            ),
        )
        try:
            await asyncio.sleep(0)
            if prompt in self.fail_once and attempt == 1:
                raise RuntimeError(f"transient failure for {prompt}")
            return _Unscored(
                batch=_batch(prompt, int(kwargs["group_size"])),
                phases={},
            )
        finally:
            self.active[prompt] -= 1

    async def score_rollouts(self, pendings: list[_Unscored]) -> list[RolloutBatch]:
        return [pending.batch for pending in pendings]


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


@pytest.mark.asyncio
async def test_terminal_generation_error_is_not_retried_or_wrapped() -> None:
    error = RayOperationTimeout(
        "rollout.generation.chunk",
        1.0,
        context="request_id=req-terminal",
    )

    class _TerminalCollector(_GatedCollector):
        async def collect_unscored(self, prompts: list[str], **kwargs: Any) -> _Unscored:
            del prompts, kwargs
            raise error

    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(_TerminalCollector(), queue)

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(queue, max_stale=0)

        with pytest.raises(RayOperationTimeout) as caught:
            await consumer.drain_for_iteration(
                rollout_id=0,
                min_groups=1,
                current_version=1,
                mode=RolloutScheduleMode.CONTINUOUS,
                wait_timeout_s=60.0,
                poll_interval_s=0.001,
                producer_state=producer.state,
            )

        assert caught.value is error
        assert producer.state.fatal_error is error
        assert producer.state.submitted_count == 1
        assert producer.state.inflight_count == 0
        assert queue.size() == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_cleanup_wrapper_around_terminal_error_is_not_retried() -> None:
    timeout = RayOperationTimeout(
        "rollout.generation.chunk",
        1.0,
        context="request_id=req-cleanup",
    )
    wrapped = PromptCollectionCleanupError(
        timeout,
        [RuntimeError("reward cleanup failed")],
    )

    class _TerminalCollector(_GatedCollector):
        async def collect_unscored(self, prompts: list[str], **kwargs: Any) -> _Unscored:
            del prompts, kwargs
            raise wrapped

    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(_TerminalCollector(), queue)

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(queue, max_stale=0)

        with pytest.raises(PromptCollectionCleanupError) as caught:
            await consumer.drain_for_iteration(
                rollout_id=0,
                min_groups=1,
                current_version=1,
                mode=RolloutScheduleMode.CONTINUOUS,
                wait_timeout_s=60.0,
                poll_interval_s=0.001,
                producer_state=producer.state,
            )

        assert caught.value is wrapped
        assert caught.value.root_cause is timeout
        assert producer.state.fatal_error is wrapped
        assert producer.state.submitted_count == 1
        assert producer.state.inflight_count == 0
        assert queue.size() == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_finite_prompt_batch_completes_each_slot_once_then_idles() -> None:
    collector = _FiniteCollector()
    queue = ContinuousRolloutQueue(max_items=4)
    producer = _producer(
        collector,
        queue,
        prompts=["p0", "p1", "p2"],
        max_inflight=2,
    )

    await producer.start()
    producer.pause_admission()
    try:
        # The drain must admit p2 even though pause caught the batch with only
        # the first two slots live.
        await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert {item.group_key for item in queue.snapshot()} == {0, 1, 2}
        assert collector.attempts == {"p0": 1, "p1": 1, "p2": 1}
        assert producer.state.submitted_count == 3
        assert producer.state.completed_count == 3

        producer.resume_admission()
        await asyncio.sleep(0.02)
        assert collector.attempts == {"p0": 1, "p1": 1, "p2": 1}
        assert producer.state.inflight_count == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_freezes_version_and_options_across_serial_retry() -> None:
    collector = _FiniteCollector(fail_once={"p0"})
    queue = ContinuousRolloutQueue(max_items=4)
    lifecycle = _Lifecycle(collector, version=7)
    producer = _producer(
        collector,
        queue,
        lifecycle=lifecycle,
        max_stale=1,
        prompts=["p0", "p1"],
        group_size=3,
        runtime_debug=True,
        poll_interval_s=60.0,
    )
    lifecycle.version = 8

    await producer.start()
    try:
        await producer.drain_prompt_batch(wait_timeout_s=5.0)

        assert collector.attempts == {"p0": 2, "p1": 1}
        assert collector.max_active == {"p0": 1, "p1": 1}
        assert {version for _, version, _, _ in collector.calls} == {7}
        assert {group_size for _, _, group_size, _ in collector.calls} == {3}
        assert {debug for _, _, _, debug in collector.calls} == {True}
        assert producer.state.error_count == 1
        assert producer.state.submitted_count == 3
        assert producer.state.completed_count == 2
        assert {item.group_key for item in queue.snapshot()} == {0, 1}
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_rejects_replacing_incomplete_work() -> None:
    collector = _GatedCollector()
    collector.allow_generate.clear()
    producer = _producer(collector, ContinuousRolloutQueue(max_items=2))

    await producer.start()
    producer.admit_now()
    try:
        await asyncio.wait_for(collector.generation_started.wait(), 5.0)
        with pytest.raises(RuntimeError, match="cannot replace an incomplete"):
            producer.set_prompt_batch(
                ["p1"],
                group_size=2,
                runtime_debug=False,
            )
    finally:
        collector.allow_generate.set()
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_rejects_replacing_unconsumed_ready_work() -> None:
    queue = ContinuousRolloutQueue(max_items=2)
    producer = _producer(_FiniteCollector(), queue)

    await producer.start()
    try:
        await producer.drain_prompt_batch(wait_timeout_s=5.0)
        with pytest.raises(RuntimeError, match="before its ready items are consumed"):
            producer.set_prompt_batch(
                ["p1"],
                group_size=2,
                runtime_debug=False,
            )
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_finite_prompt_batch_fails_when_queue_hard_cap_evicts_a_slot() -> None:
    collector = _FiniteCollector()
    queue = ContinuousRolloutQueue(max_items=2, max_bytes=1)
    producer = _producer(
        collector,
        queue,
        max_stale=1,
        poll_interval_s=60.0,
    )

    await producer.start()
    try:
        with pytest.raises(RuntimeError, match="hard cap evicted prompt-batch"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_finite_prompt_batch_fails_after_one_slot_exhausts_retry_budget() -> None:
    class _AlwaysFailCollector(_FiniteCollector):
        async def collect_unscored(self, prompts: list[Any], **kwargs: Any) -> _Unscored:
            prompt = str(getattr(prompts[0], "prompt", prompts[0]))
            self.attempts[prompt] = self.attempts.get(prompt, 0) + 1
            raise RuntimeError(f"deterministic failure for {prompt}")

    collector = _AlwaysFailCollector()
    queue = ContinuousRolloutQueue(max_items=2)
    producer = _producer(
        collector,
        queue,
        poll_interval_s=0.001,
        fail_fast_errors=2,
    )

    await producer.start()
    producer.pause_admission()
    try:
        with pytest.raises(RuntimeError, match="slot exceeded the failure budget") as caught:
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert "deterministic failure for p0" in str(caught.value.__cause__)
        assert collector.attempts == {"p0": 2}
        assert queue.size() == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_drain_times_out_when_collect_never_returns() -> None:
    collector = _GatedCollector()
    collector.allow_generate.clear()
    queue = ContinuousRolloutQueue(max_items=2)
    producer = _producer(
        collector,
        queue,
        poll_interval_s=0.001,
        fail_fast_errors=0,
    )

    await producer.start()
    producer.pause_admission()
    try:
        with pytest.raises(TimeoutError, match="weight-sync barrier timed out"):
            await producer.drain_prompt_batch(wait_timeout_s=0.02)
    finally:
        collector.allow_generate.set()
        await producer.stop()


@pytest.mark.asyncio
async def test_active_prompt_batch_fails_when_collect_is_cancelled() -> None:
    class _CancelledCollector(_FiniteCollector):
        async def collect_unscored(self, prompts: list[Any], **kwargs: Any) -> _Unscored:
            del prompts, kwargs
            raise asyncio.CancelledError

    queue = ContinuousRolloutQueue(max_items=2)
    producer = _producer(_CancelledCollector(), queue)

    await producer.start()
    producer.pause_admission()
    try:
        with pytest.raises(RuntimeError, match="prompt-batch collect was cancelled"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert producer.state.error_count == 1
        assert queue.size() == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_producer_stop_does_not_wait_forever_for_cancel_suppression() -> None:
    class _CancellationResistantCollector(_FiniteCollector):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def collect_unscored(self, prompts: list[Any], **kwargs: Any) -> _Unscored:
            self.started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
            return await super().collect_unscored(prompts, **kwargs)

    collector = _CancellationResistantCollector()
    producer = _producer(collector, ContinuousRolloutQueue(max_items=2))

    await producer.start()
    producer.admit_now()
    await asyncio.wait_for(collector.started.wait(), 5.0)
    await asyncio.wait_for(producer.stop(wait_timeout_s=0.02), 1.0)

    assert collector.cancelled.is_set()
    assert producer.state.inflight_count == 0
    collector.release.set()
    await asyncio.sleep(0)


# ------------------------------------------------------- weight-sync drain


@pytest.mark.asyncio
async def test_drain_prompt_batch_waits_for_generation_and_reward() -> None:
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
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
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
    ``test_drain_prompt_batch_waits_for_generation_and_reward``: here generation is
    already done and only ``score_rollouts`` is outstanding when the barrier
    starts. ``schedule.after_train_step`` (non_draining=False) runs
    ``drain_prompt_batch`` -> ``sync_weights_after_train``, so reward(N) must
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
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
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
        await producer.drain_prompt_batch(wait_timeout_s=5.0)

        assert queue.size() == 1
        assert queue._items[0].rollout_policy_version == 1
    finally:
        producer.resume_admission()
        await producer.stop()


# ----------------------------------------------- producer freshness gate


@pytest.mark.asyncio
async def test_prompt_batch_fails_when_group_is_stale_at_receipt() -> None:
    """An active prompt batch cannot recover after its fixed version expires."""
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
        with pytest.raises(RuntimeError, match="became stale before completion"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)

        # Generation completed, but the v1 group is stale=1 > 0 at receipt.
        assert queue.size() == 0
        assert producer.state.completed_count == 1
    finally:
        producer.resume_admission()
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_fails_when_group_is_past_stale_window() -> None:
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
        with pytest.raises(RuntimeError, match="became stale before completion"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)

        assert queue.size() == 0
    finally:
        producer.resume_admission()
        await producer.stop()


# ------------------------------------------------------------- consumer


def _item(
    group_key: int,
    version: int | None,
    phase_times: dict[str, float] | None = None,
) -> ContinuousRolloutItem:
    return ContinuousRolloutItem(
        group_key=group_key,
        rollout_policy_version=version,
        batch=_batch(f"p{group_key}"),
        stats=RolloutStats(phase_seconds=dict(phase_times or {})),
    )


def _consumer(queue: ContinuousRolloutQueue, max_stale: int) -> ContinuousRolloutConsumer:
    scheduler = RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=max_stale),
        max_inflight_groups=1,
        max_bytes=0,
    )
    return ContinuousRolloutConsumer(queue=queue, scheduler=scheduler, fail_fast_errors=3)


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
async def test_consumer_rejects_a_too_stale_ready_batch() -> None:
    """A finite batch fails instead of dropping slots it cannot regenerate."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=1))

    with pytest.raises(RuntimeError, match="older than the policy window"):
        await _drain(
            _consumer(queue, max_stale=0),
            min_groups=2,
            current_version=2,
        )
    assert queue.size() == 2


@pytest.mark.asyncio
async def test_late_reward_batch_fails_under_non_draining_max_stale_0() -> None:
    """OFF-POLICY INVARIANT, non-draining branch (max_stale=0).

    When the runtime advertises ``supports_non_draining_weight_sync``, the
    barrier skips ``drain_prompt_batch`` and lets an in-flight collect (whose reward
    completes after the bump) finish concurrently with training. The group is
    version-stamped at submit time (the pre-bump version), so the post-sync
    ready-version validation rejects it. It can NEVER reach a trained
    iteration, because reward is policy-independent and cannot relabel the
    stamped version to the new one.

    This drives the exact machinery the non-draining owner branch uses
    (``scheduler.validate_ready_versions`` after weight sync, then
    ``consumer.drain_for_iteration`` -> ``scheduler.select_iteration``).
    """
    queue = ContinuousRolloutQueue(max_items=8)
    scheduler = RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=0),
        max_inflight_groups=1,
        max_bytes=0,
    )
    consumer = ContinuousRolloutConsumer(queue=queue, scheduler=scheduler, fail_fast_errors=3)

    # The late-reward group: produced (and stamped) under v1, but its reward
    # only lands AFTER the trainer has bumped to v2 (the non-draining barrier
    # did not wait for it). It enters the ready queue stamped at v1.
    queue.put(_item(group_key=0, version=1))
    assert queue.size() == 1

    # Trainer is now at v2. Both the post-sync owner check and consumer selection
    # fail immediately with the fixed-version cause instead of deleting one slot
    # and waiting for a batch that can no longer complete.
    with pytest.raises(RuntimeError, match="older than the policy window"):
        scheduler.validate_ready_versions(queue, current_version=2)
    assert queue.size() == 1
    with pytest.raises(RuntimeError, match="older than the policy window"):
        await _drain(consumer, min_groups=1, current_version=2)


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
