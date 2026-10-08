"""Scheduler contracts for the continuous producer/batch/consumer.

These pin the behaviors the trainer relies on:
- result slots only hold complete, reward-scored batches;
- the weight-sync drain waits for in-flight generation AND reward;
- items carry the policy version captured at submission time;
- the consumer honors the staleness bound and aggregates per-item
  collect phase timings into the iteration.

The producer runs over the real stack: ``RolloutRuntimeCoordinator`` on the
real collector with the tiny SANA in-process runtime and a real trainer side,
so a "version bump" is a real weight push and every stored item is a real
scored batch. Generation and scoring are held at explicit gates ahead of the
real work so a test controls their timing.

Some cases use ``max_stale=0`` settings to pin the mechanism's exact boundary.
That value is intentionally unreachable from production continuous config,
which requires at least one stale policy version; zero-staleness execution uses
the strict-on-policy schedule.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import (
    CollectorBench,
    IndexReward,
    TrainerSide,
    real_collector,
    trainer_side,
)
from tests.rollouts.orchestration.continuous._helpers import _wait_until
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.rewards import RewardOutput, RewardSample
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.core import PromptCollectionCleanupError
from vrl.rollouts.orchestration.continuous.consumer import ContinuousRolloutConsumer
from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutSettings,
    PromptBatch,
    ScoredRollout,
)
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.stats import RolloutStats
from vrl.utils.lifecycle import RuntimeLifecycle


# A LoRA policy under continuous scheduling keeps versioned weight slots, so a
# request stamped with an older version still runs after a newer push: the
# shape the mid-flight version-bump cases need.
@dataclass
class _Stack:
    bench: CollectorBench
    trainer: TrainerSide
    coordinator: RolloutRuntimeCoordinator

    async def bump_version(self) -> int:
        """The trainer's weight push: the real export under the next version."""

        await self.coordinator.push_prepared_weights(
            self.coordinator.prepare_weight_sync_state(), RolloutStats()
        )
        return self.bench.runtime.current_policy_version


async def _stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    reward: IndexReward | None = None,
    versioned: bool = False,
) -> _Stack:
    bench = real_collector(monkeypatch, tmp_path, reward=reward, versioned_slots=versioned)
    trainer = trainer_side(bench, initialized=True)
    stack = _Stack(bench, trainer, trainer.coordinator(bench))
    # The initial push every run starts with: the runtime serves policy version 1.
    await stack.bump_version()
    return stack


class _GatedReward(IndexReward):
    """Scoring that waits at ``allow_score`` and records its completion."""

    def __init__(self) -> None:
        super().__init__()
        self.allow_score = asyncio.Event()
        self.allow_score.set()
        self.events: list[str] = []

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        await self.allow_score.wait()
        self.events.append("score_end")
        return await super().score_batch(samples)


class _GenerationGate:
    """Hold each real generation at ``allow``; record start/end on ``events``.

    The await sits where a Ray runtime awaits dispatch, ahead of the worker body.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, bench: CollectorBench, events: list[str]
    ) -> None:
        self.allow = asyncio.Event()
        self.allow.set()
        self.started = asyncio.Event()
        real = bench.runtime.generate

        async def generate(request: Any) -> Any:
            self.started.set()
            events.append("generate_start")
            await self.allow.wait()
            output = await real(request)
            events.append("generate_end")
            return output

        monkeypatch.setattr(bench.runtime, "generate", generate)


async def _gated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *, versioned: bool = False
) -> tuple[_Stack, _GatedReward, _GenerationGate]:
    reward = _GatedReward()
    stack = await _stack(monkeypatch, tmp_path, reward=reward, versioned=versioned)
    return stack, reward, _GenerationGate(monkeypatch, stack.bench, reward.events)


def _settings(
    *,
    max_inflight: int = 1,
    max_stale: int = 1,
    wait_timeout_s: float = 5.0,
    poll_interval_s: float = 0.001,
    fail_fast_errors: int = 3,
) -> ContinuousRolloutSettings:
    """Validated carrier for mechanism tests. Cases pin the exact zero-staleness
    boundary with ``max_stale=0``, which production config routes to
    strict_on_policy instead."""

    return ContinuousRolloutSettings(
        max_inflight_groups=max_inflight,
        max_stale_policy_versions=max_stale,
        wait_timeout_s=wait_timeout_s,
        queue_poll_interval_s=poll_interval_s,
        fail_fast_errors=fail_fast_errors,
        # Only the owner thread's weight sync reads this; the producer never does.
        versioned_weight_sync=False,
    )


def _producer(
    lifecycle: RolloutRuntimeCoordinator,
    *,
    max_stale: int = 0,
    prompts: list[str] | None = None,
    group_size: int = 2,
    runtime_debug: bool = False,
    max_inflight: int = 1,
    poll_interval_s: float = 0.001,
    fail_fast_errors: int = 3,
) -> ContinuousRolloutProducer:
    prompt_list = ["p0"] if prompts is None else list(prompts)
    producer = ContinuousRolloutProducer(
        lifecycle=lifecycle,
        settings=_settings(
            max_stale=max_stale,
            max_inflight=max_inflight,
            poll_interval_s=poll_interval_s,
            fail_fast_errors=fail_fast_errors,
        ),
    )
    producer.set_prompt_batch(
        prompt_list,
        group_size=group_size,
        runtime_debug=runtime_debug,
    )
    return producer


def _ready(results: list[ScoredRollout | None]) -> int:
    return sum(item is not None for item in results)


def _attempts(bench: CollectorBench) -> dict[str, int]:
    """Generation attempts per prompt, failed ones included."""

    return dict(Counter(request.prompts[0] for request in bench.trace.requests))


# ------------------------------------------------------------- batch results


@pytest.mark.asyncio
async def test_result_slot_is_filled_only_after_reward_scoring(monkeypatch, tmp_path) -> None:
    """A generated-but-unscored group must never appear in the batch results."""

    monkeypatch.setenv("VRL_PROFILE", "1")
    stack, reward, _gate = await _gated(monkeypatch, tmp_path)
    reward.allow_score.clear()
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await _wait_until(lambda: "generate_end" in reward.events)
        await asyncio.sleep(0.05)
        # Generation finished, reward still pending: nothing trainer-visible.
        assert _ready(results) == 0

        reward.allow_score.set()
        await _wait_until(lambda: _ready(results) >= 1)
        item = results[0]
        # The queued item is a complete, reward-scored, trainer-ready batch.
        assert item.batch.rewards is not None
        assert item.batch.rewards.numel() == 2
        assert item.batch.trajectory is not None
        # Collect phase timings rode along on the item, not on shared state.
        assert item.stats.as_metrics_dict()["collect.engine_generate"] > 0.0
        assert item.stats.as_metrics_dict()["collect.reward_score"] > 0.0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_failed_reward_scoring_leaves_result_slot_empty(monkeypatch, tmp_path) -> None:
    """Reward failures surface as producer errors, not as scored results."""

    stack = await _stack(monkeypatch, tmp_path)
    stack.bench.trace.fail("score", "reward model exploded", times=3)
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.error_count >= 2)
        assert _ready(results) == 0
        assert "reward model exploded" in str(producer.state.last_error)
        assert producer.state.completed_count == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_control_loop_failure_reaches_consumer_without_timeout(
    monkeypatch, tmp_path
) -> None:
    stack = await _stack(monkeypatch, tmp_path)
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None

    def fail_admission() -> None:
        raise RuntimeError("admission invariant broke")

    producer._admit = fail_admission
    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(max_stale=0, wait_timeout_s=60.0)

        with pytest.raises(RuntimeError, match="producer control loop failed") as caught:
            await consumer.collect_iteration(
                prompt_batch=prompt_batch,
                current_policy_version=1,
                producer_state=producer.state,
            )

        assert caught.value.__cause__ is producer.state.fatal_error
        assert "admission invariant broke" in str(caught.value.__cause__)
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_terminal_generation_error_is_not_retried_or_wrapped(monkeypatch, tmp_path) -> None:
    error = RayOperationTimeout(
        "rollout.generation.batch",
        1.0,
        context="request_id=req-terminal",
    )
    stack = await _stack(monkeypatch, tmp_path)
    stack.bench.trace.fail("generate", error)
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(max_stale=0, wait_timeout_s=60.0)

        with pytest.raises(RayOperationTimeout) as caught:
            await consumer.collect_iteration(
                prompt_batch=prompt_batch,
                current_policy_version=1,
                producer_state=producer.state,
            )

        assert caught.value is error
        assert producer.state.fatal_error is error
        assert producer.state.submitted_count == 1
        assert producer.inflight_count == 0
        assert _ready(results) == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_idle_sibling_failure_makes_next_collect_fatal_without_slot_retry(
    monkeypatch, tmp_path
) -> None:
    """A terminal failure between requests must poison the next slot immediately."""

    lifecycle = RuntimeLifecycle()
    sibling_failure = RayOperationTimeout("rollout.generation.batch", 0.5)
    lifecycle.fail(sibling_failure)
    stack = await _stack(monkeypatch, tmp_path)
    attempts = 0

    # The admission gate a Ray runtime holds ahead of every generate: a closed
    # lifecycle rejects the request before any worker is reached.
    async def closed_generate(request: Any) -> Any:
        nonlocal attempts
        attempts += 1
        lifecycle.require_running("generate")
        raise AssertionError("closed runtime accepted generation")

    monkeypatch.setattr(stack.bench.runtime, "generate", closed_generate)
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(max_stale=0, wait_timeout_s=60.0)

        with pytest.raises(RuntimeError, match="generate rejected") as caught:
            await consumer.collect_iteration(
                prompt_batch=prompt_batch,
                current_policy_version=1,
                producer_state=producer.state,
            )

        assert caught.value.__cause__ is sibling_failure
        assert attempts == 1
        assert producer.state.submitted_count == 1
        assert producer.state.error_count == 1
        assert producer.inflight_count == 0
        assert _ready(results) == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_cleanup_wrapper_around_terminal_error_is_not_retried(monkeypatch, tmp_path) -> None:
    timeout = RayOperationTimeout(
        "rollout.generation.batch",
        1.0,
        context="request_id=req-cleanup",
    )
    wrapped = PromptCollectionCleanupError(
        timeout,
        [RuntimeError("reward cleanup failed")],
    )
    stack = await _stack(monkeypatch, tmp_path)
    stack.bench.trace.fail("generate", wrapped)
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await _wait_until(lambda: producer.state.fatal_error is not None)
        consumer = _consumer(max_stale=0, wait_timeout_s=60.0)

        with pytest.raises(PromptCollectionCleanupError) as caught:
            await consumer.collect_iteration(
                prompt_batch=prompt_batch,
                current_policy_version=1,
                producer_state=producer.state,
            )

        assert caught.value is wrapped
        assert caught.value.root_cause is timeout
        assert producer.state.fatal_error is wrapped
        assert producer.state.submitted_count == 1
        assert producer.inflight_count == 0
        assert _ready(results) == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_finite_prompt_batch_completes_each_slot_once_then_idles(
    monkeypatch, tmp_path
) -> None:
    stack = await _stack(monkeypatch, tmp_path)
    producer = _producer(stack.coordinator, prompts=["p0", "p1", "p2"], max_inflight=2)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    producer.pause_admission()
    try:
        # The drain must admit p2 even though pause caught the batch with only
        # the first two slots live.
        await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert {item.group_slot for item in results if item is not None} == {0, 1, 2}
        assert _attempts(stack.bench) == {"p0": 1, "p1": 1, "p2": 1}
        assert producer.state.submitted_count == 3
        assert producer.state.completed_count == 3

        producer.resume_admission()
        await asyncio.sleep(0.02)
        assert _attempts(stack.bench) == {"p0": 1, "p1": 1, "p2": 1}
        assert producer.inflight_count == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_freezes_version_and_options_across_serial_retry(
    monkeypatch, tmp_path
) -> None:
    stack = await _stack(monkeypatch, tmp_path, versioned=True)
    stack.bench.trace.fail("generate", "transient failure for p0")
    producer = _producer(
        stack.coordinator,
        max_stale=1,
        prompts=["p0", "p1"],
        group_size=3,
        runtime_debug=True,
        poll_interval_s=60.0,
    )
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results
    # The trainer advances after the batch was installed at version 1.
    assert await stack.bump_version() == 2

    await producer.start()
    try:
        await producer.drain_prompt_batch(wait_timeout_s=5.0)

        requests = stack.bench.trace.requests
        assert _attempts(stack.bench) == {"p0": 2, "p1": 1}
        assert {request.policy_version for request in requests} == {1}
        assert {request.samples_per_prompt for request in requests} == {3}
        assert {request.runtime_debug for request in requests} == {True}
        assert producer.state.error_count == 1
        assert producer.state.submitted_count == 3
        assert producer.state.completed_count == 2
        items = {item.group_slot: item for item in results if item is not None}
        assert set(items) == {0, 1}
        # Identity contract: a retry keeps the same batch_id + group_slot.
        assert {item.batch_id for item in items.values()} == {0}
        # Each item carries its own admission-wait/service gauges (max-on-merge
        # views of the per-slot intervals).
        for item in items.values():
            assert item.stats.gauges["continuous.generation_queue_wait_s"] >= 0.0
            assert item.stats.gauges["continuous.generation_service_s"] >= 0.0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_rejects_replacing_incomplete_work(monkeypatch, tmp_path) -> None:
    stack, _reward, gate = await _gated(monkeypatch, tmp_path)
    gate.allow.clear()
    producer = _producer(stack.coordinator)

    await producer.start()
    producer.admit_now()
    try:
        await asyncio.wait_for(gate.started.wait(), 5.0)
        with pytest.raises(RuntimeError, match="cannot replace an incomplete"):
            producer.set_prompt_batch(
                ["p1"],
                group_size=2,
                runtime_debug=False,
            )
    finally:
        gate.allow.set()
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_rejects_replacing_unconsumed_ready_work(monkeypatch, tmp_path) -> None:
    stack = await _stack(monkeypatch, tmp_path)
    producer = _producer(stack.coordinator)

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
async def test_finite_prompt_batch_fails_after_one_slot_exhausts_retry_budget(
    monkeypatch, tmp_path
) -> None:
    stack = await _stack(monkeypatch, tmp_path)
    stack.bench.trace.fail("generate", "deterministic failure for p0", times=2)
    producer = _producer(stack.coordinator, poll_interval_s=0.001, fail_fast_errors=2)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    producer.pause_admission()
    try:
        with pytest.raises(RuntimeError, match="slot exceeded the failure budget") as caught:
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert "deterministic failure for p0" in str(caught.value.__cause__)
        assert _attempts(stack.bench) == {"p0": 2}
        assert _ready(results) == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_drain_times_out_when_collect_never_returns(
    monkeypatch, tmp_path
) -> None:
    stack, _reward, gate = await _gated(monkeypatch, tmp_path)
    gate.allow.clear()
    producer = _producer(stack.coordinator, poll_interval_s=0.001, fail_fast_errors=0)

    await producer.start()
    producer.pause_admission()
    try:
        with pytest.raises(TimeoutError, match="weight-sync barrier timed out"):
            await producer.drain_prompt_batch(wait_timeout_s=0.02)
    finally:
        gate.allow.set()
        await producer.stop()


@pytest.mark.asyncio
async def test_active_prompt_batch_fails_when_collect_is_cancelled(monkeypatch, tmp_path) -> None:
    stack = await _stack(monkeypatch, tmp_path)
    stack.bench.trace.fail("generate", asyncio.CancelledError())
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    producer.pause_admission()
    try:
        with pytest.raises(RuntimeError, match="prompt-batch collect was cancelled"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert producer.state.error_count == 1
        assert _ready(results) == 0
    finally:
        await producer.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["stop", "drain_prompt_batch"])
async def test_invalid_producer_timeout_preserves_running_work(
    monkeypatch, tmp_path, timeout_s: float, operation: str
) -> None:
    stack, reward, gate = await _gated(monkeypatch, tmp_path)
    gate.allow.clear()
    producer = _producer(stack.coordinator)
    await producer.start()
    producer.admit_now()
    await asyncio.wait_for(gate.started.wait(), 5.0)
    try:
        with pytest.raises(ValueError, match="wait_timeout_s must be finite and > 0"):
            await getattr(producer, operation)(wait_timeout_s=timeout_s)
        await asyncio.sleep(0)
        assert producer.state.running
        assert producer.inflight_count == 1
        gate.allow.set()
        await producer.drain_prompt_batch(wait_timeout_s=5.0)
        assert "score_end" in reward.events
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_producer_stop_does_not_wait_forever_for_cancel_suppression(
    monkeypatch, tmp_path
) -> None:
    stack = await _stack(monkeypatch, tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    real_generate = stack.bench.runtime.generate

    async def resistant_generate(request: Any) -> Any:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        try:
            return await real_generate(request)
        finally:
            finished.set()

    monkeypatch.setattr(stack.bench.runtime, "generate", resistant_generate)
    producer = _producer(stack.coordinator)

    await producer.start()
    producer.admit_now()
    await asyncio.wait_for(started.wait(), 5.0)
    await asyncio.wait_for(producer.stop(wait_timeout_s=0.02), 1.0)

    assert cancelled.is_set()
    assert producer.inflight_count == 0
    release.set()
    # The abandoned collect still finishes on its own; let it, so the worker
    # thread is idle when the test ends.
    await asyncio.wait_for(finished.wait(), 5.0)


# ------------------------------------------------------- weight-sync drain


@pytest.mark.asyncio
async def test_drain_prompt_batch_waits_for_generation_and_reward(monkeypatch, tmp_path) -> None:
    """The barrier drain returns only after gen + reward of in-flight work."""

    stack, reward, gate = await _gated(monkeypatch, tmp_path)
    gate.allow.clear()
    reward.allow_score.clear()
    producer = _producer(stack.coordinator)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await asyncio.wait_for(gate.started.wait(), 5.0)
        producer.pause_admission()
        barrier: list[str] = []

        async def _drain_then_sync() -> None:
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
            barrier.append("sync")

        drain_task = asyncio.create_task(_drain_then_sync())
        await asyncio.sleep(0.05)
        assert not barrier  # generation still running

        gate.allow.set()
        await _wait_until(lambda: "generate_end" in reward.events)
        await asyncio.sleep(0.05)
        assert not barrier  # reward still running

        reward.allow_score.set()
        await asyncio.wait_for(drain_task, 5.0)
        # Sync strictly after the full collect, and the drained group was
        # harvested into the batch results (drained, never dropped).
        assert reward.events == ["generate_start", "generate_end", "score_end"]
        assert barrier == ["sync"]
        assert _ready(results) == 1
    finally:
        producer.resume_admission()
        await producer.stop()


@pytest.mark.asyncio
async def test_late_reward_finishes_before_version_bump_under_draining(
    monkeypatch, tmp_path
) -> None:
    """OFF-POLICY INVARIANT, draining branch.

    A reward that completes *after* generation but is still in-flight at the
    weight-sync barrier must finish before the policy-version bump, so the group
    is never trained off-policy. This is the reward-late timing variant of
    ``test_drain_prompt_batch_waits_for_generation_and_reward``: here generation is
    already done and only scoring is outstanding when the barrier starts.
    ``schedule.after_train_step`` (non_draining=False) runs
    ``drain_prompt_batch`` -> ``sync_weights_after_train``, so reward(N) must
    complete strictly before the version advances to N+1.
    """

    stack, reward, _gate = await _gated(monkeypatch, tmp_path)
    # Generation finishes immediately; reward is the late phase still running
    # when the barrier opens.
    reward.allow_score.clear()
    producer = _producer(stack.coordinator, max_stale=0)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        # Generation has completed; the group is parked in reward scoring.
        await _wait_until(lambda: "generate_end" in reward.events)
        assert "score_end" not in reward.events  # reward still in flight
        assert _ready(results) == 0  # unscored work is never trainer-visible

        producer.pause_admission()
        order: list[str] = []

        async def _drain_then_bump() -> None:
            # Mirror schedule.after_train_step's draining branch exactly:
            # drain in-flight (gen + reward) BEFORE syncing/bumping the version.
            await producer.drain_prompt_batch(wait_timeout_s=5.0)
            order.append("drain_done")
            await stack.bump_version()  # the weight-sync push
            order.append("version_bumped")

        drain_task = asyncio.create_task(_drain_then_bump())
        await asyncio.sleep(0.05)
        # The reward is what is holding the barrier open: no bump yet.
        assert order == []
        assert "score_end" not in reward.events
        assert stack.bench.runtime.current_policy_version == 1

        reward.allow_score.set()
        await asyncio.wait_for(drain_task, 5.0)

        # Reward finished strictly before the version bump.
        assert reward.events == ["generate_start", "generate_end", "score_end"]
        assert order == ["drain_done", "version_bumped"]
        assert stack.bench.runtime.current_policy_version == 2
        # The fully-scored group is stored in its slot, stamped at the pre-bump
        # version it was generated under -- so it is on-policy for v1.
        assert _ready(results) == 1
        item = results[0]
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
async def test_items_carry_policy_version_captured_at_submission(monkeypatch, tmp_path) -> None:
    """A version bump mid-flight must not relabel an already-submitted group."""

    stack, _reward, gate = await _gated(monkeypatch, tmp_path, versioned=True)
    gate.allow.clear()
    # max_stale=1 so the mid-flight bump to v2 keeps the group inside the
    # freshness window (staleness 1 <= 1); this test pins version *stamping*,
    # not the freshness gate, so the group must survive to be inspected.
    producer = _producer(stack.coordinator, max_stale=1)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await asyncio.wait_for(gate.started.wait(), 5.0)
        producer.pause_admission()
        assert await stack.bump_version() == 2  # trainer syncs while the group is in flight
        gate.allow.set()
        await producer.drain_prompt_batch(wait_timeout_s=5.0)

        assert _ready(results) == 1
        assert results[0].rollout_policy_version == 1
        assert stack.bench.trace.requests[0].policy_version == 1
    finally:
        producer.resume_admission()
        await producer.stop()


# ----------------------------------------------- producer freshness gate


@pytest.mark.asyncio
async def test_prompt_batch_fails_when_group_is_stale_at_receipt(monkeypatch, tmp_path) -> None:
    """An active prompt batch cannot recover after its fixed version expires."""

    stack, _reward, gate = await _gated(monkeypatch, tmp_path, versioned=True)
    gate.allow.clear()
    producer = _producer(stack.coordinator, max_stale=0)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await asyncio.wait_for(gate.started.wait(), 5.0)
        producer.pause_admission()
        assert await stack.bump_version() == 2  # trainer advanced while the group was in flight
        gate.allow.set()
        with pytest.raises(RuntimeError, match="became stale before completion"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)

        # Generation completed, but the v1 group is stale=1 > 0 at receipt.
        assert _ready(results) == 0
        assert producer.state.completed_count == 1
    finally:
        producer.resume_admission()
        await producer.stop()


@pytest.mark.asyncio
async def test_prompt_batch_fails_when_group_is_past_stale_window(monkeypatch, tmp_path) -> None:
    """The gate respects the configured window, not any version change: with
    max_stale=1 a two-version-old group is still dropped."""

    stack, _reward, gate = await _gated(monkeypatch, tmp_path, versioned=True)
    gate.allow.clear()
    producer = _producer(stack.coordinator, max_stale=1)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    results = prompt_batch.results

    await producer.start()
    try:
        await asyncio.wait_for(gate.started.wait(), 5.0)
        producer.pause_admission()
        await stack.bump_version()
        assert await stack.bump_version() == 3  # two versions ahead: staleness 2 > 1
        gate.allow.set()
        with pytest.raises(RuntimeError, match="became stale before completion"):
            await producer.drain_prompt_batch(wait_timeout_s=5.0)

        assert _ready(results) == 0
    finally:
        producer.resume_admission()
        await producer.stop()


# ------------------------------------------------------------- consumer


def _batch(samples: int = 2) -> RolloutBatch:
    return RolloutBatch(
        rewards=torch.arange(samples, dtype=torch.float32),
        group_ids=torch.zeros(samples, dtype=torch.long),
    )


def _item(
    group_slot: int,
    version: int,
    phase_times: dict[str, float] | None = None,
    *,
    batch_id: int = 0,
) -> ScoredRollout:
    return ScoredRollout(
        batch_id=batch_id,
        group_slot=group_slot,
        rollout_policy_version=version,
        batch=_batch(),
        stats=RolloutStats(phase_seconds=dict(phase_times or {})),
    )


def _consumer(
    max_stale: int, *, wait_timeout_s: float = 1.0, poll_interval_s: float = 0.001
) -> ContinuousRolloutConsumer:
    return ContinuousRolloutConsumer(
        settings=_settings(
            max_stale=max_stale, wait_timeout_s=wait_timeout_s, poll_interval_s=poll_interval_s
        ),
    )


def _prompt_batch(
    results: list[ScoredRollout | None],
    *,
    batch_id: int = 0,
    version: int = 1,
) -> PromptBatch:
    return PromptBatch(
        batch_id=batch_id,
        policy_version=version,
        prompts=tuple(f"p{slot}" for slot in range(len(results))),
        group_size=2,
        runtime_debug=False,
        pending_slots=deque(slot for slot, item in enumerate(results) if item is None),
        pending_since={},
        results=results,
    )


def test_out_of_range_knobs_are_rejected_at_the_config_boundary() -> None:
    """Mechanisms trust settings validated by ContinuousRolloutConfig."""
    from vrl.trainers.core.types import ContinuousRolloutConfig

    for kwargs, message in (
        ({"max_inflight_groups": 0}, "max_inflight_groups"),
        ({"wait_timeout_s": 0.0}, "wait_timeout_s"),
        ({"queue_poll_interval_s": 0.0}, "queue_poll_interval_s"),
        ({"fail_fast_errors": -1}, "fail_fast_errors"),
    ):
        with pytest.raises(ValueError, match=message):
            ContinuousRolloutConfig(**kwargs)


async def _collect_iteration(
    consumer: ContinuousRolloutConsumer,
    prompt_batch: PromptBatch,
    *,
    current_policy_version: int,
):
    return await consumer.collect_iteration(
        prompt_batch=prompt_batch,
        current_policy_version=current_policy_version,
    )


@pytest.mark.parametrize("field, replacement", [("batch_id", 5), ("rollout_policy_version", 9)])
def test_scored_receipt_identity_cannot_change_after_completion(field, replacement) -> None:
    item = _item(group_slot=0, version=1)
    with pytest.raises(FrozenInstanceError):
        setattr(item, field, replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("ready_count", [0, 1])
async def test_consumer_timeout_identifies_incomplete_batch(ready_count) -> None:
    prompt_batch = _prompt_batch([None, None], batch_id=7)
    for slot in range(ready_count):
        prompt_batch.results[slot] = _item(group_slot=slot, version=1, batch_id=7)
    receipts = list(prompt_batch.results)

    with pytest.raises(TimeoutError) as caught:
        await _collect_iteration(
            _consumer(max_stale=1, wait_timeout_s=0.001),
            prompt_batch,
            current_policy_version=2,
        )

    message = str(caught.value)
    assert "prompt_batch_id=7" in message
    assert f"ready_groups={ready_count}/2" in message
    assert "current_policy_version=2" in message
    assert prompt_batch.results == receipts


@pytest.mark.asyncio
async def test_consumer_consumes_stale_items_within_bound() -> None:
    """max_stale=1 lets the trainer consume one-version-old groups.

    The consumer only reads the slots; releasing them is the producer's.
    """
    items = [_item(group_slot=slot, version=1) for slot in range(2)]
    prompt_batch = _prompt_batch(list(items))
    iteration = await _collect_iteration(
        _consumer(max_stale=1), prompt_batch, current_policy_version=2
    )

    phases = iteration.stats.as_metrics_dict()
    assert phases["continuous.rollout_policy_version"] == 1.0
    assert phases["continuous.stale_policy_versions"] == 1.0
    assert phases["continuous.consume_policy_version"] == 2.0
    assert prompt_batch.results == items


@pytest.mark.asyncio
async def test_consumer_rejects_a_too_stale_ready_batch() -> None:
    """A finite batch fails instead of dropping slots it cannot regenerate."""
    items = [_item(group_slot=slot, version=1) for slot in range(2)]
    prompt_batch = _prompt_batch(list(items))
    with pytest.raises(RuntimeError, match="older than the policy window"):
        await _collect_iteration(_consumer(max_stale=0), prompt_batch, current_policy_version=2)
    assert prompt_batch.results == items


@pytest.mark.asyncio
async def test_iteration_carries_batch_identity_gauges() -> None:
    prompt_batch = _prompt_batch(
        [_item(group_slot=slot, version=1, batch_id=3) for slot in range(2)], batch_id=3
    )
    iteration = await _collect_iteration(
        _consumer(max_stale=0), prompt_batch, current_policy_version=1
    )
    assert iteration.stats.as_metrics_dict()["continuous.batch_id"] == pytest.approx(3.0)


def test_item_ages_never_go_negative_under_a_skewed_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.rollouts.orchestration.continuous import types as continuous_types

    item = _item(group_slot=0, version=1)
    monkeypatch.setattr(continuous_types.time, "monotonic", lambda: item.completed_at - 10.0)
    assert item.age_s == 0.0


@pytest.mark.asyncio
async def test_consumer_rejects_future_policy_version() -> None:
    prompt_batch = _prompt_batch([_item(group_slot=0, version=2)], version=2)
    with pytest.raises(RuntimeError, match="newer than the trainer policy"):
        await _collect_iteration(_consumer(max_stale=0), prompt_batch, current_policy_version=1)


@pytest.mark.asyncio
async def test_late_reward_batch_fails_under_non_draining_max_stale_0() -> None:
    """The non-draining barrier must reject results older than its policy window."""
    prompt_batch = _prompt_batch([None])
    consumer = _consumer(max_stale=0)
    # The late reward lands after the trainer's bump, retaining the submit version.
    item = _item(group_slot=0, version=1)
    prompt_batch.results[0] = item
    with pytest.raises(RuntimeError, match="older than the policy window"):
        consumer.validate_ready_versions(prompt_batch=prompt_batch, current_policy_version=2)
    assert prompt_batch.results[0] is item
    with pytest.raises(RuntimeError, match="older than the policy window"):
        await _collect_iteration(consumer, prompt_batch, current_policy_version=2)


@pytest.mark.asyncio
async def test_consumer_aggregates_item_phase_times() -> None:
    prompt_batch = _prompt_batch(
        [
            _item(
                group_slot=0,
                version=1,
                phase_times={"collect.engine_generate": 1.5, "collect.reward_score": 0.5},
            ),
            _item(group_slot=1, version=1, phase_times={"collect.engine_generate": 2.5}),
        ]
    )
    iteration = await _collect_iteration(
        _consumer(max_stale=0), prompt_batch, current_policy_version=1
    )
    assert iteration.stats.as_metrics_dict()["collect.engine_generate"] == 4.0
    assert iteration.stats.as_metrics_dict()["collect.reward_score"] == 0.5
    assert "continuous.queue_wait_s" in iteration.stats.as_metrics_dict()


@pytest.mark.asyncio
async def test_consumer_waits_for_every_slot_and_preserves_prompt_order(
    monkeypatch, tmp_path
) -> None:
    stack = await _stack(monkeypatch, tmp_path, versioned=True)
    gates = {prompt: asyncio.Event() for prompt in ("p0", "p1", "next")}
    started: set[str] = set()
    real_generate = stack.bench.runtime.generate

    async def gated_generate(request: Any) -> Any:
        prompt = request.prompts[0]
        started.add(prompt)
        await gates[prompt].wait()
        return await real_generate(request)

    monkeypatch.setattr(stack.bench.runtime, "generate", gated_generate)
    producer = _producer(stack.coordinator, prompts=["p0", "p1"], max_inflight=2, max_stale=1)
    prompt_batch = producer.prompt_batch
    assert prompt_batch is not None
    await producer.start()
    # The trainer is one version ahead of the installed batch, inside the window.
    assert await stack.bump_version() == 2
    demand = asyncio.create_task(
        _collect_iteration(_consumer(max_stale=1), prompt_batch, current_policy_version=2)
    )
    try:
        await _wait_until(lambda: len(started) == 2)
        # Finish slot 1 first; slot 0 must still be awaited and retain first place.
        gates["p1"].set()
        await _wait_until(lambda: prompt_batch.results[1] is not None)
        assert prompt_batch.results[0] is None
        assert not demand.done()
        gates["p0"].set()
        iteration = await asyncio.wait_for(demand, 5.0)
        assert [batch.trajectory.sample_rows[0].prompt for batch in iteration.batches] == [
            "p0",
            "p1",
        ]
        assert [int(batch.group_ids[0]) for batch in iteration.batches] == [0, 1]
        assert iteration.stats.gauges["continuous.ready_groups_at_demand"] == 0
        # The owner releases the taken slots through the producer, which alone
        # writes them; only then can the next batch be installed.
        producer.release_results()
        assert prompt_batch.results == [None, None]
        assert prompt_batch.consumed is True
        producer.set_prompt_batch(["next"], group_size=2, runtime_debug=False)
        assert producer.prompt_batch.results == [None]
    finally:
        demand.cancel()
        await asyncio.gather(demand, return_exceptions=True)
        await producer.stop()


@pytest.mark.asyncio
async def test_consumer_polling_respects_remaining_wait_budget():
    consumer = _consumer(max_stale=1, wait_timeout_s=0.01, poll_interval_s=10.0)
    with pytest.raises(TimeoutError, match="continuous rollout consumer timed out"):
        await asyncio.wait_for(
            consumer.collect_iteration(
                prompt_batch=_prompt_batch([None]),
                current_policy_version=1,
            ),
            timeout=1.0,
        )
