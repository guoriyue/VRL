"""The continuous rollout schedule over the real coordinator, collector and runtime.

Every schedule here is ``build_rollout_schedule`` on the real
``RolloutCollector`` (tiny SANA family, ``InProcessGenerationRuntime``), the
real trainer side (replay bundle, ``SingleProcessStrategy``) and the real
``RayRuntimeWeightSyncer``. Versioned slots (non-draining weight sync) come
from the run's own launch contract: a LoRA continuous run on this family.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import (
    CollectorBench,
    IndexReward,
    real_collector,
    trainer_side,
)
from tests.rollouts.orchestration.continuous._helpers import (
    owner_snapshot,
    wait_for_owner_progress,
)
from vrl.generation.execution.types import StaleSlotDiscard
from vrl.rewards import RewardOutput, RewardSample
from vrl.rollouts.orchestration import ContinuousRolloutSchedule, build_rollout_schedule
from vrl.trainers.data.prompts import PromptExample


def _continuous_config(**continuous: Any) -> SimpleNamespace:
    from vrl.trainers.core.types import ContinuousRolloutConfig

    settings = {"wait_timeout_s": 5.0, "queue_poll_interval_s": 0.001}
    settings.update(continuous)
    return SimpleNamespace(
        schedule_mode="continuous",
        continuous=ContinuousRolloutConfig(**settings),
    )


def _iteration_stat(iteration: Any, name: str) -> float:
    return iteration.stats.as_metrics_dict()[name]


def _bench(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    non_draining: bool = False,
    reward: Any = None,
) -> CollectorBench:
    bench = real_collector(monkeypatch, tmp_path, reward=reward, versioned_slots=non_draining)
    bench.trace.watch(bench.collector, "shutdown", "collector_shutdown")
    return bench


def _build(
    config: SimpleNamespace,
    bench: CollectorBench,
    *,
    initially_initialized: bool = False,
) -> ContinuousRolloutSchedule:
    trainer = trainer_side(bench, initialized=initially_initialized)
    return build_rollout_schedule(
        config,
        trainer.coordinator(bench),
        versioned_weight_sync=bench.stack.resolved.built.trainer.versioned_weight_sync,
    )


def _delay_generation(
    monkeypatch: pytest.MonkeyPatch, bench: CollectorBench, seconds: float
) -> None:
    """Add dispatch latency ahead of each real generation, like a remote engine."""

    real = bench.runtime.generate

    async def generate(request: Any) -> Any:
        await asyncio.sleep(seconds)
        return await real(request)

    monkeypatch.setattr(bench.runtime, "generate", generate)


def _pushes(bench: CollectorBench) -> int:
    return bench.trace.events.count("update_weights")


async def _snapshot_when(
    schedule: ContinuousRolloutSchedule,
    condition: Callable[[Any], bool],
    *,
    timeout_s: float = 5.0,
) -> Any:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        snapshot = await owner_snapshot(schedule._rollout_thread)
        if condition(snapshot):
            return snapshot
        if loop.time() >= deadline:
            raise AssertionError("owner snapshot condition not reached before timeout")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_shutdown_joins_owner_and_is_idempotent(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    await schedule.next_iteration(["p0"], group_size=1)
    running = await owner_snapshot(schedule._rollout_thread)
    assert running.producer_state is not None
    assert running.producer_state.running is True

    await asyncio.gather(schedule.shutdown(), schedule.shutdown())
    await schedule.shutdown()

    assert bench.trace.events.count("collector_shutdown") == 1
    assert schedule._rollout_thread._stopped.is_set()
    thread = schedule._rollout_thread._thread
    assert thread is not None and not thread.is_alive()


@pytest.mark.asyncio
async def test_shutdown_failure_retries_cleanup_before_closing_owner(
    monkeypatch, tmp_path
) -> None:
    bench = _bench(monkeypatch, tmp_path)
    bench.trace.fail("collector_shutdown", "collector cleanup failed")
    schedule = _build(_continuous_config(), bench)

    await schedule.next_iteration(["p0"], group_size=1)

    with pytest.raises(RuntimeError, match="collector cleanup failed"):
        await schedule.shutdown()

    thread = schedule._rollout_thread._thread
    assert bench.trace.events.count("collector_shutdown") == 1
    assert schedule._rollout_thread._closed is False
    assert thread is not None and thread.is_alive()

    await schedule.shutdown()
    await schedule.shutdown()

    assert bench.trace.events.count("collector_shutdown") == 2
    assert schedule._rollout_thread._closed is True
    assert thread is not None and not thread.is_alive()


@pytest.mark.asyncio
async def test_owner_production_advances_while_trainer_loop_is_blocked(
    monkeypatch, tmp_path
) -> None:
    bench = _bench(monkeypatch, tmp_path)
    _delay_generation(monkeypatch, bench, 0.02)
    schedule = _build(_continuous_config(), bench)

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=["p1"],
        )
        before = await owner_snapshot(schedule._rollout_thread)
        assert before.producer_state is not None
        assert before.producer_state.submitted_count == 2

        # Blocks the trainer asyncio loop exactly like synchronous backward,
        # until the owner has completed one more item on its own thread.
        after = wait_for_owner_progress(
            schedule._rollout_thread,
            completed_above=before.producer_state.completed_count,
        )
        assert after.producer_state is not None
        assert after.producer_state.tick_count > before.producer_state.tick_count
        assert after.producer_state.submitted_count == before.producer_state.submitted_count
        assert after.producer_state.completed_count > before.producer_state.completed_count
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_initialized_runtime_does_not_receive_redundant_initial_push(
    monkeypatch, tmp_path
) -> None:
    """A resumed run: the rollout already serves the checkpoint's version and the
    trainer knows its weights are out, so the first iteration pushes nothing."""

    bench = _bench(monkeypatch, tmp_path)
    trainer = trainer_side(bench, initialized=True)
    await bench.runtime.update_weights(trainer.export(), 7)
    pushes_before = _pushes(bench)
    schedule = build_rollout_schedule(
        _continuous_config(),
        trainer.coordinator(bench),
        versioned_weight_sync=bench.stack.resolved.built.trainer.versioned_weight_sync,
    )

    try:
        iteration = await schedule.next_iteration(["p0"], group_size=1)

        assert _iteration_stat(iteration, "continuous.rollout_policy_version") == 7.0
        assert bench.runtime.current_policy_version == 7
        assert _pushes(bench) == pushes_before
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_reset_reuses_committed_runtime_weights_without_version_bump(
    monkeypatch, tmp_path
) -> None:
    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        first = await schedule.next_iteration(["p0"], group_size=1)
        assert _iteration_stat(first, "continuous.rollout_policy_version") == 1.0
        assert _pushes(bench) == 1

        schedule.reset()
        second = await schedule.next_iteration(["p0"], group_size=1)

        assert _iteration_stat(second, "continuous.rollout_policy_version") == 1.0
        assert _pushes(bench) == 1
        assert "collector_shutdown" not in bench.trace.events
    finally:
        await schedule.shutdown()


def test_continuous_rejects_zero_window(monkeypatch, tmp_path) -> None:
    """Zero-window execution belongs to strict_on_policy, not continuous."""

    bench = _bench(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=r"max_stale_policy_versions.*>= 1"):
        _build(
            _continuous_config(max_stale_policy_versions=0),
            bench,
        )


@pytest.mark.asyncio
async def test_continuous_drains_full_homogeneous_iteration(monkeypatch, tmp_path) -> None:
    """A homogeneous continuous iteration drains the full set: one policy version for rollout and
    consume, zero staleness, distinct group ids, and the item-age / ready-groups / queue-wait
    phases all reported.
    """

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        iteration = await schedule.next_iteration(["p0", "p1"], group_size=2)

        # Full set, one fresh policy version, distinct group ids 0..1.
        phases = iteration.stats.as_metrics_dict()
        assert phases["continuous.rollout_policy_version"] == 1.0
        assert phases["continuous.consume_policy_version"] == 1.0
        assert phases["continuous.stale_policy_versions"] == 0.0
        assert phases["continuous.item_age_s"] >= 0.0
        assert phases["continuous.ready_groups_at_demand"] >= 0.0
        assert len(iteration.batches) == 2
        assert sum(batch.rewards.numel() for batch in iteration.batches) == 4
        group_ids = sorted(int(b.group_ids[0]) for b in iteration.batches)
        assert group_ids == [0, 1]
        assert "continuous.queue_wait_s" in phases
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_weight_sync_barrier_advances_version_and_resumes(monkeypatch, tmp_path) -> None:
    """``after_train_step`` performs exactly one weight sync, bumps the runtime policy version and
    unpauses admission; the next iteration is produced and consumed at the new version with
    zero staleness.
    """

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        first = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert _iteration_stat(first, "continuous.rollout_policy_version") == 1.0

        pushes_before = _pushes(bench)
        await schedule.after_train_step()
        # Barrier performed exactly one post-train sync and resumed admission.
        assert _pushes(bench) == pushes_before + 1
        snapshot = await owner_snapshot(schedule._rollout_thread)
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
        assert bench.runtime.current_policy_version == 2

        second = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert _iteration_stat(second, "continuous.rollout_policy_version") == 2.0
        assert _iteration_stat(second, "continuous.consume_policy_version") == 2.0
        assert _iteration_stat(second, "continuous.stale_policy_versions") == 0.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_partial_commit_failure_closes_admission_and_runtime(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        await schedule.next_iteration(["p0"], group_size=1)

        bench.trace.fail("update_weights", "worker install ACK mismatch")
        with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
            await schedule.after_train_step()

        requests_after_failure = len(bench.trace.requests)
        failed = await owner_snapshot(schedule._rollout_thread)
        assert failed.producer_state is None
        assert failed.batch_stats == {}
        assert "worker install ACK mismatch" in str(failed.terminal_error)
        assert bench.trace.events.count("collector_shutdown") == 1

        await asyncio.sleep(0.02)
        assert len(bench.trace.requests) == requests_after_failure
        with pytest.raises(RuntimeError, match="owner has failed"):
            await schedule.next_iteration(["p0"], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_draining_sync_finishes_the_active_prompt_batch_before_commit(
    monkeypatch, tmp_path
) -> None:
    """A draining backend completes every finite prefetch slot at one version."""

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(max_stale_policy_versions=1), bench)

    try:
        first = await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p2", "p3"],
        )
        assert _iteration_stat(first, "continuous.rollout_policy_version") == 1.0
        await schedule.after_train_step()
        assert bench.runtime.current_policy_version == 2
        after = await owner_snapshot(schedule._rollout_thread)
        assert after.batch_stats["ready_items"] == 2

        second = await schedule.next_iteration(["p2", "p3"], group_size=2)
        assert _iteration_stat(second, "continuous.rollout_policy_version") == 1.0
        assert _iteration_stat(second, "continuous.stale_policy_versions") == 1.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_result_slots_fit_the_finite_prompt_batch(monkeypatch, tmp_path) -> None:
    """Every prompt group must reach the iteration even with serial admission."""

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        iteration = await schedule.next_iteration(["a", "b", "c"], group_size=2)
        assert len(iteration.batches) == 3
        assert sum(batch.rewards.numel() for batch in iteration.batches) == 6
        assert sorted(int(b.group_ids[0]) for b in iteration.batches) == [0, 1, 2]
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_persistent_producer_failure_fails_fast_with_root_cause(
    monkeypatch, tmp_path
) -> None:
    # Every generation fails. The consumer must surface the producer's root
    # cause well before the (long) wait timeout, not an opaque timeout.
    bench = _bench(monkeypatch, tmp_path)
    bench.trace.fail("generate", "reward model OOM", times=50)
    schedule = _build(_continuous_config(wait_timeout_s=30.0, fail_fast_errors=2), bench)

    try:
        with pytest.raises(RuntimeError, match="reward model OOM") as excinfo:
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert "failing every generation" in str(excinfo.value)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_reward_failure_fails_fast_and_never_reaches_queue(monkeypatch, tmp_path) -> None:
    # Reward scoring (not generation) fails persistently: the consumer must
    # surface that root cause and the result slots must stay empty.
    bench = _bench(monkeypatch, tmp_path)
    bench.trace.fail("score", "reward model exploded", times=50)
    schedule = _build(_continuous_config(wait_timeout_s=30.0, fail_fast_errors=2), bench)

    try:
        with pytest.raises(RuntimeError, match="reward model exploded"):
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert (await owner_snapshot(schedule._rollout_thread)).batch_stats == {}
    finally:
        await schedule.shutdown()


class _GatedReward(IndexReward):
    """Scoring that blocks, after a number of calls, until the test opens the gate."""

    def __init__(self) -> None:
        super().__init__()
        self.allow_score = threading.Event()
        self.allow_score.set()
        self.score_blocked = threading.Event()
        self.block_after_scores = 0
        self.score_calls = 0

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        self.score_calls += 1
        if self.score_calls > self.block_after_scores and not self.allow_score.is_set():
            self.score_blocked.set()
            await asyncio.to_thread(self.allow_score.wait)
        return await super().score_batch(samples)


@pytest.mark.asyncio
async def test_weight_sync_waits_for_inflight_reward(monkeypatch, tmp_path) -> None:
    """after_train_step must drain in-flight generation+reward before pushing
    weights; syncing earlier would mix two policies inside one request."""

    reward = _GatedReward()
    bench = _bench(monkeypatch, tmp_path, reward=reward)
    schedule = _build(_continuous_config(), bench)

    try:
        reward.block_after_scores = 2
        reward.score_blocked.clear()
        reward.allow_score.clear()
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p2", "p3"],
        )
        assert await asyncio.to_thread(reward.score_blocked.wait, 5.0)

        pushes_before = _pushes(bench)
        barrier = asyncio.create_task(schedule.after_train_step())
        await asyncio.sleep(0.05)
        # Reward still in flight: admission paused, sync not yet performed.
        blocked = await owner_snapshot(schedule._rollout_thread)
        assert blocked.producer_state is not None
        assert blocked.producer_state.paused_for_weight_sync is True
        assert _pushes(bench) == pushes_before
        assert not barrier.done()

        reward.allow_score.set()
        await asyncio.wait_for(barrier, 5.0)
        assert _pushes(bench) == pushes_before + 1
        resumed = await owner_snapshot(schedule._rollout_thread)
        assert resumed.producer_state is not None
        assert resumed.producer_state.paused_for_weight_sync is False
    finally:
        reward.allow_score.set()
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_draining_barrier_reports_mode_zero(monkeypatch, tmp_path) -> None:
    """A full-finetune run has no versioned slots: the barrier mode metric is 0 (draining)."""

    bench = _bench(monkeypatch, tmp_path)
    assert bench.stack.resolved.built.trainer.versioned_weight_sync is False
    schedule = _build(_continuous_config(), bench)

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)
        phases = await schedule.after_train_step()
        assert phases.as_metrics_dict()["continuous.weight_sync_barrier_mode"] == 0.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_non_draining_sync_skips_inflight_wait(monkeypatch, tmp_path) -> None:
    """The whole point of versioned slots: when the launch contract allows
    non-draining sync, after_train_step must NOT wait for in-flight
    generation/reward. It syncs and returns while the gated reward is still
    blocked; the in-flight request keeps its own version's slot."""

    reward = _GatedReward()
    bench = _bench(monkeypatch, tmp_path, non_draining=True, reward=reward)
    assert bench.stack.resolved.built.trainer.versioned_weight_sync is True
    schedule = _build(_continuous_config(), bench)

    try:
        reward.block_after_scores = 2
        reward.score_blocked.clear()
        reward.allow_score.clear()
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p2", "p3"],
        )
        assert await asyncio.to_thread(reward.score_blocked.wait, 5.0)

        pushes_before = _pushes(bench)
        # Must complete WITHOUT opening the reward gate (contrast with the draining
        # canary test, where this would block until allow_score.set()).
        phases = await asyncio.wait_for(schedule.after_train_step(), 5.0)

        assert phases.as_metrics_dict()["continuous.weight_sync_barrier_mode"] == 1.0
        assert _pushes(bench) == pushes_before + 1
        snapshot = await owner_snapshot(schedule._rollout_thread)
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
    finally:
        reward.allow_score.set()
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_three_gas2_updates_consume_exact_finite_prefetch_sequence(
    monkeypatch, tmp_path
) -> None:
    """Three GAS2 updates consume each announced prompt batch once within stale bound."""

    bench = _bench(monkeypatch, tmp_path, non_draining=True)
    schedule = _build(
        _continuous_config(max_inflight_groups=6, max_stale_policy_versions=1),
        bench,
    )
    prompts = [[f"update-{update}-micro-{micro}"] for update in range(3) for micro in range(2)]
    iterations = []
    sync_stats = []

    try:
        for index, current_prompts in enumerate(prompts):
            next_prompts = prompts[index + 1] if index + 1 < len(prompts) else None
            iterations.append(
                await schedule.next_iteration(
                    current_prompts,
                    group_size=4,
                    next_prompts=next_prompts,
                ),
            )
            if index % 2 == 1:
                sync_stats.append(await schedule.after_train_step())

        assert [
            [batch.trajectory.sample_rows[0].prompt for batch in iteration.batches]
            for iteration in iterations
        ] == prompts
        assert [
            _iteration_stat(iteration, "continuous.rollout_policy_version")
            for iteration in iterations
        ] == [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        assert [
            _iteration_stat(iteration, "continuous.consume_policy_version")
            for iteration in iterations
        ] == [1, 1, 2, 2, 3, 3]
        assert [
            _iteration_stat(iteration, "continuous.stale_policy_versions")
            for iteration in iterations
        ] == [0, 0, 1, 0, 1, 0]
        assert [
            iteration.stats.as_metrics_dict()["continuous.lookahead_requested"]
            for iteration in iterations
        ] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        assert [
            (request.prompts, request.samples_per_prompt, request.policy_version)
            for request in bench.trace.requests
        ] == [
            (current_prompts, 4, rollout_version)
            for current_prompts, rollout_version in zip(
                prompts,
                [1, 1, 1, 2, 2, 3],
                strict=True,
            )
        ]
        assert [
            stats.as_metrics_dict()["continuous.weight_sync_barrier_mode"] for stats in sync_stats
        ] == [1.0, 1.0, 1.0]
        assert bench.runtime.current_policy_version == 4

        snapshot = await owner_snapshot(schedule._rollout_thread)
        assert snapshot.producer_state is not None
        assert snapshot.batch_stats["ready_items"] == 0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_stale_slot_discard_fails_the_fixed_version_prompt_batch(
    monkeypatch, tmp_path
) -> None:
    """A prompt batch cannot replace one slot without violating version identity.

    The worker raises a typed StaleSlotDiscard when a request outlives its
    trainable-state slot window under non-draining sync: not a generation
    failure, and the finite batch's fixed version cannot be preserved.
    """

    bench = _bench(monkeypatch, tmp_path, non_draining=True)
    bench.trace.fail(
        "generate",
        StaleSlotDiscard("trainable-state slot evicted for policy_version=1"),
        times=50,
    )
    schedule = _build(_continuous_config(wait_timeout_s=30.0, fail_fast_errors=2), bench)

    try:
        with pytest.raises(RuntimeError, match="producer control loop failed") as exc_info:
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert exc_info.value.__cause__ is not None
        assert "lost its fixed policy-version slot" in str(exc_info.value.__cause__)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prefetch_installs_the_next_prompt_batch(monkeypatch, tmp_path) -> None:
    """The producer advances to exactly the prompt batch announced as prefetch."""

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p0", "p2"],
        )
        await schedule.next_iteration(["p0", "p2"], group_size=2)
        assert (await owner_snapshot(schedule._rollout_thread)).prompts == ("p0", "p2")
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prefetch_freezes_runtime_debug_at_generation_time(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=True,
            next_prompts=["p1"],
        )
        await schedule.next_iteration(
            ["p1"],
            group_size=1,
            runtime_debug=False,
        )
        assert [request.runtime_debug for request in bench.trace.requests] == [True, True]
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prefetch_mismatch_fails_instead_of_training_the_wrong_prompt_batch(
    monkeypatch, tmp_path
) -> None:
    """The trainer must consume the exact prompt batch it announced as prefetch."""

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(
        _continuous_config(
            max_stale_policy_versions=1,
            max_inflight_groups=2,
            wait_timeout_s=1.0,
        ),
        bench,
    )

    try:
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=1,
            next_prompts=["p2", "p3"],
        )
        with pytest.raises(RuntimeError, match="prefetch prompt batch does not match"):
            await schedule.next_iteration(["p2", "different"], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prefetch_group_size_mismatch_fails_before_consumption(
    monkeypatch, tmp_path
) -> None:
    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=["p1"],
        )
        with pytest.raises(RuntimeError, match="prefetch group size does not match"):
            await schedule.next_iteration(["p1"], group_size=2)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_prefetch_accepts_identical_prompt_with_non_scalar_metadata(
    monkeypatch, tmp_path
) -> None:
    """The sampler may present the same object whose structural equality is invalid."""

    bench = _bench(monkeypatch, tmp_path)
    schedule = _build(_continuous_config(), bench)
    prompt = PromptExample(
        prompt="p1",
        metadata={"embedding": torch.tensor([1.0, 2.0])},
    )

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=[prompt],
        )
        await schedule.next_iteration([prompt], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.parametrize("window", [1.5, "1", True])
def test_continuous_schedule_does_not_coerce_policy_window(monkeypatch, tmp_path, window) -> None:
    bench = _bench(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="max_stale_policy_versions"):
        _build(_continuous_config(max_stale_policy_versions=window), bench)


@pytest.mark.parametrize("name", ["wait_timeout_s", "queue_poll_interval_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_continuous_config_requires_finite_positive_waits(name, value) -> None:
    from vrl.trainers.core.types import ContinuousRolloutConfig

    with pytest.raises(ValueError, match=name):
        ContinuousRolloutConfig(**{name: value})
