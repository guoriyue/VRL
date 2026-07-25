"""Dedicated-thread ownership and commit tests for continuous rollout."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
import torch

from tests.rollouts.orchestration.continuous._helpers import _wait_until, owner_snapshot
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.owner import ContinuousRolloutOwner
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.rollouts.stats import RolloutStats


def _batch(prompts: list[str], group_size: int) -> RolloutBatch:
    batch_size = len(prompts) * group_size
    return RolloutBatch(
        rewards=torch.arange(batch_size, dtype=torch.float32),
        group_ids=torch.arange(len(prompts)).repeat_interleave(group_size),
    )


class _OwnerCollector:
    requires_runtime_offload_before_reward = False
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False

    def __init__(
        self,
        *,
        shutdown_failures: int = 0,
        shutdown_delay_s: float = 0.0,
    ) -> None:
        self._score_lock = threading.Lock()
        self._block_from_call: int | None = None
        self._blocked_calls = 0
        self.allow_score = threading.Event()
        self.allow_score.set()
        self.score_blocked = threading.Event()
        self.score_calls = 0
        self.collect_threads: list[int] = []
        self.score_threads: list[int] = []
        self.shutdown_threads: list[int] = []
        self.shutdown_calls = 0
        self.shutdown_failures = shutdown_failures
        self.shutdown_delay_s = float(shutdown_delay_s)
        self.active_shutdowns = 0
        self.max_concurrent_shutdowns = 0

    @property
    def blocked_calls(self) -> int:
        with self._score_lock:
            return self._blocked_calls

    def block_future_scores(self) -> None:
        self.block_scores_after(0)

    def block_scores_after(self, successful_calls: int) -> None:
        self.allow_score.clear()
        with self._score_lock:
            self._block_from_call = self.score_calls + int(successful_calls) + 1
            self._blocked_calls = 0
            self.score_blocked.clear()

    def release_scores(self) -> None:
        self.allow_score.set()

    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        self.collect_threads.append(threading.get_ident())
        prompt_list = list(prompts)
        return _batch(prompt_list, int(kwargs["group_size"]))

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        self.score_threads.append(threading.get_ident())
        with self._score_lock:
            self.score_calls += 1
            should_block = (
                self._block_from_call is not None and self.score_calls >= self._block_from_call
            )
            if should_block:
                self._blocked_calls += 1
                self.score_blocked.set()
        if should_block:
            try:
                await asyncio.to_thread(self.allow_score.wait)
            finally:
                with self._score_lock:
                    self._blocked_calls -= 1
        return list(pendings)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.shutdown_threads.append(threading.get_ident())
        self.active_shutdowns += 1
        self.max_concurrent_shutdowns = max(
            self.max_concurrent_shutdowns,
            self.active_shutdowns,
        )
        try:
            if self.shutdown_delay_s > 0:
                await asyncio.sleep(self.shutdown_delay_s)
            if self.shutdown_calls <= self.shutdown_failures:
                raise RuntimeError("collector cleanup failed")
        finally:
            self.active_shutdowns -= 1


class _OwnerLifecycle:
    def __init__(
        self,
        collector: _OwnerCollector,
        *,
        non_draining: bool = False,
        fail_push_call: int | None = None,
    ) -> None:
        self.collector = collector
        self.non_draining = non_draining
        self.fail_push_call = fail_push_call
        self.version = 0
        self.push_calls: list[Any] = []
        self.push_threads: list[int] = []

    async def push_prepared_weights(
        self,
        prepared: Any,
        stats: RolloutStats,
    ) -> int:
        del stats
        self.push_calls.append(prepared)
        self.push_threads.append(threading.get_ident())
        if len(self.push_calls) == self.fail_push_call:
            raise RuntimeError("worker install ACK mismatch")
        self.version += 1
        return self.version

    def current_policy_version(self) -> int:
        return self.version

    def supports_non_draining_weight_sync(self) -> bool:
        return self.non_draining

    async def shutdown_collector_runtime(self) -> None:
        await self.collector.shutdown()


def _owner(
    lifecycle: _OwnerLifecycle,
    *,
    max_inflight_groups: int = 1,
) -> ContinuousRolloutOwner:
    return ContinuousRolloutOwner(
        lifecycle=lifecycle,
        settings=ContinuousRolloutSettings(
            max_inflight_groups=max_inflight_groups,
            max_ready_groups=4,
            max_ready_bytes_mb=8,
            max_stale_policy_versions=1,
            wait_timeout_s=5.0,
            queue_poll_interval_s=0.001,
            fail_fast_errors=2,
        ),
    )


@pytest.mark.asyncio
async def test_owner_cadence_survives_blocked_trainer_event_loop() -> None:
    main_thread = threading.get_ident()
    collector = _OwnerCollector()
    lifecycle = _OwnerLifecycle(collector)
    owner = _owner(lifecycle)

    try:
        iteration = await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights={"w": 0},
            next_prompts=["p1"],
        )
        before = await owner_snapshot(owner)

        # Simulate synchronous trainer forward/backward starving its asyncio loop.
        time.sleep(0.05)
        after = await owner_snapshot(owner)

        assert iteration.policy_version == 1
        assert before.producer_state is not None
        assert after.producer_state is not None
        assert after.producer_state.tick_count > before.producer_state.tick_count
        assert lifecycle.push_threads and set(lifecycle.push_threads) != {main_thread}
        assert collector.collect_threads and set(collector.collect_threads) != {main_thread}
        assert collector.score_threads and set(collector.score_threads) != {main_thread}
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_owner_skips_initial_commit_for_initialized_runtime() -> None:
    collector = _OwnerCollector()
    lifecycle = _OwnerLifecycle(collector)
    lifecycle.version = 7
    owner = _owner(lifecycle)

    try:
        iteration = await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights=None,
        )

        assert iteration.policy_version == 7
        assert lifecycle.push_calls == []
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_draining_commit_waits_for_reward_then_resumes() -> None:
    collector = _OwnerCollector()
    lifecycle = _OwnerLifecycle(collector)
    owner = _owner(lifecycle)

    try:
        collector.block_scores_after(1)
        await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights={"w": 0},
            next_prompts=["p1"],
        )
        await _wait_until(lambda: collector.blocked_calls > 0)

        push_count = len(lifecycle.push_calls)
        commit = asyncio.create_task(owner.commit_weights({"w": 1}))
        await asyncio.sleep(0.05)
        blocked = await owner_snapshot(owner)

        assert commit.done() is False
        assert len(lifecycle.push_calls) == push_count
        assert blocked.producer_state is not None
        assert blocked.producer_state.paused_for_weight_sync is True

        collector.release_scores()
        phases = await asyncio.wait_for(commit, 5.0)
        resumed = await owner_snapshot(owner)

        assert phases.as_phase_dict()["continuous.weight_sync_barrier_mode"] == 0.0
        assert lifecycle.version == 2
        assert resumed.producer_state is not None
        assert resumed.producer_state.paused_for_weight_sync is False
    finally:
        collector.release_scores()
        await owner.shutdown()


@pytest.mark.asyncio
async def test_version_slots_skip_drain_but_still_gate_new_admission() -> None:
    collector = _OwnerCollector()
    lifecycle = _OwnerLifecycle(collector, non_draining=True)
    owner = _owner(lifecycle)

    try:
        collector.block_scores_after(1)
        await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights={"w": 0},
            next_prompts=["p1"],
        )
        await _wait_until(lambda: collector.blocked_calls > 0)

        phases = await asyncio.wait_for(owner.commit_weights({"w": 1}), 5.0)
        snapshot = await owner_snapshot(owner)

        assert phases.as_phase_dict()["continuous.weight_sync_barrier_mode"] == 1.0
        assert lifecycle.version == 2
        assert collector.blocked_calls > 0
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
    finally:
        collector.release_scores()
        await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_commit_closes_admission_and_preserves_root_cause() -> None:
    collector = _OwnerCollector()
    lifecycle = _OwnerLifecycle(collector, fail_push_call=2)
    owner = _owner(lifecycle)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        runtime_debug=False,
        initial_weights={"w": 0},
    )

    with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
        await owner.commit_weights({"w": 1})

    failed = await owner_snapshot(owner)
    assert failed.producer_state is None
    assert "worker install ACK mismatch" in str(failed.terminal_error)
    assert collector.shutdown_calls == 1

    with pytest.raises(RuntimeError, match="owner has failed") as exc_info:
        await owner.commit_weights({"w": 2})
    assert exc_info.value.__cause__ is not None
    assert "worker install ACK mismatch" in str(exc_info.value.__cause__)

    await owner.shutdown()
    await owner.shutdown()
    assert collector.shutdown_calls == 1
    assert len(set(collector.shutdown_threads)) == 1


@pytest.mark.asyncio
async def test_concurrent_failed_commands_share_one_terminal_cleanup() -> None:
    collector = _OwnerCollector(shutdown_delay_s=0.05)
    lifecycle = _OwnerLifecycle(collector, fail_push_call=1)
    owner = _owner(lifecycle)

    results = await asyncio.gather(
        owner.commit_weights({"w": 1}),
        owner.commit_weights({"w": 2}),
        return_exceptions=True,
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    assert any("worker install ACK mismatch" in str(result) for result in results)
    assert any("owner has failed" in str(result) for result in results)
    assert collector.shutdown_calls == 1
    assert collector.max_concurrent_shutdowns == 1

    await owner.shutdown()
    assert collector.shutdown_calls == 1


@pytest.mark.asyncio
async def test_real_runtime_cleanup_failure_does_not_replace_ack_root() -> None:
    """Runtime quarantine and owner cleanup share one first-root contract."""

    class _AckMismatchSync:
        async def push_to_rollout_workers(
            self,
            state_ref: Any,
            policy_version: int,
        ) -> None:
            del state_ref, policy_version
            raise RuntimeError("worker install ACK mismatch")

    collector = _OwnerCollector()
    runtime = RayGenerationRuntime(
        executor=None,
        weight_sync=_AckMismatchSync(),
    )
    runtime.current_policy_version = 1
    cleanup_calls = 0

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise RuntimeError("runtime cleanup failed")

    runtime._teardown_owned_resources = teardown

    class _RuntimeLifecycle(_OwnerLifecycle):
        async def push_prepared_weights(
            self,
            prepared: Any,
            stats: RolloutStats,
        ) -> int:
            del stats
            version = int(runtime.current_policy_version or 0) + 1
            await runtime.update_weights(prepared, version)
            return version

        def current_policy_version(self) -> int:
            return int(runtime.current_policy_version or 0)

        def supports_non_draining_weight_sync(self) -> bool:
            return False

        async def shutdown_collector_runtime(self) -> None:
            await runtime.shutdown()
            await self.collector.shutdown()

    lifecycle = _RuntimeLifecycle(collector)
    owner = _owner(lifecycle)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        runtime_debug=False,
        initial_weights=None,
    )

    with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
        await owner.commit_weights({"w": 1})

    failed = await owner_snapshot(owner)
    assert "worker install ACK mismatch" in str(failed.terminal_error)
    assert "runtime cleanup failed" not in str(failed.terminal_error)
    assert cleanup_calls == 2

    await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_terminal_cleanup_is_retried_by_shutdown_once() -> None:
    collector = _OwnerCollector(shutdown_failures=1)
    lifecycle = _OwnerLifecycle(collector, fail_push_call=2)
    owner = _owner(lifecycle)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        runtime_debug=False,
        initial_weights={"w": 0},
    )
    with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
        await owner.commit_weights({"w": 1})
    assert collector.shutdown_calls == 1

    await owner.shutdown()
    await owner.shutdown()

    assert collector.shutdown_calls == 2
    assert len(set(collector.shutdown_threads)) == 1


@pytest.mark.asyncio
async def test_failed_shutdown_keeps_owner_alive_for_cleanup_retry() -> None:
    collector = _OwnerCollector(shutdown_failures=1)
    lifecycle = _OwnerLifecycle(collector)
    owner = _owner(lifecycle)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        runtime_debug=False,
        initial_weights={"w": 0},
    )

    with pytest.raises(RuntimeError, match="collector cleanup failed"):
        await owner.shutdown()

    thread = owner._thread
    assert collector.shutdown_calls == 1
    assert thread is not None and thread.is_alive()
    assert owner._closed is False

    await owner.shutdown()
    await owner.shutdown()

    assert collector.shutdown_calls == 2
    assert owner._stopped.is_set()
    assert thread is not None and not thread.is_alive()
