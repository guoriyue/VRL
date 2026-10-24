"""Dedicated-thread ownership and commit tests for continuous rollout.

Every owner here drives the real ``RolloutRuntimeCoordinator`` over the real
collector on the tiny SANA in-process runtime (``real_collector``) and the real
trainer side (``trainer_side``): weight pushes are real trainable-state
exports through ``RayRuntimeWeightSyncer``. A ``Trace`` records which thread
each real call ran on and injects the one-shot failures the quarantine tests
need. The two Ray quarantine tests run the same owner over a real
``RayGenerationRuntime`` (real launcher, placement group and tiny-SANA
workers on a module-scoped cluster).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.generation.ray._helpers import ray_sana_runtime
from tests.rollouts.collector._helpers import (
    CollectorBench,
    IndexReward,
    Trace,
    TrainerSide,
    real_collector,
    trainer_side,
)
from tests.rollouts.orchestration.continuous._helpers import owner_snapshot
from tests.tiny_real_stack import (
    TinySanaStack,
    tiny_sana_ray_cluster,
    write_tiny_sana_snapshot,
)
from vrl.ray.actor_pool import RayActorCallError
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import RolloutCollector, RolloutCollectorConfig
from vrl.rollouts.orchestration.continuous.thread import ContinuousRolloutThread
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.utils.lifecycle import RuntimePhase
from vrl.utils.stats import recording


async def _stack(
    monkeypatch,
    tmp_path,
    *,
    reward: IndexReward | None = None,
    non_draining: bool = False,
) -> tuple[CollectorBench, TrainerSide]:
    """A real collector bench plus its trainer side."""

    bench = real_collector(monkeypatch, tmp_path, reward=reward, versioned_slots=non_draining)
    bench.trace.watch(bench.collector, "shutdown", "collector_shutdown")
    trainer = trainer_side(bench)
    return bench, trainer


def _owner(
    bench: CollectorBench, trainer: TrainerSide, *, max_inflight_groups: int = 1
) -> ContinuousRolloutThread:
    return ContinuousRolloutThread(
        lifecycle=trainer.coordinator(bench),
        settings=ContinuousRolloutSettings(
            max_inflight_groups=max_inflight_groups,
            max_stale_policy_versions=1,
            wait_timeout_s=5.0,
            queue_poll_interval_s=0.001,
            fail_fast_errors=2,
            versioned_weight_sync=bench.stack.built.trainer.versioned_weight_sync,
        ),
    )


def _shutdowns(bench: CollectorBench) -> int:
    return bench.trace.events.count("collector_shutdown")


def _pushes(bench: CollectorBench) -> int:
    return bench.trace.events.count("update_weights")


@pytest.mark.asyncio
async def test_empty_prefetch_fails_before_initial_weight_sync(monkeypatch, tmp_path) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    owner = _owner(bench, trainer)
    try:
        with pytest.raises(ValueError, match="prefetch prompts must be non-empty"):
            await owner.next_iteration(
                ["p0"],
                group_size=1,
                initial_weights=trainer.export(),
                next_prompts=[],
            )
        assert _pushes(bench) == 0
        assert bench.trace.requests == []
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_owner_cadence_survives_blocked_trainer_event_loop(monkeypatch, tmp_path) -> None:
    main_thread = threading.get_ident()
    bench, trainer = await _stack(monkeypatch, tmp_path)
    owner = _owner(bench, trainer)

    try:
        with recording() as stats:
            await owner.next_iteration(
                ["p0"],
                group_size=1,
                initial_weights=trainer.export(),
                next_prompts=["p1"],
            )
        before = await owner_snapshot(owner)

        # Simulate synchronous trainer forward/backward starving its asyncio loop.
        time.sleep(0.05)
        after = await owner_snapshot(owner)

        # What the owner thread recorded, the startup weight push included,
        # reached the trainer-side scope.
        assert stats.gauges["continuous.rollout_policy_version"] == 1.0
        assert stats.phase_seconds["rollout.weight_sync_s"] > 0.0
        assert stats.phase_seconds["collect.engine_generate"] > 0.0
        assert before.producer_state is not None
        assert after.producer_state is not None
        assert after.producer_state.tick_count > before.producer_state.tick_count
        pushes = bench.trace.thread_ids("update_weights")
        assert pushes and set(pushes) != {main_thread}
        collects = bench.trace.thread_ids("generate")
        assert collects and set(collects) != {main_thread}
        scores = bench.trace.thread_ids("score")
        assert scores and set(scores) != {main_thread}
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_commit_closes_admission_and_preserves_root_cause(
    monkeypatch, tmp_path
) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    owner = _owner(bench, trainer)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        initial_weights=trainer.export(),
    )

    bench.trace.fail("update_weights", "worker install ACK mismatch")
    with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
        await owner.commit_weights(trainer.export())

    failed = await owner_snapshot(owner)
    assert failed.producer_state is None
    assert "worker install ACK mismatch" in str(failed.terminal_error)
    assert _shutdowns(bench) == 1

    with pytest.raises(RuntimeError, match="owner has failed") as exc_info:
        await owner.commit_weights(trainer.export())
    assert exc_info.value.__cause__ is not None
    assert "worker install ACK mismatch" in str(exc_info.value.__cause__)

    await owner.shutdown()
    await owner.shutdown()
    assert _shutdowns(bench) == 1
    assert len(set(bench.trace.thread_ids("collector_shutdown"))) == 1


@pytest.mark.asyncio
async def test_concurrent_failed_commands_share_one_terminal_cleanup(
    monkeypatch, tmp_path
) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    # A slow collector shutdown exposes any second cleanup racing the first.
    real_shutdown = bench.collector.shutdown
    active = 0
    max_concurrent = 0

    async def slow_shutdown() -> None:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        try:
            await asyncio.sleep(0.05)
            await real_shutdown()
        finally:
            active -= 1

    monkeypatch.setattr(bench.collector, "shutdown", slow_shutdown)
    bench.trace.fail("update_weights", "worker install ACK mismatch")
    owner = _owner(bench, trainer)

    results = await asyncio.gather(
        owner.commit_weights(trainer.export()),
        owner.commit_weights(trainer.export()),
        return_exceptions=True,
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    assert any("worker install ACK mismatch" in str(result) for result in results)
    assert any("owner has failed" in str(result) for result in results)
    assert _shutdowns(bench) == 1
    assert max_concurrent == 1

    await owner.shutdown()
    assert _shutdowns(bench) == 1


@pytest.fixture(scope="module")
def ray_sana_snapshot(tmp_path_factory) -> Path:
    """The one tiny SANA snapshot the module's Ray workers serve."""

    return write_tiny_sana_snapshot(tmp_path_factory.mktemp("ray-sana") / "sana-snapshot")


@pytest.fixture(scope="module")
def sana_ray_cluster(ray_sana_snapshot) -> Iterator[Any]:
    """A real local Ray cluster whose workers serve ``ray_sana_snapshot``."""

    with tiny_sana_ray_cluster(ray_sana_snapshot) as ray:
        yield ray


@contextlib.asynccontextmanager
async def _ray_owner(
    monkeypatch, tmp_path, snapshot: Path
) -> AsyncIterator[tuple[Any, CollectorBench, TrainerSide, ContinuousRolloutThread]]:
    """An owner over a real collector whose generation runtime is a real
    ``RayGenerationRuntime`` (real launcher, placement group and workers)."""

    async with ray_sana_runtime(monkeypatch, tmp_path, snapshot) as ray_run:
        built, resources = ray_run.built, ray_run.resources
        reward = IndexReward()
        collector = RolloutCollector.from_family(
            built.family,
            reward_runtime=RewardFunctionRuntime(reward),
            config=RolloutCollectorConfig.from_root(built.root),
            lifecycle=resources.lifecycle,
        )
        collector.set_generation_runtime(ray_run.runtime)
        trace = Trace(monkeypatch)
        trace.watch(collector, "shutdown", "collector_shutdown")
        replay_build = built.family.resolve_model_build(
            built.root,
            torch.device(resources.trainer_torch_device),
            precision=built.precision,
            for_rollout=False,
        )
        replay_identity = replay_build.checkpoint_identity()
        bench = CollectorBench(
            stack=TinySanaStack(
                built=built,
                resources=resources,
                replay_build=replay_build,
                replay_identity=replay_identity,
                runtime=ray_run.runtime,
            ),
            collector=collector,
            runtime=ray_run.runtime,
            reward=reward,
            trace=trace,
        )
        trainer = TrainerSide(
            bundle=built.family.build_replay(replay_build, expected_identity=replay_identity),
            strategy=SingleProcessStrategy(),
            initialized=True,
        )
        owner = _owner(bench, trainer)
        try:
            yield ray_run, bench, trainer, owner
        finally:
            await owner.shutdown()


@pytest.mark.asyncio
async def test_real_runtime_cleanup_failure_does_not_replace_ack_root(
    sana_ray_cluster, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Runtime quarantine and owner cleanup share one first-root contract.

    The workers reject a mis-shaped real payload; the runtime's own terminal
    teardown then fails once, and the owner's cleanup retries it.
    """

    async with _ray_owner(monkeypatch, tmp_path, ray_sana_snapshot) as (
        ray_run,
        bench,
        trainer,
        owner,
    ):
        runtime = ray_run.runtime
        await owner.next_iteration(
            ["p0"],
            group_size=1,
            initial_weights=None,
        )
        closes = Trace(monkeypatch)
        closes.watch(runtime._session, "close", "session.close")
        closes.fail("session.close", "runtime cleanup failed")
        payload = trainer.export()
        payload[next(iter(payload))] = torch.zeros(1)

        with pytest.raises(RayActorCallError, match=r"rollout\.weight_sync"):
            await owner.commit_weights(payload)

        failed = await owner_snapshot(owner)
        assert "rollout.weight_sync" in str(failed.terminal_error)
        assert "runtime cleanup failed" not in str(failed.terminal_error)
        # The runtime's teardown failed once; the owner's cleanup released it.
        assert closes.events == ["session.close", "session.close"]
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime.current_policy_version == 0
        assert _shutdowns(bench) == 1


@pytest.mark.asyncio
async def test_sibling_failure_after_weight_ack_never_resumes_owner_admission(
    sana_ray_cluster, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A sibling failure after the ACK is a failed commit, never a published version."""

    sibling_failure = RayOperationTimeout("rollout.generation.batch", 0.5)
    async with _ray_owner(monkeypatch, tmp_path, ray_sana_snapshot) as (
        ray_run,
        bench,
        trainer,
        owner,
    ):
        runtime = ray_run.runtime
        await owner.next_iteration(
            ["p0"],
            group_size=1,
            initial_weights=None,
        )
        session = runtime._session
        real_update = session.update_weights
        acked: list[int] = []

        async def ack_then_sibling_fails(state: Any, policy_version: int) -> None:
            # The workers really install and ACK; a sibling request's failure
            # then closes admission before the driver publishes the version.
            await real_update(state, policy_version)
            acked.append(policy_version)
            runtime.lifecycle.fail(sibling_failure)

        monkeypatch.setattr(session, "update_weights", ack_then_sibling_fails)

        with pytest.raises(RayOperationTimeout) as caught:
            await owner.commit_weights(trainer.export())

        failed = await owner_snapshot(owner)
        assert acked == [1]
        assert caught.value is sibling_failure
        assert failed.producer_state is None
        assert failed.terminal_error == repr(sibling_failure)
        assert runtime.current_policy_version == 0
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert _shutdowns(bench) == 1

        with pytest.raises(RuntimeError, match="owner has failed") as rejected:
            await owner.commit_weights(trainer.export())
        assert rejected.value.__cause__ is sibling_failure


@pytest.mark.asyncio
async def test_failed_terminal_cleanup_is_retried_by_shutdown_once(monkeypatch, tmp_path) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    bench.trace.fail("collector_shutdown", "collector cleanup failed")
    owner = _owner(bench, trainer)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        initial_weights=trainer.export(),
    )
    bench.trace.fail("update_weights", "worker install ACK mismatch")
    with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
        await owner.commit_weights(trainer.export())
    assert _shutdowns(bench) == 1

    await owner.shutdown()
    await owner.shutdown()

    assert _shutdowns(bench) == 2
    assert len(set(bench.trace.thread_ids("collector_shutdown"))) == 1


@pytest.mark.asyncio
async def test_immediate_shutdown_publishes_result_before_owner_stops(
    monkeypatch, tmp_path
) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    owner = _owner(bench, trainer)

    await asyncio.wait_for(owner.shutdown(), timeout=2.0)

    assert _shutdowns(bench) == 1
    assert owner._shutdown_future is not None
    assert owner._shutdown_future.done()
    assert owner._shutdown_future.result() is None
    assert owner._stopped.is_set()
    await owner.shutdown()
    assert _shutdowns(bench) == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_does_not_abandon_cleanup(monkeypatch, tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    bench, trainer = await _stack(monkeypatch, tmp_path)
    real_shutdown = bench.collector.shutdown

    async def gated_shutdown() -> None:
        started.set()
        await asyncio.to_thread(release.wait)
        await real_shutdown()

    monkeypatch.setattr(bench.collector, "shutdown", gated_shutdown)
    owner = _owner(bench, trainer)
    waiter = asyncio.create_task(owner.shutdown())
    try:
        assert await asyncio.to_thread(started.wait, 2.0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        # Cleanup must stop the owner without another shutdown caller.
        await asyncio.wait_for(owner._wait_until_stopped(), timeout=2.0)
        assert _shutdowns(bench) == 1
        assert owner._shutdown_future.result() is None
        await owner.shutdown()
        assert _shutdowns(bench) == 1
    finally:
        release.set()
        await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("group_size", [True, 0])
async def test_owner_rejects_invalid_group_size_before_weight_push(
    monkeypatch, tmp_path, group_size
) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    owner = _owner(bench, trainer)
    try:
        with pytest.raises(ValueError, match="group_size"):
            await owner.next_iteration(
                ["p0"],
                group_size=group_size,
                initial_weights=trainer.export(),
            )
        assert _pushes(bench) == 0
        assert bench.trace.requests == []
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_shutdown_retries_producer_stop_before_closing_collector(monkeypatch, tmp_path):
    from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer

    original_stop = ContinuousRolloutProducer.stop
    attempts = []

    async def stop(self, *, wait_timeout_s=30.0):
        attempts.append(self)
        if len(attempts) == 1:
            raise RuntimeError("producer stop failed")
        await original_stop(self, wait_timeout_s=wait_timeout_s)

    monkeypatch.setattr(ContinuousRolloutProducer, "stop", stop)
    bench, trainer = await _stack(monkeypatch, tmp_path)
    owner = _owner(bench, trainer)
    try:
        await owner.next_iteration(["p0"], group_size=1, initial_weights=trainer.export())
        with pytest.raises(RuntimeError, match="producer stop failed"):
            await owner.shutdown()
        assert _shutdowns(bench) == 0
        assert (await owner_snapshot(owner)).producer_state is not None
        await owner.shutdown()
        assert len(attempts) == 2
        assert attempts[0] is attempts[1]
        assert _shutdowns(bench) == 1
    finally:
        await owner.shutdown()
