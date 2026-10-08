"""Dedicated-thread ownership and commit tests for continuous rollout.

Every owner here drives the real ``RolloutRuntimeCoordinator`` over the real
collector on the tiny SANA in-process runtime (``real_collector``) and the real
trainer side (``trainer_side``): weight pushes are real trainable-state
exports through ``RayRuntimeWeightSyncer``. A ``Trace`` records which thread
each real call ran on and injects the one-shot failures the quarantine tests
need; scoring is gated by a real ``RewardFunction`` that blocks on an event.
The two Ray quarantine tests run the same owner over a real
``RayGenerationRuntime`` (real launcher, placement group and tiny-SANA
workers on a module-scoped cluster).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
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
from tests.rollouts.orchestration.continuous._helpers import _wait_until, owner_snapshot
from tests.scripts.eval.fixtures import (
    TinySanaStack,
    tiny_sana_ray_cluster,
    write_tiny_sana_snapshot,
)
from vrl import run
from vrl.ray.actor_pool import RayActorCallError
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import RolloutCollector
from vrl.rollouts.orchestration.continuous.thread import ContinuousRolloutThread
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.utils.lifecycle import RuntimePhase


class _GatedReward(IndexReward):
    """Index scoring that blocks on an event after N successful calls.

    The owner loop awaits the block through a thread, as a Ray reward actor
    call would, so the test controls when an in-flight reward completes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._block_from_call: int | None = None
        self._blocked_calls = 0
        self.allow_score = threading.Event()
        self.allow_score.set()
        self.score_calls = 0
        self.score_threads: list[int] = []

    @property
    def blocked_calls(self) -> int:
        with self._lock:
            return self._blocked_calls

    def block_scores_after(self, successful_calls: int) -> None:
        self.allow_score.clear()
        with self._lock:
            self._block_from_call = self.score_calls + int(successful_calls) + 1
            self._blocked_calls = 0

    def release_scores(self) -> None:
        self.allow_score.set()

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        self.score_threads.append(threading.get_ident())
        with self._lock:
            self.score_calls += 1
            should_block = (
                self._block_from_call is not None and self.score_calls >= self._block_from_call
            )
            if should_block:
                self._blocked_calls += 1
        if should_block:
            try:
                await asyncio.to_thread(self.allow_score.wait)
            finally:
                with self._lock:
                    self._blocked_calls -= 1
        return await super().score_batch(samples)


async def _stack(
    monkeypatch,
    tmp_path,
    *,
    reward: IndexReward | None = None,
    non_draining: bool = False,
    version: int | None = None,
) -> tuple[CollectorBench, TrainerSide]:
    """A real collector bench plus its trainer side; ``version`` pre-installs
    the trainer's weights at that policy version, as a resumed run finds them."""

    bench = real_collector(monkeypatch, tmp_path, reward=reward, versioned_slots=non_draining)
    bench.trace.watch(bench.collector, "shutdown", "collector_shutdown")
    trainer = trainer_side(bench)
    if version is not None:
        await bench.runtime.update_weights(trainer.export(), version)
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
            versioned_weight_sync=bench.stack.resolved.built.trainer.versioned_weight_sync,
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
                runtime_debug=False,
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
    reward = _GatedReward()
    bench, trainer = await _stack(monkeypatch, tmp_path, reward=reward)
    owner = _owner(bench, trainer)

    try:
        iteration = await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights=trainer.export(),
            next_prompts=["p1"],
        )
        before = await owner_snapshot(owner)

        # Simulate synchronous trainer forward/backward starving its asyncio loop.
        time.sleep(0.05)
        after = await owner_snapshot(owner)

        assert iteration.stats.as_metrics_dict()["continuous.rollout_policy_version"] == 1.0
        assert before.producer_state is not None
        assert after.producer_state is not None
        assert after.producer_state.tick_count > before.producer_state.tick_count
        pushes = bench.trace.thread_ids("update_weights")
        assert pushes and set(pushes) != {main_thread}
        collects = bench.trace.thread_ids("generate")
        assert collects and set(collects) != {main_thread}
        assert reward.score_threads and set(reward.score_threads) != {main_thread}
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_owner_skips_initial_commit_for_initialized_runtime(monkeypatch, tmp_path) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path, version=7)
    owner = _owner(bench, trainer)
    pushes_before = _pushes(bench)

    try:
        iteration = await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights=None,
        )

        assert iteration.stats.as_metrics_dict()["continuous.rollout_policy_version"] == 7.0
        assert _pushes(bench) == pushes_before
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
async def test_draining_commit_waits_for_reward_then_resumes(monkeypatch, tmp_path) -> None:
    reward = _GatedReward()
    bench, trainer = await _stack(monkeypatch, tmp_path, reward=reward)
    owner = _owner(bench, trainer)

    try:
        reward.block_scores_after(1)
        await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights=trainer.export(),
            next_prompts=["p1"],
        )
        await _wait_until(lambda: reward.blocked_calls > 0)

        push_count = _pushes(bench)
        commit = asyncio.create_task(owner.commit_weights(trainer.export()))
        await asyncio.sleep(0.05)
        blocked = await owner_snapshot(owner)

        assert commit.done() is False
        assert _pushes(bench) == push_count
        assert blocked.producer_state is not None
        assert blocked.producer_state.paused_for_weight_sync is True

        reward.release_scores()
        phases = await asyncio.wait_for(commit, 5.0)
        resumed = await owner_snapshot(owner)

        assert phases.as_metrics_dict()["continuous.weight_sync_barrier_mode"] == 0.0
        assert bench.runtime.current_policy_version == 2
        assert resumed.producer_state is not None
        assert resumed.producer_state.paused_for_weight_sync is False
    finally:
        reward.release_scores()
        await owner.shutdown()


@pytest.mark.asyncio
async def test_version_slots_skip_drain_but_still_gate_new_admission(
    monkeypatch, tmp_path
) -> None:
    reward = _GatedReward()
    bench, trainer = await _stack(monkeypatch, tmp_path, reward=reward, non_draining=True)
    assert bench.stack.resolved.built.trainer.versioned_weight_sync is True
    owner = _owner(bench, trainer)

    try:
        reward.block_scores_after(1)
        await owner.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=False,
            initial_weights=trainer.export(),
            next_prompts=["p1"],
        )
        await _wait_until(lambda: reward.blocked_calls > 0)

        phases = await asyncio.wait_for(owner.commit_weights(trainer.export()), 5.0)
        snapshot = await owner_snapshot(owner)

        assert phases.as_metrics_dict()["continuous.weight_sync_barrier_mode"] == 1.0
        assert bench.runtime.current_policy_version == 2
        assert reward.blocked_calls > 0
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
    finally:
        reward.release_scores()
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
        runtime_debug=False,
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
        resolved = ray_run.resolved
        reward = IndexReward()
        collector = RolloutCollector.from_family(
            resolved.family,
            reward_runtime=RewardFunctionRuntime(reward),
            config=resolved.collector,
            lifecycle=resolved.resources.lifecycle,
        )
        collector.set_generation_runtime(ray_run.runtime)
        trace = Trace(monkeypatch)
        trace.watch(collector, "shutdown", "collector_shutdown")
        replay = run.resolve_model(
            resolved.family,
            resolved.built.root,
            resolved.device,
            precision=resolved.built.precision,
            for_rollout=False,
        )
        bench = CollectorBench(
            stack=TinySanaStack(resolved=resolved, replay=replay, runtime=ray_run.runtime),
            collector=collector,
            runtime=ray_run.runtime,
            reward=reward,
            trace=trace,
        )
        trainer = TrainerSide(
            bundle=replay.materialize(context="owner thread ray test"),
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
            runtime_debug=False,
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
            runtime_debug=False,
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
        runtime_debug=False,
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
async def test_failed_shutdown_keeps_owner_alive_for_cleanup_retry(monkeypatch, tmp_path) -> None:
    bench, trainer = await _stack(monkeypatch, tmp_path)
    bench.trace.fail("collector_shutdown", "collector cleanup failed")
    owner = _owner(bench, trainer)

    await owner.next_iteration(
        ["p0"],
        group_size=1,
        runtime_debug=False,
        initial_weights=trainer.export(),
    )

    with pytest.raises(RuntimeError, match="collector cleanup failed"):
        await owner.shutdown()

    thread = owner._thread
    assert _shutdowns(bench) == 1
    assert thread is not None and thread.is_alive()
    assert owner._closed is False

    await owner.shutdown()
    await owner.shutdown()

    assert _shutdowns(bench) == 2
    assert owner._stopped.is_set()
    assert thread is not None and not thread.is_alive()


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


class _PromptValueReward(IndexReward):
    """Scores a sample by the integer its prompt spells, so equal prompt orders
    give equal reward batches."""

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        await super().score_batch(samples)
        return RewardOutput(scores=tuple(float(int(sample.prompt)) for sample in samples))


@pytest.mark.asyncio
async def test_checkpointed_sampler_replays_preview_prompt_order_in_a_new_owner(
    tmp_path, monkeypatch
):
    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state
    from vrl.trainers.data.prompt_sampler import PromptBatchSampler

    def sampler(generator):
        return PromptBatchSampler(
            generator=generator,
            num_examples=12,
            prompts_per_rank=3,
            strategy="random_without_replacement",
        )

    rng = torch.Generator().manual_seed(23)
    original_sampler = sampler(rng)
    current = [str(index) for index in original_sampler.sample(epoch=0)]
    preview = [str(index) for index in original_sampler.preview(epoch=1)]
    bench, trainer = await _stack(monkeypatch, tmp_path / "original", reward=_PromptValueReward())
    original = _owner(bench, trainer)
    restored = None
    try:
        await original.next_iteration(
            current,
            group_size=2,
            runtime_debug=False,
            initial_weights=trainer.export(),
            next_prompts=preview,
        )
        await original.commit_weights(trainer.export())
        checkpoint = tmp_path / "sampler.pt"
        torch.save(
            {
                "rng": capture_rng_state(prompt_generator=rng),
                "version": bench.runtime.current_policy_version,
            },
            checkpoint,
        )
        next_current = [str(index) for index in original_sampler.sample(epoch=1)]
        reference = await original.next_iteration(
            next_current,
            group_size=2,
            runtime_debug=False,
            initial_weights=None,
        )
        saved = torch.load(checkpoint, weights_only=False)
        restored_rng = torch.Generator().manual_seed(999)
        restore_rng_state(saved["rng"], prompt_generator=restored_rng)
        resumed_prompts = [str(index) for index in sampler(restored_rng).sample(epoch=1)]
        assert resumed_prompts == next_current == preview
        # A new owner over a new stack whose weights sit at the checkpointed version.
        resumed_bench, resumed_trainer = await _stack(
            monkeypatch,
            tmp_path / "resumed",
            reward=_PromptValueReward(),
            version=saved["version"],
        )
        restored = _owner(resumed_bench, resumed_trainer)
        resumed = await restored.next_iteration(
            resumed_prompts,
            group_size=2,
            runtime_debug=False,
            initial_weights=None,
        )
        for before, after in zip(reference.batches, resumed.batches, strict=True):
            assert torch.equal(before.rewards, after.rewards)
            assert torch.equal(before.group_ids, after.group_ids)
        # A restart regenerates with restored weights; matching prompt RNG does
        # not preserve the pre-crash old-policy trajectory or its numerical values.
        assert reference.stats.gauges["continuous.rollout_policy_version"] == 1
        assert resumed.stats.gauges["continuous.rollout_policy_version"] == 2
    finally:
        await original.shutdown()
        if restored is not None:
            await restored.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("group_size", [True, 2.5, "2", 0, -1])
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
                runtime_debug=False,
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
        await owner.next_iteration(
            ["p0"], group_size=1, runtime_debug=False, initial_weights=trainer.export()
        )
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
