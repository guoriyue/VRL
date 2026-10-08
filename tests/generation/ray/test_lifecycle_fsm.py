"""Terminal runtime lifecycle, shared shutdown, and retryable teardown.

The ``RuntimeLifecycle`` state machine is tested on its own. Every runtime
test drives a real ``RayGenerationRuntime`` launched by the real launcher over
real ``RayGenerationWorker`` actors serving tiny SANA (``ray_sana_runtime``).
Faults are real: a configured generation stall deadline that a slow request
overruns, a tightened weight-sync RPC deadline, a killed worker actor, a
cancelled in-flight request. Ordering is controlled by wrapping a real method
with an awaited gate, and one-shot cleanup failures are queued on the real
method with ``Trace``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest

from tests.generation.ray._helpers import install_slow_workers, ray_sana_runtime
from tests.rollouts.collector._helpers import Trace
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.actor_pool import RayActorCallError
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout
from vrl.runtime_errors import TerminalRuntimeError, root_failure_cause
from vrl.utils.lifecycle import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)

# ``SlowGenerationWorker`` holds a "slow" request for a fixed second; the stall
# deadline below expires well before that, while an ordinary request finishes
# well inside it.
_STALL = "distributed.rollout.generation_stall_timeout_s=0.5"


@pytest.fixture(autouse=True)
def _slow_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    install_slow_workers(monkeypatch)


def _actors(runtime: RayGenerationRuntime) -> list[Any]:
    session = runtime._session
    assert session is not None
    return [rank.actor for rank in session.rank_handles]


def _is_dead(ray: Any, actor: Any, *, settle_s: float = 10.0) -> bool:
    """Whether the actor stops answering; ``ray.kill`` takes effect asynchronously."""

    import time

    deadline = time.monotonic() + settle_s
    while True:
        try:
            ray.get(actor.worker_metadata.remote(), timeout=30)
        except ray.exceptions.RayActorError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _gate(
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
    method: str,
    *,
    before: Callable[..., Any] | None = None,
    after: Callable[..., Any] | None = None,
) -> None:
    """Await ``before(*args)`` ahead of the real coroutine method, ``after`` behind it."""

    real = getattr(target, method)

    async def gated(*args: Any, **kwargs: Any) -> Any:
        if before is not None:
            await before(*args, **kwargs)
        result = await real(*args, **kwargs)
        if after is not None:
            await after(*args, **kwargs)
        return result

    monkeypatch.setattr(target, method, gated)


async def _until(condition: Callable[[], bool], timeout_s: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.001)


# ------------------------------------------------------------ lifecycle FSM


def test_terminal_lifecycle_closes_admission_and_finishes_once() -> None:
    lifecycle = RuntimeLifecycle()
    assert lifecycle.phase is RuntimePhase.RUNNING

    lifecycle.begin_shutdown()
    lifecycle.begin_shutdown()
    assert lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    with pytest.raises(RuntimeLifecycleError, match="shutting down"):
        lifecycle.require_running("generate")

    lifecycle.finish_shutdown()
    assert lifecycle.phase is RuntimePhase.TERMINATED
    with pytest.raises(RuntimeLifecycleError, match="terminated"):
        lifecycle.require_running("generate")
    with pytest.raises(RuntimeLifecycleError, match="terminated"):
        lifecycle.finish_shutdown()


def test_failure_preserves_first_root_cause() -> None:
    lifecycle = RuntimeLifecycle()
    root = RuntimeError("actor died")
    lifecycle.fail(root)
    lifecycle.fail(RuntimeError("secondary noise"))
    assert lifecycle.failure is root
    with pytest.raises(RuntimeLifecycleError) as caught:
        lifecycle.require_running("generate")
    assert caught.value.__cause__ is root


def test_concurrent_failure_publishers_share_one_first_root_cause() -> None:
    lifecycle = RuntimeLifecycle()
    errors = [RuntimeError(f"failure-{index}") for index in range(8)]
    start = threading.Barrier(len(errors))
    retained: list[BaseException | None] = [None] * len(errors)

    def publish(index: int) -> None:
        start.wait()
        retained[index] = lifecycle.fail(errors[index])

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(len(errors))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert lifecycle.failure in errors
    assert all(error is lifecycle.failure for error in retained)
    assert (
        sum(
            error is retained_error for error, retained_error in zip(errors, retained, strict=True)
        )
        == 1
    )
    assert lifecycle.phase is RuntimePhase.SHUTTING_DOWN


# -------------------------------------------------------- shutdown and admission


@pytest.mark.asyncio
async def test_terminated_runtime_fail_fasts_public_operations(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        actors = _actors(runtime)

        await runtime.shutdown()
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        await runtime.shutdown()
        with pytest.raises(RuntimeLifecycleError, match="generate"):
            await runtime.generate(ray_run.request(["p"]))
        with pytest.raises(RuntimeLifecycleError, match="update_weights"):
            await runtime.update_weights({}, 1)
        with pytest.raises(RuntimeLifecycleError, match="activate"):
            await runtime.activate()
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_shutdown_closes_new_admission_before_cleanup_yields(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        close_started = asyncio.Event()
        finish_close = asyncio.Event()

        async def hold(*_args: Any, **_kwargs: Any) -> None:
            close_started.set()
            await finish_close.wait()

        _gate(monkeypatch, runtime._session, "close", before=hold)
        shutdown = asyncio.create_task(runtime.shutdown())
        await asyncio.wait_for(close_started.wait(), timeout=10)

        with pytest.raises(RuntimeLifecycleError, match="shutting down"):
            await runtime.generate(ray_run.request(["p"]))
        with pytest.raises(RuntimeLifecycleError, match="shutting down"):
            await runtime.update_weights({}, 1)

        finish_close.set()
        await asyncio.wait_for(shutdown, timeout=30)
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_share_cleanup(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        close_started = asyncio.Event()
        finish_close = asyncio.Event()
        closes = Trace(monkeypatch)

        async def hold(*_args: Any, **_kwargs: Any) -> None:
            close_started.set()
            await finish_close.wait()

        _gate(monkeypatch, runtime._session, "close", before=hold)
        closes.watch(runtime._session, "close", "session.close")
        first = asyncio.create_task(runtime.shutdown())
        await asyncio.wait_for(close_started.wait(), timeout=10)
        second = asyncio.create_task(runtime.shutdown())
        await asyncio.sleep(0)
        assert closes.events == ["session.close"]
        assert not first.done()
        assert not second.done()

        finish_close.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=30)
        assert closes.events == ["session.close"]
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_does_not_cancel_cleanup(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        actors = _actors(runtime)
        close_started = asyncio.Event()
        finish_close = asyncio.Event()

        async def hold(*_args: Any, **_kwargs: Any) -> None:
            close_started.set()
            await finish_close.wait()

        _gate(monkeypatch, runtime._session, "close", before=hold)
        cancelled_waiter = asyncio.create_task(runtime.shutdown())
        await asyncio.wait_for(close_started.wait(), timeout=10)
        surviving_waiter = asyncio.create_task(runtime.shutdown())
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter

        assert not surviving_waiter.done()
        finish_close.set()
        await asyncio.wait_for(surviving_waiter, timeout=30)
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_cleanup_failure_can_be_retried_without_reporting_terminated(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        actors = _actors(runtime)
        cleanup_error = RuntimeError("cleanup failed")
        closes = Trace(monkeypatch)
        closes.watch(runtime._session, "close", "session.close")
        closes.fail("session.close", cleanup_error)

        with pytest.raises(RuntimeError, match="cleanup failed") as caught:
            await runtime.shutdown()
        assert caught.value is cleanup_error
        assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        assert runtime.lifecycle.failure is cleanup_error

        await runtime.shutdown()
        assert closes.events == ["session.close", "session.close"]
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        with pytest.raises(RuntimeLifecycleError) as rejected:
            await runtime.generate(ray_run.request(["p"]))
        assert rejected.value.__cause__ is cleanup_error
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_existing_failure_survives_successful_cleanup(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        actors = _actors(runtime)
        root = RuntimeError("actor died")
        runtime.lifecycle.fail(root)

        await runtime.shutdown()

        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime.lifecycle.failure is root
        with pytest.raises(RuntimeLifecycleError) as rejected:
            await runtime.generate(ray_run.request(["p"]))
        assert rejected.value.__cause__ is root
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_cleanup_error_chains_existing_root_without_replacing_it(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        root = RuntimeError("actor died")
        cleanup_error = RuntimeError("cleanup failed")
        runtime.lifecycle.fail(root)
        closes = Trace(monkeypatch)
        closes.watch(runtime._session, "close", "session.close")
        closes.fail("session.close", cleanup_error)

        with pytest.raises(RuntimeError, match="cleanup failed") as caught:
            await runtime.shutdown()

        assert caught.value is cleanup_error
        assert caught.value.__cause__ is root
        assert runtime.lifecycle.failure is root
        assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        # The retry releases the fleet the failed cleanup retained.
        await runtime.shutdown()
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


# --------------------------------------------- terminal failures force-kill


@pytest.mark.asyncio
async def test_generation_timeout_force_kills_without_release_rpc(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=(_STALL,)
    ) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        actors = _actors(runtime)
        barrier = Trace(monkeypatch)
        # The graceful release waits on the release refs with ray.get; a
        # force-close never submits them.
        barrier.watch(local_ray, "get", "ray.get")

        with pytest.raises(RayOperationTimeout) as caught:
            await runtime.generate(ray_run.request(["slow"]))

        assert "rollout.generation.batch" in str(caught.value)
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime._session is None
        assert session._force_close is True
        assert barrier.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_submitted_generation_cancellation_force_kills_and_stays_cancelled(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        actors = _actors(runtime)
        submitted = asyncio.Event()

        async def mark(*_args: Any, **_kwargs: Any) -> None:
            submitted.set()

        _gate(monkeypatch, session.executor, "execute", before=mark)
        generation = asyncio.create_task(runtime.generate(ray_run.request(["slow"])))
        await submitted.wait()
        # The batch is now on the worker; cancelling abandons a submitted call.
        await asyncio.sleep(0.1)
        generation.cancel()

        with pytest.raises(asyncio.CancelledError) as caught:
            await generation

        terminal = caught.value.__cause__
        assert isinstance(terminal, RayOperationCancelled)
        assert runtime.lifecycle.failure is terminal
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._force_close is True
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_pre_submission_generation_cancellation_keeps_runtime_running(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        executor = runtime._session.executor
        reached = asyncio.Event()
        never = asyncio.Event()

        async def hold(*_args: Any, **_kwargs: Any) -> None:
            reached.set()
            await never.wait()

        real_execute = executor.execute
        _gate(monkeypatch, executor, "execute", before=hold)
        generation = asyncio.create_task(runtime.generate(ray_run.request(["p"])))
        await reached.wait()
        generation.cancel()

        with pytest.raises(asyncio.CancelledError) as caught:
            await generation

        assert caught.value.__cause__ is None
        assert runtime.lifecycle.phase is RuntimePhase.RUNNING
        assert runtime.lifecycle.failure is None
        # Nothing was submitted, so the fleet still serves the next request.
        monkeypatch.setattr(executor, "execute", real_execute)
        output = await runtime.generate(ray_run.request(["p"]))
        assert output.output.shape[0] == 2


@pytest.mark.asyncio
async def test_timeout_preserves_root_when_force_cleanup_also_fails(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=(_STALL,)
    ) as ray_run:
        runtime = ray_run.runtime
        closes = Trace(monkeypatch)
        closes.watch(runtime._session, "close", "session.close")
        closes.fail("session.close", RuntimeError("actor kill failed"))

        with pytest.raises(RayOperationTimeout) as caught:
            await runtime.generate(ray_run.request(["slow"]))

        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        assert any("actor kill failed" in note for note in caught.value.__notes__)
        await runtime.shutdown()
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_terminal_executor_error_closes_runtime(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A dead worker is a terminal executor error: the runtime closes for good."""

    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        local_ray.kill(_actors(runtime)[0], no_restart=True)

        with pytest.raises(RayActorCallError) as caught:
            await runtime.generate(ray_run.request(["p"]))

        assert isinstance(caught.value, TerminalRuntimeError)
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


async def _sibling_timeout(
    runtime: RayGenerationRuntime, ray_run: Any, *, policy_version: int | None = None
) -> asyncio.Task[Any]:
    """A slow request whose real stall timeout terminates the fleet."""

    return asyncio.create_task(
        runtime.generate(ray_run.request(["slow"], policy_version=policy_version)),
    )


@pytest.mark.asyncio
async def test_active_sibling_failure_escapes_as_the_first_failure_identity(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A request admitted before a sibling's timeout killed the fleet reports the
    sibling's timeout, not the actor error its own call hits on the dead fleet."""

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=(_STALL, "distributed.resources.rollout.num_engines=2"),
    ) as ray_run:
        runtime = ray_run.runtime
        executor = runtime._session.executor
        actor_errors: list[BaseException] = []

        async def after_sibling_failed(request: Any) -> None:
            if request.prompts == ["healthy"]:
                await _until(lambda: runtime.lifecycle.phase is RuntimePhase.TERMINATED)

        real_execute = executor.execute

        async def execute(request: Any) -> Any:
            await after_sibling_failed(request)
            try:
                return await real_execute(request)
            except BaseException as error:
                actor_errors.append(error)
                raise

        monkeypatch.setattr(executor, "execute", execute)
        sibling = await _sibling_timeout(runtime, ray_run)
        healthy = asyncio.create_task(runtime.generate(ray_run.request(["healthy"])))

        with pytest.raises(RayOperationTimeout) as timed_out:
            await sibling
        with pytest.raises(RayOperationTimeout) as caught:
            await healthy

        assert caught.value is timed_out.value
        assert runtime.lifecycle.failure is timed_out.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        # The healthy request's own call failed on the killed fleet ...
        assert actor_errors and not isinstance(actor_errors[-1], RayOperationTimeout)
        # ... and the restart policy keys on the first failure.
        assert root_failure_cause(caught.value) is timed_out.value


@pytest.mark.asyncio
async def test_active_sibling_failure_wins_over_a_later_ordinary_error(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=(_STALL,)
    ) as ray_run:
        runtime = ray_run.runtime
        executor = runtime._session.executor
        later_error = RuntimeError("batch correlation failed after fleet kill")
        real_execute = executor.execute

        async def execute(request: Any) -> Any:
            if request.prompts == ["healthy"]:
                await _until(lambda: runtime.lifecycle.failure is not None)
                raise later_error
            return await real_execute(request)

        monkeypatch.setattr(executor, "execute", execute)
        healthy = asyncio.create_task(runtime.generate(ray_run.request(["healthy"])))
        sibling = await _sibling_timeout(runtime, ray_run)

        with pytest.raises(RayOperationTimeout) as timed_out:
            await sibling
        with pytest.raises(RayOperationTimeout) as caught:
            await healthy

        assert caught.value is timed_out.value
        assert root_failure_cause(caught.value) is timed_out.value
        assert runtime.lifecycle.failure is timed_out.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_active_sibling_failure_keeps_cancelled_surface_with_first_cause(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=(_STALL,)
    ) as ray_run:
        runtime = ray_run.runtime
        executor = runtime._session.executor
        waiting = asyncio.Event()
        never = asyncio.Event()
        real_execute = executor.execute

        async def execute(request: Any) -> Any:
            if request.prompts == ["healthy"]:
                waiting.set()
                await never.wait()
            return await real_execute(request)

        monkeypatch.setattr(executor, "execute", execute)
        healthy = asyncio.create_task(runtime.generate(ray_run.request(["healthy"])))
        await waiting.wait()
        sibling = await _sibling_timeout(runtime, ray_run)
        with pytest.raises(RayOperationTimeout) as timed_out:
            await sibling
        healthy.cancel()

        with pytest.raises(asyncio.CancelledError) as caught:
            await healthy

        assert caught.value.__cause__ is timed_out.value
        assert root_failure_cause(caught.value) is timed_out.value
        assert runtime.lifecycle.failure is timed_out.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_weight_ack_timeout_keeps_previous_version_and_force_kills(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        actors = _actors(runtime)
        payload = ray_run.trainable_state()
        await runtime.update_weights(payload, 6)
        # Launch shares this RPC budget with the workers' policy load, so the
        # deadline is tightened only for the next install; the real worker
        # cannot ACK inside it.
        session.weight_sync.worker_rpc_timeout_s = 1e-6
        barrier = Trace(monkeypatch)
        barrier.watch(local_ray, "get", "ray.get")

        with pytest.raises(RayOperationTimeout) as caught:
            await runtime.update_weights(payload, 7)

        assert "rollout.weight_sync" in str(caught.value)
        assert runtime.current_policy_version == 6
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._force_close is True
        assert barrier.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_sibling_failure_after_weight_ack_blocks_version_publication(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """The fleet ACKed the install, but a sibling request's timeout closed
    admission before the driver published the version."""

    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=(_STALL,)
    ) as ray_run:
        runtime = ray_run.runtime
        payload = ray_run.trainable_state()
        await runtime.update_weights(payload, 6)
        acked: list[int] = []

        async def hold_after_ack(_state: Any, policy_version: int) -> None:
            acked.append(policy_version)
            await _until(lambda: runtime.lifecycle.failure is not None)

        _gate(monkeypatch, runtime._session, "update_weights", after=hold_after_ack)
        update = asyncio.create_task(runtime.update_weights(payload, 7))
        await _until(lambda: acked == [7])
        # The sibling targets the version the fleet just installed.
        sibling = await _sibling_timeout(runtime, ray_run, policy_version=7)

        with pytest.raises(RayOperationTimeout) as timed_out:
            await sibling
        with pytest.raises(RayOperationTimeout) as caught:
            await update

        assert caught.value is timed_out.value
        assert runtime.current_policy_version == 6
        assert runtime.lifecycle.failure is timed_out.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert root_failure_cause(caught.value) is timed_out.value


@pytest.mark.asyncio
async def test_later_timeout_upgrades_ordinary_failure_to_force_cleanup(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        actors = _actors(runtime)
        ordinary = RuntimeError("health failed first")
        runtime.lifecycle.fail(ordinary)
        timeout = RayOperationTimeout("rollout.generation.batch", 1.0)
        barrier = Trace(monkeypatch)
        barrier.watch(local_ray, "get", "ray.get")

        await runtime._terminalize_after_failure(timeout)

        assert runtime.lifecycle.failure is ordinary
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._force_close is True
        assert barrier.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_timeout_interrupts_an_already_waiting_release_barrier(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        actors = _actors(runtime)
        release_entered = threading.Event()
        finish_release = threading.Event()
        real_get = local_ray.get

        def held_get(refs: Any, *, timeout: float) -> Any:
            # The graceful release barrier: wait for the workers' policy release.
            assert timeout == 60
            release_entered.set()
            finish_release.wait()
            return real_get(refs, timeout=timeout)

        monkeypatch.setattr(local_ray, "get", held_get)
        graceful_shutdown = asyncio.create_task(runtime.shutdown())
        assert await asyncio.to_thread(release_entered.wait, 30.0)

        timeout = RayOperationTimeout("rollout.generation.batch", 1.0)
        try:
            await asyncio.wait_for(runtime._terminalize_after_failure(timeout), timeout=30.0)
        finally:
            finish_release.set()
            shutdown_results = await asyncio.gather(graceful_shutdown, return_exceptions=True)
            monkeypatch.setattr(local_ray, "get", real_get)

        assert shutdown_results == [None]
        assert runtime.lifecycle.failure is timeout
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._release_wait_task is None
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_timeout_prevents_sibling_generation_from_returning_output(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A sibling that finished its real generation still cannot return its
    output once another request's timeout terminated the runtime."""

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=(_STALL, "distributed.resources.rollout.num_engines=2"),
    ) as ray_run:
        runtime = ray_run.runtime
        sibling_generated = asyncio.Event()

        async def hold_output(request: Any) -> None:
            if request.prompts == ["sibling"]:
                sibling_generated.set()
                await _until(lambda: runtime.lifecycle.failure is not None)

        _gate(monkeypatch, runtime._session.executor, "execute", after=hold_output)
        sibling = asyncio.create_task(runtime.generate(ray_run.request(["sibling"])))
        await asyncio.wait_for(sibling_generated.wait(), timeout=30)
        slow = await _sibling_timeout(runtime, ray_run)

        with pytest.raises(RayOperationTimeout) as timed_out:
            await slow
        with pytest.raises(RayOperationTimeout) as caught:
            await sibling
        assert caught.value is timed_out.value


# ---------------------------------------------------------- actor cleanup


@pytest.mark.asyncio
async def test_actor_cleanup_failure_retains_owned_handle_for_retry(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        ranks = list(runtime._session.rank_handles)
        kills = Trace(monkeypatch)
        kills.watch(local_ray, "kill", "ray.kill")
        kills.fail("ray.kill", "kill transport failed")

        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            await runtime.shutdown()
        assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        assert runtime._session.rank_handles == ranks

        await runtime.shutdown()
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime._session is None
        assert kills.events == ["ray.kill", "ray.kill"]
        assert _is_dead(local_ray, ranks[0].actor)


@pytest.mark.asyncio
async def test_partial_engine_cleanup_retries_only_failed_rank(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=("distributed.resources.rollout.num_engines=2",),
    ) as ray_run:
        runtime = ray_run.runtime
        ranks = list(runtime._session.rank_handles)
        assert len(ranks) == 2
        killed: list[Any] = []
        real_kill = local_ray.kill

        def kill(actor: Any, *, no_restart: bool) -> None:
            killed.append(actor)
            if actor is ranks[1].actor and len(killed) == 2:
                raise RuntimeError("second rank kill failed")
            real_kill(actor, no_restart=no_restart)

        monkeypatch.setattr(local_ray, "kill", kill)

        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            await runtime.shutdown()
        assert runtime._session.rank_handles == [ranks[1]]
        assert _is_dead(local_ray, ranks[0].actor)

        await runtime.shutdown()
        monkeypatch.setattr(local_ray, "kill", real_kill)
        assert runtime._session is None
        assert killed == [ranks[0].actor, ranks[1].actor, ranks[1].actor]
        assert _is_dead(local_ray, ranks[1].actor)


@pytest.mark.asyncio
async def test_async_launcher_loads_off_loop(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """The deferred launch builds the real fleet on a worker thread, never the loop."""

    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        await ray_run.runtime.shutdown()
        from vrl import run

        resolved = ray_run.resolved
        replay = run.resolve_model(
            resolved.family,
            resolved.built.root,
            resolved.device,
            precision=resolved.built.precision,
            for_rollout=False,
        )
        launches = Trace(monkeypatch)
        launches.watch(RayGenerationLauncher, "_launch_session", "launch")

        session = await RayGenerationLauncher()._launch_session_async(
            resolved.generation,
            resolved.ray_launch_inputs(replay),
            placement=ray_run.placement_owner.rollout_placement,
        )
        try:
            assert launches.events == ["launch"]
            assert launches.thread_ids("launch") != [threading.get_ident()]
            assert len(session.rank_handles) == 1
        finally:
            await session.close(force=True)


@pytest.mark.asyncio
async def test_shutdown_kills_only_owned_actor(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with (
        ray_sana_runtime(monkeypatch, tmp_path / "owned", ray_sana_snapshot) as owned,
        ray_sana_runtime(monkeypatch, tmp_path / "bystander", ray_sana_snapshot) as bystander,
    ):
        owned_actor = _actors(owned.runtime)[0]
        bystander_actor = _actors(bystander.runtime)[0]

        await owned.runtime.shutdown()

        assert _is_dead(local_ray, owned_actor)
        assert not _is_dead(local_ray, bystander_actor)
        output = await bystander.runtime.generate(bystander.request(["p"]))
        assert output.output.shape[0] == 2
