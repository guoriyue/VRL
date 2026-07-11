"""Terminal runtime lifecycle, shared shutdown, and retryable teardown."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vrl.generation.ray.lifecycle_fsm import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)
from vrl.generation.ray.runtime import RayGenerationRuntime


def _resident_runtime() -> RayGenerationRuntime:
    return RayGenerationRuntime(executor=SimpleNamespace())


def _request() -> SimpleNamespace:
    return SimpleNamespace(sampling={}, policy_version=None)


async def _wait_for_shutdown_idle(runtime: RayGenerationRuntime) -> None:
    async def _poll() -> None:
        while runtime._shutdown_task is not None:
            await asyncio.sleep(0)

    await asyncio.wait_for(_poll(), timeout=1)


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


def test_terminated_runtime_fail_fasts_public_operations() -> None:
    runtime = _resident_runtime()
    asyncio.run(runtime.shutdown())
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    asyncio.run(runtime.shutdown())
    with pytest.raises(RuntimeLifecycleError, match="generate"):
        asyncio.run(runtime.generate(SimpleNamespace()))
    with pytest.raises(RuntimeLifecycleError, match="update_weights"):
        asyncio.run(runtime.update_weights({}, 1))
    with pytest.raises(RuntimeLifecycleError, match="activate"):
        asyncio.run(runtime.activate())


@pytest.mark.asyncio
async def test_shutdown_closes_new_admission_before_cleanup_yields() -> None:
    teardown_started = asyncio.Event()
    finish_teardown = asyncio.Event()
    runtime = _resident_runtime()

    async def teardown() -> None:
        teardown_started.set()
        await finish_teardown.wait()

    runtime._teardown_owned_resources = teardown
    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.wait_for(teardown_started.wait(), timeout=1)

    with pytest.raises(RuntimeLifecycleError, match="shutting down"):
        await runtime.generate(_request())
    with pytest.raises(RuntimeLifecycleError, match="shutting down"):
        await runtime.update_weights({}, 1)

    finish_teardown.set()
    await asyncio.wait_for(shutdown, timeout=1)


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_share_cleanup() -> None:
    teardown_started = asyncio.Event()
    finish_teardown = asyncio.Event()
    teardown_calls = 0
    runtime = _resident_runtime()

    async def teardown() -> None:
        nonlocal teardown_calls
        teardown_calls += 1
        teardown_started.set()
        await finish_teardown.wait()

    runtime._teardown_owned_resources = teardown
    first = asyncio.create_task(runtime.shutdown())
    await asyncio.wait_for(teardown_started.wait(), timeout=1)
    second = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    assert teardown_calls == 1
    assert not first.done()
    assert not second.done()

    finish_teardown.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert teardown_calls == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_does_not_cancel_cleanup() -> None:
    teardown_started = asyncio.Event()
    finish_teardown = asyncio.Event()
    runtime = _resident_runtime()

    async def teardown() -> None:
        teardown_started.set()
        await finish_teardown.wait()

    runtime._teardown_owned_resources = teardown
    cancelled_waiter = asyncio.create_task(runtime.shutdown())
    await asyncio.wait_for(teardown_started.wait(), timeout=1)
    surviving_waiter = asyncio.create_task(runtime.shutdown())
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    assert not surviving_waiter.done()
    finish_teardown.set()
    await asyncio.wait_for(surviving_waiter, timeout=1)
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cleanup_failure_after_waiter_cancellation_can_be_retried() -> None:
    teardown_started = asyncio.Event()
    fail_first_teardown = asyncio.Event()
    cleanup_calls = 0
    runtime = _resident_runtime()

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            teardown_started.set()
            await fail_first_teardown.wait()
            raise RuntimeError("first cleanup failed")

    runtime._teardown_owned_resources = teardown
    cancelled_waiter = asyncio.create_task(runtime.shutdown())
    await asyncio.wait_for(teardown_started.wait(), timeout=1)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    fail_first_teardown.set()
    await _wait_for_shutdown_idle(runtime)
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    await runtime.shutdown()
    assert cleanup_calls == 2
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cleanup_failure_can_be_retried_without_reporting_terminated() -> None:
    cleanup_error = RuntimeError("cleanup failed")
    cleanup_calls = 0
    runtime = _resident_runtime()

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise cleanup_error

    runtime._teardown_owned_resources = teardown
    with pytest.raises(RuntimeError, match="cleanup failed") as caught:
        await runtime.shutdown()
    assert caught.value is cleanup_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime.lifecycle.failure is cleanup_error

    await runtime.shutdown()
    assert cleanup_calls == 2
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    with pytest.raises(RuntimeLifecycleError) as rejected:
        await runtime.generate(_request())
    assert rejected.value.__cause__ is cleanup_error


@pytest.mark.asyncio
async def test_existing_failure_survives_successful_cleanup() -> None:
    root = RuntimeError("actor died")
    runtime = _resident_runtime()
    runtime.lifecycle.fail(root)
    cleanup_calls = 0

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    runtime._teardown_owned_resources = teardown
    await runtime.shutdown()
    assert cleanup_calls == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.failure is root
    with pytest.raises(RuntimeLifecycleError) as rejected:
        await runtime.generate(_request())
    assert rejected.value.__cause__ is root


@pytest.mark.asyncio
async def test_cleanup_error_chains_existing_root_without_replacing_it() -> None:
    root = RuntimeError("actor died")
    cleanup_error = RuntimeError("cleanup failed")
    runtime = _resident_runtime()
    runtime.lifecycle.fail(root)

    async def teardown() -> None:
        raise cleanup_error

    runtime._teardown_owned_resources = teardown
    with pytest.raises(RuntimeError, match="cleanup failed") as caught:
        await runtime.shutdown()
    assert caught.value is cleanup_error
    assert caught.value.__cause__ is root
    assert runtime.lifecycle.failure is root
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN


@pytest.mark.asyncio
async def test_actor_cleanup_failure_retains_owned_handle_for_retry(monkeypatch) -> None:
    class _Actor:
        release_policy = SimpleNamespace(remote=lambda: "release-ref")

    class _RayApi:
        kill_calls = 0

        @staticmethod
        def get(refs, timeout=None):
            del refs, timeout

        @classmethod
        def kill(cls, actor, *, no_restart: bool) -> None:
            del actor, no_restart
            cls.kill_calls += 1
            if cls.kill_calls == 1:
                raise RuntimeError("kill transport failed")

    import vrl.generation.ray.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "require_ray", lambda: _RayApi)
    actor = _Actor()
    worker = SimpleNamespace(actor=actor, worker_id="w0")
    runtime = RayGenerationRuntime(
        executor=SimpleNamespace(),
        owned_workers=[worker],
    )

    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        await runtime.shutdown()
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._owned_workers == [worker]

    await runtime.shutdown()
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._owned_workers == []
    assert _RayApi.kill_calls == 2


@pytest.mark.asyncio
async def test_async_launcher_initializes_on_caller_then_loads_off_loop(monkeypatch) -> None:
    import vrl.generation.ray.launcher as launcher_module

    caller_thread = threading.get_ident()
    init_threads: list[int] = []
    launch_threads: list[int] = []

    class _RayApi:
        initialized = False

        @classmethod
        def is_initialized(cls) -> bool:
            return cls.initialized

        @classmethod
        def init(cls, **kwargs) -> None:
            del kwargs
            init_threads.append(threading.get_ident())
            cls.initialized = True

    launcher = launcher_module.RayGenerationLauncher()

    def launch(*args, **kwargs):
        del args, kwargs
        launch_threads.append(threading.get_ident())
        return "runtime"

    monkeypatch.setattr(launcher_module, "require_ray", lambda: _RayApi)
    monkeypatch.setattr(launcher_module.RayGenerationLauncher, "launch", launch)
    result = await launcher.launch_async(None, None, None, placement=None)

    assert result == "runtime"
    assert init_threads == [caller_thread]
    assert len(launch_threads) == 1
    assert launch_threads[0] != caller_thread


class _ReleaseWorker:
    """Real Ray actor exposing the release_policy call teardown drives."""

    def release_policy(self) -> str:
        return "released"


@pytest.mark.slow_test
def test_shutdown_kills_only_owned_actor(local_ray) -> None:
    actor_cls = local_ray.remote(num_cpus=0)(_ReleaseWorker)
    actor = actor_cls.remote()
    bystander = actor_cls.remote()
    worker = SimpleNamespace(actor=actor, worker_id="w0")
    runtime = RayGenerationRuntime(
        executor=SimpleNamespace(),
        owned_workers=[worker],
    )

    asyncio.run(runtime.shutdown())

    with pytest.raises(local_ray.exceptions.RayActorError):
        local_ray.get(actor.release_policy.remote())
    assert local_ray.get(bystander.release_policy.remote()) == "released"
