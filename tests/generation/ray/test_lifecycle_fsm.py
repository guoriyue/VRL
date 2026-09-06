"""Terminal runtime lifecycle, shared shutdown, and retryable teardown."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import vrl.generation.ray.session as session_module
from tests.generation.ray._helpers import engine as _engine
from vrl.generation.ray.health_monitor import RolloutWorkerUnreachable
from vrl.generation.ray.pipeline_protocol import PipelinedProgressError
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorCallError
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout
from vrl.runtime_errors import failure_identity_cause
from vrl.utils.lifecycle import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)


def _runtime(
    executor: Any | None = None,
    *,
    weight_sync: Any | None = None,
    owned_workers: list[RayActorHandle] | None = None,
) -> RayGenerationRuntime:
    return RayGenerationRuntime(
        session=RayGenerationSession(
            executor=SimpleNamespace() if executor is None else executor,
            weight_sync=weight_sync,
            owned_engines=list(owned_workers or []),
        ),
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request-0",
        sampling={},
        samples_per_generation_batch=None,
        policy_version=None,
    )


def test_resident_runtime_tracks_only_worker_ownership() -> None:
    runtime = _runtime()

    assert runtime._owned_ranks == []


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


def test_terminated_runtime_fail_fasts_public_operations() -> None:
    runtime = _runtime()
    asyncio.run(runtime.shutdown())
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    asyncio.run(runtime.shutdown())
    with pytest.raises(RuntimeLifecycleError, match="generate"):
        asyncio.run(runtime.generate(_request()))
    with pytest.raises(RuntimeLifecycleError, match="update_weights"):
        asyncio.run(runtime.update_weights({}, 1))
    with pytest.raises(RuntimeLifecycleError, match="activate"):
        asyncio.run(runtime.activate())


@pytest.mark.asyncio
async def test_shutdown_closes_new_admission_before_cleanup_yields() -> None:
    teardown_started = asyncio.Event()
    finish_teardown = asyncio.Event()
    runtime = _runtime()

    async def teardown() -> None:
        teardown_started.set()
        await finish_teardown.wait()

    runtime._teardown_session = teardown
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
    runtime = _runtime()

    async def teardown() -> None:
        nonlocal teardown_calls
        teardown_calls += 1
        teardown_started.set()
        await finish_teardown.wait()

    runtime._teardown_session = teardown
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
async def test_shutdown_joins_health_monitor_without_blocking_event_loop() -> None:
    runtime = _runtime()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    loop_progressed = threading.Event()
    stop_threads: list[int] = []

    def blocking_stop() -> None:
        stop_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(loop_progressed.set)
        if not loop_progressed.wait(timeout=0.1):
            raise RuntimeError("event loop could not run while health monitor stopped")

    runtime._health_monitor.stop = blocking_stop

    await runtime.shutdown()

    assert loop_progressed.is_set()
    assert stop_threads and stop_threads[0] != loop_thread
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_does_not_cancel_cleanup() -> None:
    teardown_started = asyncio.Event()
    finish_teardown = asyncio.Event()
    runtime = _runtime()

    async def teardown() -> None:
        teardown_started.set()
        await finish_teardown.wait()

    runtime._teardown_session = teardown
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
    runtime = _runtime()

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            teardown_started.set()
            await fail_first_teardown.wait()
            raise RuntimeError("first cleanup failed")

    runtime._teardown_session = teardown
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
    runtime = _runtime()

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise cleanup_error

    runtime._teardown_session = teardown
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
    runtime = _runtime()
    runtime.lifecycle.fail(root)
    cleanup_calls = 0

    async def teardown() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    runtime._teardown_session = teardown
    await runtime.shutdown()
    assert cleanup_calls == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.failure is root
    with pytest.raises(RuntimeLifecycleError) as rejected:
        await runtime.generate(_request())
    assert rejected.value.__cause__ is root


@pytest.mark.asyncio
async def test_generation_timeout_force_kills_without_release_rpc(monkeypatch) -> None:
    timeout = RayOperationTimeout("rollout.generation.batch", 1.0)

    class _Executor:
        async def execute(self, _request) -> None:
            raise timeout

    class _Release:
        calls = 0

        @classmethod
        def remote(cls) -> None:
            cls.calls += 1

    class _Actor:
        release_policy = _Release()

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, actor: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(actor)

    actor = _Actor()
    runtime = _runtime(
        _Executor(),
        owned_workers=[_engine("w0", actor)],
    )
    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.generate(_request())

    assert caught.value is timeout
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._owned_ranks == []
    assert _Release.calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_submitted_generation_cancellation_force_kills_and_stays_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = RayOperationCancelled(
        "rollout.generation.batch",
        context="submitted_refs=1",
    )
    cancellation = asyncio.CancelledError()
    cancellation.__cause__ = terminal

    class _Executor:
        async def execute(self, _request) -> None:
            raise cancellation

    class _Release:
        calls = 0

        @classmethod
        def remote(cls) -> None:
            cls.calls += 1

    class _Actor:
        release_policy = _Release()

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, actor: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(actor)

    actor = _Actor()
    runtime = _runtime(
        _Executor(),
        owned_workers=[_engine("w0", actor)],
    )
    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)

    with pytest.raises(asyncio.CancelledError) as caught:
        await runtime.generate(_request())

    assert caught.value is cancellation
    assert caught.value.__cause__ is terminal
    assert runtime.lifecycle.failure is terminal
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._force_shutdown is True
    assert _Release.calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_pre_submission_generation_cancellation_keeps_runtime_running() -> None:
    cancellation = asyncio.CancelledError()

    class _Executor:
        async def execute(self, _request) -> None:
            raise cancellation

    runtime = _runtime(_Executor())

    with pytest.raises(asyncio.CancelledError) as caught:
        await runtime.generate(_request())

    assert caught.value is cancellation
    assert caught.value.__cause__ is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
    assert runtime.lifecycle.failure is None
    assert runtime._force_shutdown is False


@pytest.mark.asyncio
async def test_timeout_preserves_root_when_force_cleanup_also_fails() -> None:
    timeout = RayOperationTimeout("rollout.generation.batch", 1.0)
    cleanup_error = RuntimeError("actor kill failed")

    class _Executor:
        async def execute(self, _request) -> None:
            raise timeout

    runtime = _runtime(_Executor())

    async def fail_cleanup() -> None:
        raise cleanup_error

    runtime._teardown_session = fail_cleanup

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.generate(_request())

    assert caught.value is timeout
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert any("actor kill failed" in note for note in timeout.__notes__)


@pytest.mark.asyncio
async def test_terminal_progress_protocol_error_closes_runtime() -> None:
    error = PipelinedProgressError("progress request_id mismatch")

    class _Executor:
        async def execute(self, _request) -> None:
            raise error

    runtime = _runtime(_Executor())

    with pytest.raises(PipelinedProgressError) as caught:
        await runtime.generate(_request())

    assert caught.value is error
    assert runtime._force_shutdown is True
    assert runtime.lifecycle.failure is error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_active_health_failure_escapes_as_the_first_failure_identity() -> None:
    probe_timeout = TimeoutError("health probe timed out")
    health_failure = RolloutWorkerUnreachable("wedged", 0.5, probe_timeout)
    actor_error = RayActorCallError(
        "rollout.generation.batch",
        worker_id="healthy-killed-with-fleet",
        job_index=0,
    )
    actor_error.__cause__ = RuntimeError("actor died after fleet kill")
    runtime: RayGenerationRuntime

    class _Executor:
        async def execute(self, _request) -> None:
            runtime.lifecycle.fail(health_failure)
            raise actor_error

    runtime = _runtime(_Executor())

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.generate(_request())

    assert caught.value is health_failure
    assert caught.value.__cause__ is probe_timeout
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    # write_run_verdict uses this exact selector for its error_class.
    assert failure_identity_cause(caught.value) is health_failure


@pytest.mark.asyncio
async def test_active_health_failure_wins_over_a_later_ordinary_error() -> None:
    health_failure = RolloutWorkerUnreachable(
        "wedged",
        0.5,
        TimeoutError("health probe timed out"),
    )
    later_error = RuntimeError("batch correlation failed after fleet kill")
    runtime: RayGenerationRuntime

    class _Executor:
        async def execute(self, _request) -> None:
            runtime.lifecycle.fail(health_failure)
            raise later_error

    runtime = _runtime(_Executor())

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.generate(_request())

    assert caught.value is health_failure
    assert failure_identity_cause(caught.value) is health_failure
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_active_health_failure_keeps_cancelled_surface_with_first_cause() -> None:
    health_failure = RolloutWorkerUnreachable(
        "wedged",
        0.5,
        TimeoutError("health probe timed out"),
    )
    cancellation = asyncio.CancelledError()
    cancellation.__cause__ = RayOperationCancelled("rollout.generation.batch")
    runtime: RayGenerationRuntime

    class _Executor:
        async def execute(self, _request) -> None:
            runtime.lifecycle.fail(health_failure)
            raise cancellation

    runtime = _runtime(_Executor())

    with pytest.raises(asyncio.CancelledError) as caught:
        await runtime.generate(_request())

    assert caught.value is cancellation
    assert caught.value.__cause__ is health_failure
    assert failure_identity_cause(caught.value) is health_failure
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_weight_ack_timeout_keeps_previous_version_and_force_kills(
    monkeypatch,
) -> None:
    timeout = RayOperationTimeout("rollout.weight_sync", 1.0)

    class _WeightSync:
        async def push_to_rollout_engines(self, _state_ref, _policy_version) -> None:
            raise timeout

    class _Release:
        calls = 0

        @classmethod
        def remote(cls) -> None:
            cls.calls += 1

    class _Actor:
        release_policy = _Release()

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, actor: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(actor)

    actor = _Actor()
    runtime = _runtime(
        SimpleNamespace(),
        weight_sync=_WeightSync(),
        owned_workers=[_engine("w0", actor)],
    )
    runtime.current_policy_version = 6
    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.update_weights(object(), policy_version=7)

    assert caught.value is timeout
    assert runtime.current_policy_version == 6
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert _Release.calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_health_failure_after_weight_ack_blocks_version_publication() -> None:
    health_failure = RolloutWorkerUnreachable(
        "rollout-1",
        0.5,
        TimeoutError("health probe timed out"),
    )
    runtime: RayGenerationRuntime

    class _WeightSync:
        async def push_to_rollout_engines(
            self,
            _state_ref,
            _policy_version,
        ) -> None:
            # The remote ACK transaction completed, but the independent health
            # thread closed admission before the driver could publish its version.
            runtime.lifecycle.fail(health_failure)

    runtime = _runtime(SimpleNamespace(), weight_sync=_WeightSync())
    runtime.current_policy_version = 6

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.update_weights(object(), policy_version=7)

    assert caught.value is health_failure
    assert runtime.current_policy_version == 6
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert failure_identity_cause(caught.value) is health_failure


@pytest.mark.asyncio
async def test_later_timeout_upgrades_ordinary_failure_to_force_cleanup(monkeypatch) -> None:
    class _Release:
        calls = 0

        @classmethod
        def remote(cls) -> None:
            cls.calls += 1

    class _Actor:
        release_policy = _Release()

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, actor: object, *, no_restart: bool) -> None:
            del no_restart
            cls.killed.append(actor)

    actor = _Actor()
    runtime = _runtime(
        SimpleNamespace(),
        owned_workers=[_engine("w0", actor)],
    )
    ordinary = RuntimeError("health failed first")
    runtime.lifecycle.fail(ordinary)
    timeout = RayOperationTimeout("rollout.generation.batch", 1.0)
    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)

    await runtime._terminalize_after_failure(timeout)

    assert runtime.lifecycle.failure is ordinary
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._force_shutdown is True
    assert _Release.calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_timeout_interrupts_an_already_waiting_release_barrier(monkeypatch) -> None:
    release_entered = threading.Event()
    finish_release = threading.Event()

    class _Release:
        calls = 0

        @classmethod
        def remote(cls) -> object:
            cls.calls += 1
            return object()

    class _Actor:
        release_policy = _Release()

    class _Ray:
        killed: ClassVar[list[object]] = []

        @staticmethod
        def get(_refs, *, timeout: float) -> None:
            assert timeout == 60
            release_entered.set()
            finish_release.wait()

        @classmethod
        def kill(cls, actor: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(actor)

    actor = _Actor()
    runtime = _runtime(
        SimpleNamespace(),
        owned_workers=[_engine("w0", actor)],
    )
    session = runtime._session
    assert session is not None
    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)
    graceful_shutdown = asyncio.create_task(runtime.shutdown())
    assert await asyncio.to_thread(release_entered.wait, 1.0)

    timeout = RayOperationTimeout("rollout.generation.batch", 1.0)
    try:
        await asyncio.wait_for(
            runtime._terminalize_after_failure(timeout),
            timeout=1.0,
        )
    finally:
        finish_release.set()
        shutdown_results = await asyncio.gather(
            graceful_shutdown,
            return_exceptions=True,
        )

    assert shutdown_results == [None]
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert session._release_wait_task is None
    assert _Release.calls == 1
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_timeout_prevents_sibling_generation_from_returning_output() -> None:
    timeout = RayOperationTimeout("rollout.generation.batch", 1.0)
    timeout_entered = asyncio.Event()
    sibling_entered = asyncio.Event()
    fail_timeout = asyncio.Event()
    finish_sibling = asyncio.Event()

    class _Executor:
        async def execute(self, request) -> str:
            if request.request_id == "timeout":
                timeout_entered.set()
                await fail_timeout.wait()
                raise timeout
            sibling_entered.set()
            await finish_sibling.wait()
            return "must-not-escape"

    runtime = _runtime(_Executor())
    timeout_request = SimpleNamespace(
        request_id="timeout",
        sampling={},
        samples_per_generation_batch=None,
        policy_version=None,
    )
    sibling_request = SimpleNamespace(
        request_id="sibling",
        sampling={},
        samples_per_generation_batch=None,
        policy_version=None,
    )
    timeout_task = asyncio.create_task(runtime.generate(timeout_request))
    sibling_task = asyncio.create_task(runtime.generate(sibling_request))
    await asyncio.gather(timeout_entered.wait(), sibling_entered.wait())

    fail_timeout.set()
    with pytest.raises(RayOperationTimeout):
        await timeout_task
    finish_sibling.set()
    with pytest.raises(RayOperationTimeout) as sibling:
        await sibling_task
    assert sibling.value is timeout


@pytest.mark.asyncio
async def test_cleanup_error_chains_existing_root_without_replacing_it() -> None:
    root = RuntimeError("actor died")
    cleanup_error = RuntimeError("cleanup failed")
    runtime = _runtime()
    runtime.lifecycle.fail(root)

    async def teardown() -> None:
        raise cleanup_error

    runtime._teardown_session = teardown
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

    monkeypatch.setattr(session_module, "require_ray", lambda: _RayApi)
    actor = _Actor()
    worker = _engine("w0", actor)
    runtime = _runtime(
        SimpleNamespace(),
        owned_workers=[worker],
    )

    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        await runtime.shutdown()
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._owned_ranks == list(worker.ranks)

    await runtime.shutdown()
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._owned_ranks == []
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

    def launch_session(*args, **kwargs):
        del args, kwargs
        launch_threads.append(threading.get_ident())
        return "session"

    monkeypatch.setattr(launcher_module, "require_ray", lambda: _RayApi)
    monkeypatch.setattr(
        launcher_module.RayGenerationLauncher,
        "_launch_session",
        launch_session,
    )
    result = await launcher._launch_session_async(None, None, placement=None)

    assert result == "session"
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
    worker = _engine("w0", actor)
    runtime = _runtime(
        SimpleNamespace(),
        owned_workers=[worker],
    )

    asyncio.run(runtime.shutdown())

    with pytest.raises(local_ray.exceptions.RayActorError):
        local_ray.get(actor.release_policy.remote())
    assert local_ray.get(bystander.release_policy.remote()) == "released"
    # The cluster is shared: the bystander survived on purpose, so this test has
    # to retire it itself.
    local_ray.kill(bystander, no_restart=True)
