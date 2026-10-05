"""Explicit activation/offload for shared-GPU Ray rollout workers."""

from __future__ import annotations

import asyncio
import gc
import weakref
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from tests.generation.ray._helpers import NeverRef, ResolvedRef
from tests.generation.ray._helpers import engine as _engine
from tests.generation.ray._helpers import parking_snapshot as _parking_snapshot
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.ray.actor_pool import RayActorCallError
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout
from vrl.runtime_errors import root_failure_cause
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer
from vrl.utils.lifecycle import RuntimeLifecycleError, RuntimePhase


class _FakeSession:
    """Record ordered session operations without requiring a Ray cluster."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.current_policy_version: int | None = 0
        self.workers: list[Any] = []
        self.executor = SimpleNamespace(workers=[])
        self.weight_sync = object()
        self.supports_non_draining_weight_sync = False
        self.force_close_calls = 0

    async def sleep_engines(self) -> None:
        self.calls.append("sleep")

    async def wake_engines(self) -> None:
        self.calls.append("wake")

    async def shutdown(self) -> None:
        self.calls.append("shutdown")

    async def close(self, *, force: bool) -> None:
        if force:
            self.force_close()
        await self.shutdown()

    def force_close(self) -> None:
        self.force_close_calls += 1

    async def update_weights(self, state_ref: Any, version: int) -> None:
        self.calls.append(("update", state_ref, version))
        self.current_policy_version = version


class _BlockingRestoreSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.restore_started = asyncio.Event()
        self.finish_restore = asyncio.Event()

    async def update_weights(self, state_ref: Any, version: int) -> None:
        self.restore_started.set()
        await self.finish_restore.wait()
        await super().update_weights(state_ref, version)


class _BlockingSleepSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.sleep_started = asyncio.Event()
        self.finish_sleep = asyncio.Event()

    async def sleep_engines(self) -> None:
        self.sleep_started.set()
        await self.finish_sleep.wait()
        await super().sleep_engines()


class _BlockingWakeSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.wake_started = asyncio.Event()
        self.finish_wake = asyncio.Event()

    async def wake_engines(self) -> None:
        self.wake_started.set()
        await self.finish_wake.wait()
        await super().wake_engines()


class _NonRetainingSession(_FakeSession):
    """Acknowledge policy installs without keeping the test payload alive."""

    async def update_weights(self, state_ref: Any, version: int) -> None:
        del state_ref
        self.calls.append(("update", version))
        self.current_policy_version = version


class _PolicyPayload:
    pass


class _TimeoutWeightSync:
    def __init__(self, error: RayOperationTimeout) -> None:
        self.error = error

    async def push_to_rollout_engines(self, _state_ref: Any, _version: int) -> None:
        raise self.error


class _ReleaseActor:
    def __init__(self) -> None:
        self.release_calls = 0
        self.release_policy = SimpleNamespace(remote=self._release)

    def _release(self) -> object:
        self.release_calls += 1
        return object()


def _timeout_session(
    error: RayOperationTimeout,
) -> tuple[RayGenerationSession, _ReleaseActor]:
    actor = _ReleaseActor()
    session = RayGenerationSession(
        SimpleNamespace(),
        _TimeoutWeightSync(error),
        [_engine("w0", actor)],
    )
    return session, actor


def _on_demand_runtime() -> RayGenerationRuntime:
    async def unexpected_session() -> RayGenerationSession:
        raise AssertionError("test did not configure a cold session")

    runtime = RayGenerationRuntime(
        session=None,
        session_factory=unexpected_session,
        initial_policy_version=0,
    )
    return runtime


async def _stage_pending_install(
    runtime: RayGenerationRuntime,
    state_ref: Any,
    policy_version: int,
) -> None:
    await runtime.update_weights(state_ref, policy_version)


def _assert_pending_install(
    runtime: RayGenerationRuntime,
    state_ref: Any,
    policy_version: int,
) -> None:
    policy = runtime._pending_install
    assert policy is not None
    assert policy.trainable_state == state_ref
    assert policy.policy_version == policy_version


def _attach_active_session(
    runtime: RayGenerationRuntime,
    session: _FakeSession,
    policy_version: int,
) -> None:
    runtime._session = session
    runtime.current_policy_version = policy_version
    session.current_policy_version = policy_version


class _RemoteResult:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def remote(self) -> ResolvedRef:
        self.calls += 1
        return ResolvedRef(self.value)


class _ParkingActor:
    def __init__(self, report: Any) -> None:
        self.sleep = _RemoteResult(report)
        self.wake = _RemoteResult(None)
        self.release_policy = _RemoteResult(None)


class _CleanupRay:
    def __init__(self) -> None:
        self.killed: list[Any] = []

    def get(self, refs: list[Any], *, timeout: float) -> list[None]:
        del timeout
        return [None for _ in refs]

    def kill(self, actor: Any, *, no_restart: bool) -> None:
        assert no_restart is True
        self.killed.append(actor)


@pytest.fixture
def cleanup_ray(monkeypatch: pytest.MonkeyPatch) -> _CleanupRay:
    import vrl.generation.ray.session as session_module

    ray = _CleanupRay()
    monkeypatch.setattr(session_module, "require_ray", lambda: ray)
    return ray


def _parking_runtime(*reports: Any) -> RayGenerationRuntime:
    engines = [
        _engine(f"rollout-{index}", _ParkingActor(report)) for index, report in enumerate(reports)
    ]
    runtime = _on_demand_runtime()
    runtime._session = RayGenerationSession(SimpleNamespace(), None, engines)
    return runtime


def _failed_parking_session() -> tuple[RayGenerationSession, _ParkingActor]:
    actor = _ParkingActor(_parking_snapshot())
    session = RayGenerationSession(
        SimpleNamespace(),
        None,
        [_engine("rollout-0", actor)],
    )
    return session, actor


@pytest.mark.asyncio
async def test_offload_accepts_complete_worker_parking_evidence() -> None:
    runtime = _parking_runtime(
        _parking_snapshot("rollout-0"),
        _parking_snapshot("rollout-1"),
    )

    await runtime.offload()

    assert runtime._session_parked is True
    assert [worker.actor.sleep.calls for worker in runtime._session.rank_handles] == [1, 1]


@pytest.mark.asyncio
async def test_offload_rejects_mismatched_worker_parking_evidence(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime(_parking_snapshot("another-worker"))
    actor = runtime._session.rank_handles[0].actor

    with pytest.raises(
        RuntimeError,
        match="mismatched rank memory-parking report",
    ) as caught:
        await runtime.offload()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_offload_rejects_worker_gpu_residual(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime(_parking_snapshot(residual_bytes=1))
    actor = runtime._session.rank_handles[0].actor

    with pytest.raises(
        RuntimeError,
        match="incomplete cpu_offload memory parking",
    ) as caught:
        await runtime.offload()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_offload_requires_every_worker_parking_rpc_to_succeed(
    cleanup_ray: _CleanupRay,
) -> None:
    sleep_error = RuntimeError("worker sleep failed")
    runtime = _parking_runtime(
        _parking_snapshot("rollout-0"),
        sleep_error,
    )
    actors = [worker.actor for worker in runtime._session.rank_handles]

    with pytest.raises(RuntimeError, match="worker sleep failed") as caught:
        await runtime.offload()

    assert caught.value is sleep_error
    assert runtime.lifecycle.failure is sleep_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert all(actor.release_policy.calls == 0 for actor in actors)
    assert cleanup_ray.killed == actors


@pytest.mark.asyncio
async def test_worker_sleep_remote_error_force_kills_without_graceful_release(
    cleanup_ray: _CleanupRay,
) -> None:
    timeout = TimeoutError("worker sleep timed out")
    runtime = _parking_runtime(timeout)
    actor = runtime._session.rank_handles[0].actor

    with pytest.raises(TimeoutError, match="worker sleep timed out") as caught:
        await runtime.offload()

    assert caught.value is timeout
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_worker_wake_remote_error_force_kills_without_graceful_release(
    cleanup_ray: _CleanupRay,
) -> None:
    timeout = TimeoutError("worker wake timed out")
    runtime = _parking_runtime(_parking_snapshot())
    actor = runtime._session.rank_handles[0].actor
    actor.wake = _RemoteResult(timeout)
    runtime._session_parked = True

    with pytest.raises(TimeoutError, match="worker wake timed out") as caught:
        await runtime.activate()

    assert caught.value is timeout
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "remote_name", "workers_offloaded"),
    [
        ("offload", "sleep", False),
        ("activate", "wake", True),
    ],
)
async def test_worker_parking_deadline_force_kills_without_graceful_release(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_ray: _CleanupRay,
    method_name: str,
    remote_name: str,
    workers_offloaded: bool,
) -> None:
    import vrl.generation.ray.session as session_module

    runtime = _parking_runtime(_parking_snapshot())
    actor = runtime._session.rank_handles[0].actor
    setattr(actor, remote_name, SimpleNamespace(remote=lambda: NeverRef()))
    runtime._session_parked = workers_offloaded
    real_wait_for = asyncio.wait_for

    async def expire_at_test_deadline(awaitable: Any, *, timeout: float) -> Any:
        # The budget is a monotonic deadline's remainder, so it is 120s minus
        # the (sub-millisecond) time spent since construction.
        assert timeout == pytest.approx(120, abs=1)
        return await real_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr(session_module.asyncio, "wait_for", expire_at_test_deadline)

    with pytest.raises(TimeoutError) as caught:
        await getattr(runtime, method_name)()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_offload_rejects_invalid_worker_parking_report_type(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime({"worker_id": "rollout-0"})
    actor = runtime._session.rank_handles[0].actor

    with pytest.raises(TypeError, match="invalid memory-parking report") as caught:
        await runtime.offload()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


def test_on_demand_weight_sync_does_not_require_active_workers() -> None:
    assert RayRuntimeWeightSyncer.if_supported(_on_demand_runtime()) is not None


@pytest.mark.parametrize("supports_non_draining_weight_sync", [False, True])
def test_resident_runtime_publishes_session_non_draining_capability(
    supports_non_draining_weight_sync: bool,
) -> None:
    session = _FakeSession()
    session.supports_non_draining_weight_sync = supports_non_draining_weight_sync

    runtime = RayGenerationRuntime(session=session)

    assert runtime.supports_non_draining_weight_sync is supports_non_draining_weight_sync


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_non_draining_weight_sync", [False, True])
async def test_deferred_activation_publishes_candidate_non_draining_capability(
    supports_non_draining_weight_sync: bool,
) -> None:
    runtime = _on_demand_runtime()
    candidate = _FakeSession()
    candidate.supports_non_draining_weight_sync = supports_non_draining_weight_sync

    async def launch_session() -> _FakeSession:
        return candidate

    runtime._session_factory = launch_session
    assert runtime.supports_non_draining_weight_sync is False

    await runtime.activate()

    assert runtime._session is candidate
    assert runtime.supports_non_draining_weight_sync is supports_non_draining_weight_sync


@pytest.mark.asyncio
async def test_generate_requires_explicit_activation() -> None:
    runtime = _on_demand_runtime()
    request = SimpleNamespace(
        request_id="request-0",
        sampling={},
        samples_per_generation_batch=None,
        policy_version=None,
    )

    with pytest.raises(RuntimeError, match=r"await activate\(\) first"):
        await runtime.generate(request)


@pytest.mark.asyncio
async def test_cold_weights_are_staged_then_applied_during_activation() -> None:
    runtime = _on_demand_runtime()
    candidate = _FakeSession()

    class _Factory:
        async def launch_session(self):
            return candidate

    runtime._session_factory = _Factory().launch_session
    await runtime.update_weights("W2", 2)

    _assert_pending_install(runtime, "W2", 2)
    assert runtime._session is None
    await runtime.activate()
    assert runtime._session is candidate
    assert runtime._pending_install is None
    assert candidate.calls == [("update", "W2", 2)]


@pytest.mark.asyncio
async def test_offload_keeps_workers_and_activation_wakes_latest_policy() -> None:
    runtime = _on_demand_runtime()
    inner = _FakeSession()
    _attach_active_session(runtime, inner, 1)

    await runtime.offload()
    await runtime.update_weights("W2", 2)

    assert runtime._session is inner
    assert runtime._session_parked is True
    _assert_pending_install(runtime, "W2", 2)
    assert inner.calls == ["sleep"]

    await runtime.activate()
    assert runtime._session is inner
    assert runtime._session_parked is False
    assert runtime._pending_install is None
    assert inner.calls == ["sleep", "wake", ("update", "W2", 2)]


@pytest.mark.asyncio
async def test_active_update_publishes_only_after_session_ack() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingRestoreSession()
    _attach_active_session(runtime, inner, 1)

    update = asyncio.create_task(runtime.update_weights("W2", 2))
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    assert runtime._pending_install is None
    assert runtime.current_policy_version == 1

    inner.finish_restore.set()
    await asyncio.wait_for(update, timeout=1)
    assert runtime._pending_install is None
    assert runtime.current_policy_version == 2


@pytest.mark.asyncio
async def test_active_update_releases_acknowledged_payload() -> None:
    runtime = _on_demand_runtime()
    inner = _NonRetainingSession()
    _attach_active_session(runtime, inner, 1)
    payload = _PolicyPayload()
    payload_ref = weakref.ref(payload)

    await runtime.update_weights(payload, 2)
    del payload
    gc.collect()

    assert payload_ref() is None
    assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_cold_activation_releases_staged_payload_after_ack() -> None:
    runtime = _on_demand_runtime()
    candidate = _NonRetainingSession()

    class _Factory:
        async def launch_session(self):
            return candidate

    runtime._session_factory = _Factory().launch_session
    payload = _PolicyPayload()
    payload_ref = weakref.ref(payload)
    await runtime.update_weights(payload, 2)
    del payload
    gc.collect()
    assert payload_ref() is not None

    await runtime.activate()
    gc.collect()

    assert payload_ref() is None
    assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_wake_releases_staged_payload_after_ack() -> None:
    runtime = _on_demand_runtime()
    inner = _NonRetainingSession()
    _attach_active_session(runtime, inner, 1)
    runtime._session_parked = True
    payload = _PolicyPayload()
    payload_ref = weakref.ref(payload)
    await runtime.update_weights(payload, 2)
    del payload
    gc.collect()
    assert payload_ref() is not None

    await runtime.activate()
    gc.collect()

    assert payload_ref() is None
    assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_shutdown_releases_uninstalled_payload() -> None:
    runtime = _on_demand_runtime()
    payload = _PolicyPayload()
    payload_ref = weakref.ref(payload)
    await runtime.update_weights(payload, 2)
    del payload
    gc.collect()
    assert payload_ref() is not None

    await runtime.shutdown()
    gc.collect()

    assert payload_ref() is None
    assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_active_update_failure_preserves_installed_version_and_terminates() -> None:
    runtime = _on_demand_runtime()
    update_error = RuntimeError("worker update failed")

    class _FailingSession(_FakeSession):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise update_error

    inner = _FailingSession()
    _attach_active_session(runtime, inner, 1)

    with pytest.raises(RuntimeError, match="worker update failed") as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is update_error
    assert runtime._pending_install is None
    assert runtime.current_policy_version == 1
    assert runtime.lifecycle.failure is update_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert inner.calls == ["shutdown"]


@pytest.mark.asyncio
async def test_active_update_sibling_failure_does_not_publish_outer_version() -> None:
    runtime = _on_demand_runtime()
    sibling_failure = RayOperationTimeout("rollout.generation.batch", 0.5)

    class _HealthRaceSession(_FakeSession):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            runtime.lifecycle.fail(sibling_failure)

    inner = _HealthRaceSession()
    _attach_active_session(runtime, inner, 1)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is sibling_failure
    assert runtime._pending_install is None
    assert runtime.current_policy_version == 1
    assert runtime._session is None
    assert inner.current_policy_version == 2
    assert inner.calls == [("update", "W2", 2), "shutdown"]
    assert runtime.lifecycle.failure is sibling_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_active_sibling_failure_propagates_session_first_failure() -> None:
    runtime = _on_demand_runtime()
    sibling_failure = RayOperationTimeout("rollout.generation.batch", 0.5)
    actor_error = RayActorCallError(
        "rollout.generation.batch",
        worker_id="healthy-killed-with-fleet",
        job_index=0,
    )

    class _Executor:
        async def execute(self, _request: Any) -> None:
            runtime.lifecycle.fail(sibling_failure)
            raise actor_error

    runtime._session = RayGenerationSession(_Executor(), None, [])
    runtime.current_policy_version = None
    request = SimpleNamespace(
        request_id="sibling-failure",
        sampling={},
        samples_per_generation_batch=None,
        policy_version=None,
    )

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.generate(request)

    assert caught.value is sibling_failure
    assert root_failure_cause(caught.value) is sibling_failure
    assert runtime.lifecycle.failure is sibling_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None


@pytest.mark.asyncio
async def test_active_timeout_force_kills_the_session_owner(
    monkeypatch,
) -> None:
    import vrl.generation.ray.session as session_module

    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W1", 1)
    timeout = RayOperationTimeout("rollout.weight_sync", 1.0)
    session, actor = _timeout_session(timeout)
    runtime._session = session

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, target: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is timeout
    assert runtime.current_policy_version == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_submitted_cancellation_force_kills_the_session_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.session as session_module

    runtime = _on_demand_runtime()
    terminal = RayOperationCancelled("rollout.generation.pipelined")
    cancellation = asyncio.CancelledError()
    cancellation.__cause__ = terminal

    class _Executor:
        async def execute(self, _request: Any) -> None:
            raise cancellation

    actor = _ReleaseActor()
    session = RayGenerationSession(
        _Executor(),
        None,
        [_engine("w0", actor)],
    )
    runtime._session = session
    runtime.current_policy_version = None

    class _Ray:
        killed: ClassVar[list[Any]] = []

        @classmethod
        def kill(cls, target: Any, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)
    request = SimpleNamespace(
        request_id="req-cancelled",
        sampling={},
        samples_per_generation_batch=None,
        policy_version=None,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await runtime.generate(request)

    assert caught.value is cancellation
    assert caught.value.__cause__ is terminal
    assert runtime.lifecycle.failure is terminal
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_update_cleanup_failure_retains_session_for_retry() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W1", 1)
    update_error = RuntimeError("worker update failed")
    cleanup_error = RuntimeError("worker cleanup failed")

    class _FailingSession(_FakeSession):
        shutdown_calls = 0

        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise update_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise cleanup_error
            self.calls.append("shutdown")

    inner = _FailingSession()
    runtime._session = inner

    with pytest.raises(RuntimeError, match="worker update failed") as caught:
        await runtime.update_weights("W2", 2)
    assert caught.value is update_error
    assert runtime.lifecycle.failure is update_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._session is inner

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["offload", "activate"])
async def test_sibling_failure_before_lease_transition_preserves_root_and_force_kills(
    cleanup_ray: _CleanupRay,
    transition: str,
) -> None:
    """A failure recorded before either lease transition is the error
    that surfaces, the runtime terminates, and the fleet is force-killed."""

    sibling_failure = RayOperationTimeout("rollout.generation.batch", 0.5)
    session, actor = _failed_parking_session()
    runtime = _on_demand_runtime()
    runtime._session = session
    runtime.lifecycle.fail(sibling_failure)

    with pytest.raises(RayOperationTimeout) as caught:
        await getattr(runtime, transition)()

    assert caught.value is sibling_failure
    assert runtime.lifecycle.failure is sibling_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_offload_failure_runs_terminal_cleanup() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")

    class _FailingSession(_FakeSession):
        async def sleep_engines(self) -> None:
            raise offload_error

    inner = _FailingSession()
    runtime._session = inner

    with pytest.raises(RuntimeError, match="sleep failed") as caught:
        await runtime.offload()
    assert caught.value is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert inner.calls == ["shutdown"]


@pytest.mark.asyncio
async def test_offload_cleanup_failure_retries_without_replacing_root() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")
    cleanup_error = RuntimeError("worker cleanup failed")

    class _FailingSession(_FakeSession):
        shutdown_calls = 0

        async def sleep_engines(self) -> None:
            raise offload_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise cleanup_error
            self.calls.append("shutdown")

    inner = _FailingSession()
    runtime._session = inner

    with pytest.raises(RuntimeError, match="sleep failed") as caught:
        await runtime.offload()
    assert caught.value is offload_error
    assert any("worker cleanup failed" in note for note in getattr(offload_error, "__notes__", ()))
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._session is inner
    assert inner.shutdown_calls == 1

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cancelled_restore_closes_the_unpublished_candidate() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W", 3)
    candidate = _BlockingRestoreSession()
    candidate.supports_non_draining_weight_sync = True

    async def launch_session() -> _BlockingRestoreSession:
        return candidate

    runtime._session_factory = launch_session
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)

    activation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await activation

    assert candidate.calls == ["shutdown"]
    assert candidate.force_close_calls == 1
    assert runtime._session is None
    assert runtime.supports_non_draining_weight_sync is False
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cancelled_launch_closes_the_fleet_its_thread_still_produces() -> None:
    """Actor startup runs in a thread cancellation cannot stop; its fleet is closed."""

    runtime = _on_demand_runtime()
    candidate = _FakeSession()
    launch_started = asyncio.Event()
    finish_launch = asyncio.Event()

    async def launch_session() -> _FakeSession:
        launch_started.set()
        await finish_launch.wait()
        return candidate

    runtime._session_factory = launch_session
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(launch_started.wait(), timeout=1)

    activation.cancel()
    await asyncio.sleep(0)
    assert not activation.done()
    finish_launch.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(activation, timeout=1)

    assert candidate.calls == ["shutdown"]
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_activation_restore_failure_cleans_candidate_and_terminates() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W", 3)
    restore_error = RuntimeError("restore failed")

    class _Candidate(_FakeSession):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise restore_error

    candidate = _Candidate()

    class _Factory:
        async def launch_session(self):
            return candidate

    runtime._session_factory = _Factory().launch_session
    with pytest.raises(RuntimeError, match="restore failed") as caught:
        await runtime.activate()

    assert caught.value is restore_error
    assert candidate.calls == ["shutdown"]
    assert runtime._session is None
    assert runtime.lifecycle.failure is restore_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cold_restore_timeout_force_kills_unpublished_candidate(
    monkeypatch,
) -> None:
    import vrl.generation.ray.session as session_module

    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W2", 2)
    timeout = RayOperationTimeout("rollout.weight_sync", 1.0)
    candidate, actor = _timeout_session(timeout)

    class _Factory:
        async def launch_session(self):
            return candidate

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, target: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    runtime._session_factory = _Factory().launch_session
    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.activate()

    assert caught.value is timeout
    assert runtime._session is None
    assert runtime.current_policy_version == 2
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_activation_launch_failure_terminalizes_without_publishing_candidate() -> None:
    runtime = _on_demand_runtime()
    launch_error = RuntimeError("rollout worker startup failed")

    class _Factory:
        async def launch_session(self):
            raise launch_error

    runtime._session_factory = _Factory().launch_session

    with pytest.raises(RuntimeError, match="rollout worker startup failed") as caught:
        await runtime.activate()

    assert caught.value is launch_error
    assert runtime._session is None
    assert runtime.lifecycle.failure is launch_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_on_demand_shutdown_is_idempotent_for_offloaded_workers() -> None:
    runtime = _on_demand_runtime()
    runtime._session_parked = True
    inner = _FakeSession()
    runtime._session = inner

    await runtime.shutdown()
    await runtime.shutdown()

    assert inner.calls == ["shutdown"]
    assert runtime._session is None
    assert runtime._session_parked is False
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_terminated_runtime_rejects_activation() -> None:
    runtime = _on_demand_runtime()
    await runtime.shutdown()

    with pytest.raises(RuntimeLifecycleError, match="terminated"):
        await runtime.activate()
