"""Explicit activation/offload for shared-GPU Ray rollout workers."""

from __future__ import annotations

import asyncio
import gc
import weakref
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from vrl.generation.execution.types import (
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.health_monitor import RolloutWorkerUnreachable
from vrl.utils.lifecycle import RuntimeLifecycleError, RuntimePhase
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.types import GenerationRequest
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorCallError, RayActorDispatcher
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout
from vrl.runtime_errors import failure_identity_cause
from vrl.trainers.weight_sync import build_runtime_weight_syncer


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

    async def sleep_workers(self) -> None:
        self.calls.append("sleep")

    async def wake_workers(self) -> None:
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

    async def sleep_workers(self) -> None:
        self.sleep_started.set()
        await self.finish_sleep.wait()
        await super().sleep_workers()


class _BlockingWakeSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.wake_started = asyncio.Event()
        self.finish_wake = asyncio.Event()

    async def wake_workers(self) -> None:
        self.wake_started.set()
        await self.finish_wake.wait()
        await super().wake_workers()


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

    async def push_to_rollout_workers(self, _state_ref: Any, _version: int) -> None:
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
        [RayActorHandle(worker_id="w0", actor=actor)],
    )
    return session, actor


def _on_demand_runtime(
    *,
    sync_trainable_state: bool = True,
    colocated: bool = True,
) -> RayGenerationRuntime:
    async def unexpected_session() -> RayGenerationSession:
        raise AssertionError("test did not configure a cold session")

    runtime = RayGenerationRuntime(
        session=None,
        session_factory=unexpected_session,
        initial_policy_version=0,
        supports_weight_sync=sync_trainable_state,
        colocated=colocated,
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
    assert policy.state_ref == state_ref
    assert policy.policy_version == policy_version


def _attach_active_session(
    runtime: RayGenerationRuntime,
    session: _FakeSession,
    policy_version: int,
) -> None:
    runtime._session = session
    runtime._installed_policy_version = policy_version
    runtime.current_policy_version = policy_version
    session.current_policy_version = policy_version


def _parking_snapshot(
    worker_id: str = "rollout-0",
    *,
    residual_bytes: int = 0,
) -> WorkerMemoryParkingSnapshot:
    return WorkerMemoryParkingSnapshot(
        worker_id=worker_id,
        backend="cpu_offload",
        baseline_gpu_used_bytes=0,
        loaded_gpu_used_bytes=1024,
        residual_gpu_used_bytes=residual_bytes,
        residual_bytes_limit=0,
    )


class _ResolvedRef:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __await__(self):
        async def resolve() -> Any:
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

        return resolve().__await__()


class _NeverRef:
    def __await__(self):
        async def wait_forever() -> None:
            await asyncio.Event().wait()

        return wait_forever().__await__()


class _RemoteResult:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def remote(self) -> _ResolvedRef:
        self.calls += 1
        return _ResolvedRef(self.value)


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
    workers = [
        RayActorHandle(
            worker_id=f"rollout-{index}",
            actor=_ParkingActor(report),
        )
        for index, report in enumerate(reports)
    ]
    runtime = _on_demand_runtime()
    runtime._session = RayGenerationSession(SimpleNamespace(), None, workers)
    return runtime


def _failed_parking_session() -> tuple[RayGenerationSession, _ParkingActor]:
    actor = _ParkingActor(_parking_snapshot())
    session = RayGenerationSession(
        SimpleNamespace(),
        None,
        [
            RayActorHandle(
                worker_id="rollout-0",
                actor=actor,
            ),
        ],
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
    assert [worker.actor.sleep.calls for worker in runtime._owned_workers] == [1, 1]


@pytest.mark.asyncio
async def test_offload_rejects_mismatched_worker_parking_evidence(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime(_parking_snapshot("another-worker"))
    actor = runtime._owned_workers[0].actor

    with pytest.raises(
        RuntimeError,
        match="mismatched worker memory-parking report",
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
    actor = runtime._owned_workers[0].actor

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
    actors = [worker.actor for worker in runtime._owned_workers]

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
    actor = runtime._owned_workers[0].actor

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
    actor = runtime._owned_workers[0].actor
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
    actor = runtime._owned_workers[0].actor
    setattr(actor, remote_name, SimpleNamespace(remote=lambda: _NeverRef()))
    runtime._session_parked = workers_offloaded
    real_wait_for = asyncio.wait_for

    async def expire_at_test_deadline(awaitable: Any, *, timeout: float) -> Any:
        assert timeout == 120
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
    actor = runtime._owned_workers[0].actor

    with pytest.raises(TypeError, match="invalid memory-parking report") as caught:
        await runtime.offload()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


def test_on_demand_weight_sync_capability_does_not_require_active_workers() -> None:
    runtime = _on_demand_runtime()
    disabled = _on_demand_runtime(sync_trainable_state=False)

    assert runtime.supports_weight_sync is True
    assert build_runtime_weight_syncer(runtime) is not None
    assert disabled.supports_weight_sync is False
    assert build_runtime_weight_syncer(disabled) is None


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
async def test_deferred_activation_rejects_candidate_capability_mismatch() -> None:
    runtime = _on_demand_runtime(sync_trainable_state=True)
    candidate = _FakeSession()
    candidate.weight_sync = None
    candidate.supports_non_draining_weight_sync = True

    async def launch_session() -> _FakeSession:
        return candidate

    runtime._session_factory = launch_session
    with pytest.raises(RuntimeError, match="different weight-sync capability") as caught:
        await runtime.activate()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert runtime.supports_non_draining_weight_sync is False
    assert candidate.force_close_calls == 1
    assert candidate.calls == ["shutdown"]


def test_driver_model_offload_is_derived_from_actual_gpu_overlap() -> None:
    shared_gpu = _on_demand_runtime(colocated=True)
    separate_gpu = _on_demand_runtime(colocated=False)

    assert shared_gpu.requires_driver_model_offload is True
    assert separate_gpu.requires_driver_model_offload is False


@pytest.mark.asyncio
async def test_runtime_orders_health_monitor_around_session_parking() -> None:
    runtime = _on_demand_runtime()
    events: list[Any] = []

    class _OrderedSession(_FakeSession):
        async def sleep_workers(self) -> None:
            events.append("sleep")
            await super().sleep_workers()

        async def wake_workers(self) -> None:
            events.append("wake")
            await super().wake_workers()

        async def update_weights(self, state_ref: Any, version: int) -> None:
            events.append(("update", state_ref, version))
            await super().update_weights(state_ref, version)

    session = _OrderedSession()
    runtime._session = session
    runtime._session_parked = True
    await _stage_pending_install(runtime, "W1", 1)
    runtime._health_monitor.resume = lambda: events.append("resume")

    await runtime.activate()

    assert events == ["wake", ("update", "W1", 1), "resume"]
    runtime._health_monitor.pause = lambda: events.append("pause")

    await runtime.offload()

    assert events[-2:] == ["pause", "sleep"]


@pytest.mark.asyncio
async def test_deferred_runtime_starts_health_monitoring_paused() -> None:
    async def launch_session() -> RayGenerationSession:
        raise AssertionError("paused monitor must not launch a session")

    runtime = RayGenerationRuntime(
        session=None,
        session_factory=launch_session,
        supports_weight_sync=False,
        health_check_interval_s=0.01,
    )
    runtime.start_health_monitoring()
    try:
        assert runtime._health_monitor._paused.is_set()
        await asyncio.sleep(0.02)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_generate_requires_explicit_activation() -> None:
    runtime = _on_demand_runtime()
    request = SimpleNamespace(
        request_id="request-0",
        sampling={},
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
    assert runtime._installed_policy_version == 2
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
    assert runtime._installed_policy_version == 2
    assert runtime._pending_install is None
    assert inner.calls == ["sleep", "wake", ("update", "W2", 2)]


@pytest.mark.asyncio
async def test_activation_does_not_reinstall_an_already_active_policy() -> None:
    runtime = _on_demand_runtime()
    inner = _FakeSession()
    _attach_active_session(runtime, inner, 2)
    runtime._session_parked = True
    await _stage_pending_install(runtime, "W2", 2)

    await runtime.activate()

    assert inner.calls == ["wake"]
    assert runtime._installed_policy_version == 2
    assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_active_update_publishes_only_after_session_ack() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingRestoreSession()
    _attach_active_session(runtime, inner, 1)

    update = asyncio.create_task(runtime.update_weights("W2", 2))
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    assert runtime._pending_install is None
    assert runtime._installed_policy_version == 1
    assert runtime.current_policy_version == 1

    inner.finish_restore.set()
    await asyncio.wait_for(update, timeout=1)
    assert runtime._pending_install is None
    assert runtime._installed_policy_version == 2
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
    assert runtime._installed_policy_version == 2


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
    assert runtime._installed_policy_version == 2


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
    assert runtime._installed_policy_version == 2


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
async def test_active_update_health_race_does_not_publish_outer_version() -> None:
    runtime = _on_demand_runtime()
    health_failure = RolloutWorkerUnreachable(
        "rollout-1",
        0.5,
        TimeoutError("health probe timed out"),
    )

    class _HealthRaceSession(_FakeSession):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            runtime.lifecycle.fail(health_failure)

    inner = _HealthRaceSession()
    _attach_active_session(runtime, inner, 1)

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is health_failure
    assert runtime._pending_install is None
    assert runtime.current_policy_version == 1
    assert runtime._installed_policy_version is None
    assert runtime._session is None
    assert inner.current_policy_version == 2
    assert inner.calls == [("update", "W2", 2), "shutdown"]
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_wake_health_race_terminalizes_outer_before_active_publication() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W2", 2)
    runtime._installed_policy_version = 1
    runtime._session_parked = True
    health_failure = RolloutWorkerUnreachable(
        "rollout-1",
        0.5,
        TimeoutError("health probe timed out"),
    )

    class _WakeHealthRaceSession(_FakeSession):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            runtime.lifecycle.fail(health_failure)

    inner = _WakeHealthRaceSession()
    inner.current_policy_version = 1
    runtime._session = inner

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.activate()

    assert caught.value is health_failure
    assert inner.calls == ["wake", ("update", "W2", 2), "shutdown"]
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert runtime._installed_policy_version is None
    with pytest.raises(RuntimeLifecycleError, match="terminated") as retry:
        await runtime.activate()
    assert retry.value.__cause__ is health_failure


@pytest.mark.asyncio
async def test_active_health_race_propagates_session_first_failure() -> None:
    runtime = _on_demand_runtime()
    health_failure = RolloutWorkerUnreachable(
        "wedged",
        0.5,
        TimeoutError("health probe timed out"),
    )
    actor_error = RayActorCallError(
        "rollout.generation.chunk",
        worker_id="healthy-killed-with-fleet",
        job_index=0,
    )

    class _Executor:
        async def execute(self, _request: Any) -> None:
            runtime.lifecycle.fail(health_failure)
            raise actor_error

    runtime._session = RayGenerationSession(_Executor(), None, [])
    runtime.current_policy_version = None
    request = SimpleNamespace(
        request_id="health-race",
        sampling={},
        policy_version=None,
    )

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.generate(request)

    assert caught.value is health_failure
    assert failure_identity_cause(caught.value) is health_failure
    assert runtime.lifecycle.failure is health_failure
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
    runtime._installed_policy_version = 1

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
async def test_auto_probe_timeout_force_kills_the_session_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.session as session_module
    import vrl.ray.operation_deadline as deadline_module

    runtime = _on_demand_runtime()
    probe_ref = _NeverRef()

    class _ProbeActor(_ReleaseActor):
        def __init__(self) -> None:
            super().__init__()
            self.probe_chunk_size = SimpleNamespace(
                remote=lambda *_args, **_kwargs: probe_ref,
            )

    actor = _ProbeActor()
    worker = RayActorHandle(worker_id="w0", actor=actor)
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [worker],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=0.01,
    )
    session = RayGenerationSession(
        executor,
        None,
        [worker],
    )
    runtime._session = session

    class _Ray:
        cancelled: ClassVar[list[Any]] = []
        killed: ClassVar[list[Any]] = []

        @classmethod
        def cancel(cls, ref: Any, *, force: bool) -> None:
            assert force is False
            cls.cancelled.append(ref)

        @classmethod
        def kill(cls, target: Any, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    monkeypatch.setattr(session_module, "require_ray", lambda: _Ray)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    request = GenerationRequest(
        request_id="req-auto-timeout",
        family="sd3_5",
        task="text_to_image",
        inputs=["prompt"],
        samples_per_prompt=2,
        sampling={"samples_per_chunk": "auto"},
        policy_version=None,
    )

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.generate(request)

    assert caught.value.operation == "rollout.generation.chunk_size_probe"
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert actor.release_calls == 0
    assert _Ray.cancelled == [probe_ref]
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
        [RayActorHandle(worker_id="w0", actor=actor)],
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
async def test_health_failure_before_offload_preserves_root_and_force_kills(
    cleanup_ray: _CleanupRay,
) -> None:
    health_failure = RolloutWorkerUnreachable(
        "rollout-0",
        0.5,
        TimeoutError("health probe timed out"),
    )
    session, actor = _failed_parking_session()
    runtime = _on_demand_runtime()
    runtime._session = session
    runtime.lifecycle.fail(health_failure)

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.offload()

    assert caught.value is health_failure
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_health_failure_before_reactivation_preserves_root_and_force_kills(
    cleanup_ray: _CleanupRay,
) -> None:
    health_failure = RolloutWorkerUnreachable(
        "rollout-0",
        0.5,
        TimeoutError("health probe timed out"),
    )
    session, actor = _failed_parking_session()
    runtime = _on_demand_runtime()
    runtime._session = session
    runtime.lifecycle.fail(health_failure)

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.activate()

    assert caught.value is health_failure
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_offload_is_single_flight_and_idempotent() -> None:
    runtime = _on_demand_runtime()
    session = _BlockingSleepSession()
    runtime._session = session

    first = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(session.sleep_started.wait(), timeout=1)
    second = asyncio.create_task(runtime.offload())
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()

    session.finish_sleep.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    await runtime.offload()
    assert session.calls == ["sleep"]
    assert runtime._session_parked is True


@pytest.mark.asyncio
async def test_activation_waits_for_in_flight_offload_before_waking() -> None:
    runtime = _on_demand_runtime()
    session = _BlockingSleepSession()
    runtime._session = session

    offload = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(session.sleep_started.wait(), timeout=1)
    activation = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)

    assert not activation.done()
    assert session.calls == []

    session.finish_sleep.set()
    await asyncio.wait_for(asyncio.gather(offload, activation), timeout=1)

    assert session.calls == ["sleep", "wake"]
    assert runtime._session_parked is False


@pytest.mark.asyncio
async def test_generate_rejects_in_flight_offload() -> None:
    runtime = _on_demand_runtime()
    session = _BlockingSleepSession()
    runtime._session = session
    offload = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(session.sleep_started.wait(), timeout=1)

    request = SimpleNamespace(
        request_id="offload-race",
        sampling={},
        policy_version=None,
    )
    with pytest.raises(RuntimeError, match="offload to be idle"):
        await runtime.generate(request)

    session.finish_sleep.set()
    await asyncio.wait_for(offload, timeout=1)


@pytest.mark.asyncio
async def test_generate_rejects_in_flight_activation() -> None:
    runtime = _on_demand_runtime()
    session = _BlockingRestoreSession()
    runtime._session = session
    runtime._session_parked = True
    await _stage_pending_install(runtime, "W2", 2)

    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(session.restore_started.wait(), timeout=1)
    request = SimpleNamespace(
        request_id="activation-race",
        sampling={},
        policy_version=None,
    )

    with pytest.raises(RuntimeError, match="activation to complete"):
        await runtime.generate(request)

    session.finish_restore.set()
    await asyncio.wait_for(activation, timeout=1)


@pytest.mark.asyncio
async def test_weight_update_rejects_in_flight_offload() -> None:
    runtime = _on_demand_runtime()
    session = _BlockingSleepSession()
    runtime._session = session
    offload = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(session.sleep_started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="offload to be idle"):
        await runtime.update_weights("W2", 2)

    session.finish_sleep.set()
    await asyncio.wait_for(offload, timeout=1)


@pytest.mark.asyncio
async def test_terminal_failure_cancels_offload_with_stable_root() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepSession()
    runtime._session = inner
    timeout = RayOperationTimeout("rollout.generation.sleep", 0.5)

    offload = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.sleep_started.wait(), timeout=1)
    terminalize = asyncio.create_task(runtime._terminalize_after_failure(timeout))

    with pytest.raises(asyncio.CancelledError) as caught:
        await offload
    assert await asyncio.wait_for(terminalize, timeout=1) is timeout

    assert caught.value.__cause__ is timeout
    assert failure_identity_cause(caught.value) is timeout
    assert inner.calls == ["shutdown"]
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_waiter_cancellation_does_not_cancel_offload_or_forge_root() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepSession()
    runtime._session = inner

    waiter = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.sleep_started.wait(), timeout=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await waiter

    assert caught.value.__cause__ is None
    assert runtime.lifecycle.failure is None
    assert runtime._offload_task is not None
    assert not runtime._offload_task.done()

    inner.finish_sleep.set()
    await asyncio.wait_for(runtime.offload(), timeout=1)
    assert inner.calls == ["sleep"]
    assert runtime._session_parked is True
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


@pytest.mark.asyncio
async def test_offload_task_cleans_up_failure_after_waiter_cancellation() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("late sleep failure")

    class _LateFailingSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.sleep_started = asyncio.Event()
            self.finish_sleep = asyncio.Event()

        async def sleep_workers(self) -> None:
            self.sleep_started.set()
            await self.finish_sleep.wait()
            raise offload_error

    inner = _LateFailingSession()
    runtime._session = inner
    waiter = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.sleep_started.wait(), timeout=1)
    control_task = runtime._offload_task
    assert control_task is not None

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    inner.finish_sleep.set()
    with pytest.raises(RuntimeError, match="late sleep failure") as caught:
        await asyncio.wait_for(control_task, timeout=1)

    assert caught.value is offload_error
    assert inner.calls == ["shutdown"]
    assert runtime._session is None
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_offload_failure_runs_terminal_cleanup() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")

    class _FailingSession(_FakeSession):
        async def sleep_workers(self) -> None:
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

        async def sleep_workers(self) -> None:
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
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert inner.shutdown_calls == 2

    await runtime.shutdown()
    assert inner.shutdown_calls == 2


@pytest.mark.asyncio
async def test_repeated_offload_cleanup_failure_preserves_root_for_later_retry() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")
    cleanup_error = RuntimeError("worker cleanup keeps failing")

    class _FailingSession(_FakeSession):
        shutdown_calls = 0

        async def sleep_workers(self) -> None:
            raise offload_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls <= 2:
                raise cleanup_error
            self.calls.append("shutdown")

    inner = _FailingSession()
    runtime._session = inner

    with pytest.raises(RuntimeError, match="sleep failed") as caught:
        await runtime.offload()

    assert caught.value is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._session is inner
    assert inner.shutdown_calls == 2
    notes = getattr(offload_error, "__notes__", ())
    assert any("cleanup also failed" in note for note in notes)
    assert any("cleanup retry also failed" in note for note in notes)

    await runtime.shutdown()
    assert inner.shutdown_calls == 3
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_waiter_cancellation_during_cleanup_retry_preserves_root_cause() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")
    cleanup_error = RuntimeError("first cleanup failed")

    class _BlockingRetrySession(_FakeSession):
        shutdown_calls = 0

        def __init__(self) -> None:
            super().__init__()
            self.retry_started = asyncio.Event()
            self.finish_retry = asyncio.Event()

        async def sleep_workers(self) -> None:
            raise offload_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise cleanup_error
            self.retry_started.set()
            await self.finish_retry.wait()
            self.calls.append("shutdown")

    inner = _BlockingRetrySession()
    runtime._session = inner
    waiter = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.retry_started.wait(), timeout=1)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await waiter

    assert caught.value.__cause__ is offload_error
    assert failure_identity_cause(caught.value) is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._session is inner

    inner.finish_retry.set()
    await asyncio.wait_for(runtime.shutdown(), timeout=1)
    assert inner.shutdown_calls == 2
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_waiter_cancellation_uses_root_published_during_graceful_shutdown() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("late offload failure")
    cleanup_error = RuntimeError("graceful cleanup failed")

    class _LateFailingSession(_FakeSession):
        shutdown_calls = 0

        def __init__(self) -> None:
            super().__init__()
            self.sleep_started = asyncio.Event()
            self.finish_sleep = asyncio.Event()

        async def sleep_workers(self) -> None:
            self.sleep_started.set()
            await self.finish_sleep.wait()
            raise offload_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise cleanup_error
            self.calls.append("shutdown")

    inner = _LateFailingSession()
    runtime._session = inner
    waiter = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.sleep_started.wait(), timeout=1)
    graceful_shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime.lifecycle.failure is None

    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()
    assert runtime.lifecycle.failure is None
    inner.finish_sleep.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await waiter
    with pytest.raises(RuntimeError, match="graceful cleanup failed") as cleanup:
        await graceful_shutdown

    assert cleanup.value is cleanup_error
    assert caught.value.__cause__ is offload_error
    assert failure_identity_cause(caught.value) is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._session is inner

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_concurrent_cold_activation_launches_and_restores_once() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W", 3)
    candidate = _BlockingRestoreSession()
    launch_calls = 0

    class _Factory:
        async def launch_session(self):
            nonlocal launch_calls
            launch_calls += 1
            return candidate

    runtime._session_factory = _Factory().launch_session
    first = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)
    second = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)

    assert launch_calls == 1
    assert runtime._session is None
    assert not first.done()
    assert not second.done()

    candidate.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert runtime._session is candidate
    assert runtime._installed_policy_version == 3
    assert candidate.calls == [("update", "W", 3)]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_publish_candidate_capability() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W", 3)
    candidate = _BlockingRestoreSession()
    candidate.supports_non_draining_weight_sync = True

    async def launch_session() -> _BlockingRestoreSession:
        return candidate

    runtime._session_factory = launch_session
    cancelled_waiter = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    assert runtime._session is None
    assert runtime.supports_non_draining_weight_sync is False

    surviving_waiter = asyncio.create_task(runtime.activate())
    candidate.finish_restore.set()
    await asyncio.wait_for(surviving_waiter, timeout=1)

    assert runtime._session is candidate
    assert runtime.supports_non_draining_weight_sync is True


@pytest.mark.asyncio
async def test_concurrent_activation_wakes_and_restores_once() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingRestoreSession()
    runtime._session = inner
    runtime._session_parked = True
    await _stage_pending_install(runtime, "W", 3)

    first = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    assert runtime._session_parked is False
    assert runtime._activation_task is not None
    second = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()

    inner.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert inner.calls == ["wake", ("update", "W", 3)]


@pytest.mark.asyncio
async def test_activation_waiter_cancellation_does_not_cancel_parked_wake() -> None:
    runtime = _on_demand_runtime()
    session = _BlockingWakeSession()
    _attach_active_session(runtime, session, 0)
    runtime._session_parked = True
    await _stage_pending_install(runtime, "W1", 1)

    first_waiter = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(session.wake_started.wait(), timeout=1)
    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await first_waiter

    assert cancelled.value.__cause__ is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
    assert runtime.lifecycle.failure is None
    assert session.force_close_calls == 0
    second_waiter = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)
    assert not second_waiter.done()

    session.finish_wake.set()
    await asyncio.wait_for(second_waiter, timeout=1)

    assert session.calls == ["wake", ("update", "W1", 1)]
    assert runtime._installed_policy_version == 1
    assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_update_rejects_activation_overlap_while_offload_joins_it() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W1", 1)
    candidate = _BlockingRestoreSession()

    class _Factory:
        async def launch_session(self):
            return candidate

    runtime._session_factory = _Factory().launch_session
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="activation to be idle"):
        await runtime.update_weights("W2", 2)
    offload = asyncio.create_task(runtime.offload())
    await asyncio.sleep(0)
    assert not offload.done()
    _assert_pending_install(runtime, "W1", 1)

    candidate.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(activation, offload), timeout=1)
    assert candidate.calls == [("update", "W1", 1), "sleep"]
    assert runtime._session_parked is True


@pytest.mark.asyncio
async def test_shutdown_joins_activation_after_waiter_cancellation() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W", 4)
    candidate = _BlockingRestoreSession()

    class _Factory:
        async def launch_session(self):
            return candidate

    runtime._session_factory = _Factory().launch_session
    waiter = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await waiter
    assert cancelled.value.__cause__ is None

    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert "shutdown" not in candidate.calls

    candidate.finish_restore.set()
    await asyncio.wait_for(shutdown, timeout=1)
    assert candidate.calls == [("update", "W", 4), "shutdown"]
    assert runtime._session is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_late_shutdown_joins_activation_owned_terminal_cleanup() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W2", 2)
    runtime._installed_policy_version = 1
    runtime._session_parked = True
    health_failure = RolloutWorkerUnreachable(
        "rollout-1",
        0.5,
        TimeoutError("health probe timed out"),
    )
    finish_calls = 0
    finish_outer_shutdown = runtime.lifecycle.finish_shutdown

    def count_finish_shutdown() -> None:
        nonlocal finish_calls
        finish_calls += 1
        finish_outer_shutdown()

    runtime.lifecycle.finish_shutdown = count_finish_shutdown

    class _BlockingShutdownSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.shutdown_started = asyncio.Event()
            self.finish_shutdown = asyncio.Event()

        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            runtime.lifecycle.fail(health_failure)

        async def shutdown(self) -> None:
            self.calls.append("shutdown")
            self.shutdown_started.set()
            await self.finish_shutdown.wait()

    inner = _BlockingShutdownSession()
    inner.current_policy_version = 1
    runtime._session = inner

    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(inner.shutdown_started.wait(), timeout=1)
    shutdown_waiters = [
        asyncio.create_task(runtime.shutdown()),
        asyncio.create_task(runtime.shutdown()),
    ]
    await asyncio.sleep(0)
    assert all(not waiter.done() for waiter in shutdown_waiters)

    inner.finish_shutdown.set()
    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await activation
    await asyncio.wait_for(asyncio.gather(*shutdown_waiters), timeout=1)

    assert caught.value is health_failure
    assert inner.calls == ["wake", ("update", "W2", 2), "shutdown"]
    assert finish_calls == 1
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._session is None
    assert not any(
        "cleanup also failed" in note for note in getattr(health_failure, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_shutdown_joins_offload_before_teardown() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepSession()
    runtime._session = inner

    offload = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.sleep_started.wait(), timeout=1)
    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    assert "shutdown" not in inner.calls

    inner.finish_sleep.set()
    await asyncio.wait_for(asyncio.gather(offload, shutdown), timeout=1)
    assert inner.calls == ["sleep", "shutdown"]
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
    assert runtime._installed_policy_version is None
    assert runtime.current_policy_version == 2
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_cold_candidate_health_failure_terminalizes_outer_before_publication() -> None:
    runtime = _on_demand_runtime()
    await _stage_pending_install(runtime, "W2", 2)
    health_failure = RolloutWorkerUnreachable(
        "rollout-0",
        0.5,
        TimeoutError("health probe timed out"),
    )
    launch_calls = 0

    class _ColdHealthRaceCandidate(_FakeSession):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            runtime.lifecycle.fail(health_failure)

    candidate = _ColdHealthRaceCandidate()
    candidate.supports_non_draining_weight_sync = True

    class _Factory:
        async def launch_session(self):
            nonlocal launch_calls
            launch_calls += 1
            return candidate

    runtime._session_factory = _Factory().launch_session

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.activate()

    assert caught.value is health_failure
    assert candidate.calls == [("update", "W2", 2), "shutdown"]
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._pending_install is None
    assert runtime._session is None
    assert runtime._installed_policy_version is None
    assert runtime.supports_non_draining_weight_sync is False
    with pytest.raises(RuntimeLifecycleError, match="terminated") as retry:
        await runtime.activate()
    assert retry.value.__cause__ is health_failure
    assert launch_calls == 1


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
