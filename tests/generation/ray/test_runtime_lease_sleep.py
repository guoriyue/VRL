"""Explicit activation/offload for shared-GPU Ray rollout workers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from omegaconf import OmegaConf

from vrl.generation.execution.types import (
    DistributedWorkerHandle,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.health_monitor import RolloutWorkerUnreachable
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.lifecycle_fsm import (
    RuntimeLifecycle,
    RuntimeLifecycleError,
    RuntimePhase,
)
from vrl.generation.ray.on_demand_runtime import _OnDemandRayGenerationRuntime
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.actor_pool import RayActorCallError, RayActorDispatcher
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout
from vrl.ray.resources import resolve_distributed_resources
from vrl.runtime_errors import failure_identity_cause
from vrl.trainers.weight_sync import build_runtime_weight_syncer


class _Gatherer:
    def gather_chunks(self, *_args: Any) -> Any:
        raise AssertionError("runtime lifecycle fixture must not gather chunks")


class _UnexpectedLauncher:
    async def launch_async(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("test did not configure a cold runtime launch")


class _FakeInner:
    """Record ordered worker operations without requiring a Ray cluster."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.current_policy_version: int | None = 0
        self._owned_workers: list[Any] = []
        self.executor = SimpleNamespace(workers=[])
        self.lifecycle = RuntimeLifecycle()

    async def activate(self) -> None:
        self.lifecycle.require_running("activate")

    async def sleep_workers(self) -> None:
        self.calls.append("sleep")

    async def wake_workers(self) -> None:
        self.calls.append("wake")

    async def shutdown(self) -> None:
        self.calls.append("shutdown")

    async def update_weights(self, state_ref: Any, version: int) -> None:
        self.calls.append(("update", state_ref, version))
        self.current_policy_version = version


class _BlockingRestoreInner(_FakeInner):
    def __init__(self) -> None:
        super().__init__()
        self.restore_started = asyncio.Event()
        self.finish_restore = asyncio.Event()

    async def update_weights(self, state_ref: Any, version: int) -> None:
        self.restore_started.set()
        await self.finish_restore.wait()
        await super().update_weights(state_ref, version)


class _BlockingWakeInner(_FakeInner):
    def __init__(self) -> None:
        super().__init__()
        self.wake_started = asyncio.Event()
        self.finish_wake = asyncio.Event()

    async def wake_workers(self) -> None:
        self.calls.append("wake")
        self.wake_started.set()
        await self.finish_wake.wait()


class _BlockingSleepInner(_FakeInner):
    def __init__(self) -> None:
        super().__init__()
        self.sleep_started = asyncio.Event()
        self.finish_sleep = asyncio.Event()

    async def sleep_workers(self) -> None:
        self.sleep_started.set()
        await self.finish_sleep.wait()
        await super().sleep_workers()


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


def _timeout_inner(
    error: RayOperationTimeout,
) -> tuple[RayGenerationRuntime, _ReleaseActor]:
    actor = _ReleaseActor()
    runtime = RayGenerationRuntime(
        SimpleNamespace(),
        weight_sync=_TimeoutWeightSync(error),
        owned_workers=[DistributedWorkerHandle(worker_id="w0", actor=actor)],
    )
    runtime.current_policy_version = 1
    return runtime, actor


def _ray_config(
    *,
    sync_trainable_state: bool = True,
    topology: str = "trainer_shared",
) -> RayGenerationConfig:
    if topology == "trainer_shared":
        visible_devices = [0]
        rollout = {
            "gpu_pool": "trainer",
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
        reward = None
    elif topology == "reward_shared":
        visible_devices = [0, 1]
        rollout = {
            "devices": [1],
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
        reward = {
            "devices": [1],
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
            "gpu_pool": "rollout",
        }
    elif topology == "resident":
        visible_devices = [0, 1]
        rollout = {
            "devices": [1],
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
        reward = None
    else:  # pragma: no cover - test fixture guard
        raise ValueError(f"unknown topology: {topology}")

    resources = {
        "visible_devices": visible_devices,
        "trainer": {"devices": [0]},
        "rollout": rollout,
    }
    if reward is not None:
        resources["reward"] = reward
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": resources,
                "rollout": {
                    "sync_trainable_state": sync_trainable_state,
                },
            },
        },
    )
    return RayGenerationConfig.from_cfg(
        cfg,
        resources=resolve_distributed_resources(cfg),
    )


def _launch_contract(*, policy_version: int = 0) -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="sd3_5",
        model_build={},
        expected_model_identity={"schema": "test"},
        policy_version=policy_version,
    )


def _on_demand_runtime(
    *,
    sync_trainable_state: bool = True,
    colocated: bool = True,
    launcher: Any | None = None,
) -> _OnDemandRayGenerationRuntime:
    config = _ray_config(
        sync_trainable_state=sync_trainable_state,
        topology="trainer_shared" if colocated else "reward_shared",
    )
    return _OnDemandRayGenerationRuntime(
        config,
        RayGenerationLaunchInputs(
            launch_contract=_launch_contract(),
            gatherer=_Gatherer(),
        ),
        launcher=launcher if launcher is not None else _UnexpectedLauncher(),
        placement=SimpleNamespace(),
    )


async def _set_desired_policy(
    runtime: _OnDemandRayGenerationRuntime,
    state_ref: Any,
    policy_version: int,
) -> None:
    await runtime.update_weights(state_ref, policy_version)


def _assert_desired_policy(
    runtime: _OnDemandRayGenerationRuntime,
    state_ref: Any,
    policy_version: int,
) -> None:
    policy = runtime._desired_policy
    assert policy is not None
    assert policy.state_ref == state_ref
    assert policy.policy_version == policy_version


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
    import vrl.generation.ray.runtime as runtime_module

    ray = _CleanupRay()
    monkeypatch.setattr(runtime_module, "require_ray", lambda: ray)
    return ray


def _parking_runtime(*reports: Any) -> RayGenerationRuntime:
    workers = [
        DistributedWorkerHandle(
            worker_id=f"rollout-{index}",
            actor=_ParkingActor(report),
        )
        for index, report in enumerate(reports)
    ]
    return RayGenerationRuntime(
        SimpleNamespace(),
        owned_workers=workers,
    )


def _failed_parking_runtime(
    failure: BaseException,
) -> tuple[RayGenerationRuntime, _ParkingActor]:
    actor = _ParkingActor(_parking_snapshot())
    runtime = RayGenerationRuntime(
        SimpleNamespace(),
        owned_workers=[
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=actor,
            ),
        ],
    )
    runtime.lifecycle.fail(failure)
    return runtime, actor


@pytest.mark.asyncio
async def test_runtime_validates_complete_worker_parking_evidence() -> None:
    runtime = _parking_runtime(
        _parking_snapshot("rollout-0"),
        _parking_snapshot("rollout-1"),
    )

    snapshots = await runtime.sleep_workers()

    assert tuple(snapshot.worker_id for snapshot in snapshots) == (
        "rollout-0",
        "rollout-1",
    )


@pytest.mark.asyncio
async def test_runtime_rejects_mismatched_worker_parking_evidence(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime(_parking_snapshot("another-worker"))
    actor = runtime._owned_workers[0].actor

    with pytest.raises(
        RuntimeError,
        match="mismatched worker memory-parking report",
    ) as caught:
        await runtime.sleep_workers()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_runtime_rejects_worker_gpu_residual(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime(_parking_snapshot(residual_bytes=1))
    actor = runtime._owned_workers[0].actor

    with pytest.raises(
        RuntimeError,
        match="incomplete cpu_offload memory parking",
    ) as caught:
        await runtime.sleep_workers()

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_runtime_requires_every_worker_parking_rpc_to_succeed(
    cleanup_ray: _CleanupRay,
) -> None:
    sleep_error = RuntimeError("worker sleep failed")
    runtime = _parking_runtime(
        _parking_snapshot("rollout-0"),
        sleep_error,
    )
    actors = [worker.actor for worker in runtime._owned_workers]

    with pytest.raises(RuntimeError, match="worker sleep failed") as caught:
        await runtime.sleep_workers()

    assert caught.value is sleep_error
    assert runtime.lifecycle.failure is sleep_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert all(actor.release_policy.calls == 0 for actor in actors)
    assert cleanup_ray.killed == actors


@pytest.mark.asyncio
async def test_worker_sleep_timeout_force_kills_without_graceful_release(
    cleanup_ray: _CleanupRay,
) -> None:
    timeout = TimeoutError("worker sleep timed out")
    runtime = _parking_runtime(timeout)
    actor = runtime._owned_workers[0].actor

    with pytest.raises(TimeoutError, match="worker sleep timed out") as caught:
        await runtime.sleep_workers()

    assert caught.value is timeout
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_worker_wake_timeout_force_kills_without_graceful_release(
    cleanup_ray: _CleanupRay,
) -> None:
    timeout = TimeoutError("worker wake timed out")
    runtime = _parking_runtime(_parking_snapshot())
    actor = runtime._owned_workers[0].actor
    actor.wake = _RemoteResult(timeout)

    with pytest.raises(TimeoutError, match="worker wake timed out") as caught:
        await runtime.wake_workers()

    assert caught.value is timeout
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_cancelled_worker_sleep_force_kills_unknown_parking_state(
    cleanup_ray: _CleanupRay,
) -> None:
    runtime = _parking_runtime(_parking_snapshot())
    actor = runtime._owned_workers[0].actor
    actor.sleep = SimpleNamespace(remote=lambda: _NeverRef())
    parking = asyncio.create_task(runtime.sleep_workers())
    await asyncio.sleep(0)

    parking.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await parking

    assert runtime.lifecycle.failure is caught.value
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_runtime_rejects_missing_worker_actor_before_parking() -> None:
    runtime = RayGenerationRuntime(
        SimpleNamespace(),
        owned_workers=[DistributedWorkerHandle(worker_id="rollout-0")],
    )

    with pytest.raises(RuntimeError, match="generation workers have no actor"):
        await runtime.sleep_workers()

    assert runtime.lifecycle.failure is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


@pytest.mark.asyncio
async def test_runtime_rejects_duplicate_worker_ids_before_parking() -> None:
    runtime = RayGenerationRuntime(
        SimpleNamespace(),
        owned_workers=[
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=_ParkingActor(_parking_snapshot("rollout-0")),
            ),
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=_ParkingActor(_parking_snapshot("rollout-0")),
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="duplicate generation worker ids"):
        await runtime.sleep_workers()

    assert runtime.lifecycle.failure is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


def test_on_demand_factory_stamps_contract_for_cumem_offload() -> None:
    runtime = _on_demand_runtime()
    assert runtime._launch_inputs.launch_contract.sleep_offload is True


def test_on_demand_weight_sync_capability_does_not_require_active_workers() -> None:
    runtime = _on_demand_runtime()
    disabled = _on_demand_runtime(sync_trainable_state=False)

    assert runtime.supports_weight_sync is True
    assert build_runtime_weight_syncer(runtime) is not None
    assert disabled.supports_weight_sync is False
    assert build_runtime_weight_syncer(disabled) is None


def test_driver_model_offload_is_derived_from_actual_gpu_overlap() -> None:
    shared_gpu = _on_demand_runtime(colocated=True)
    separate_gpu = _on_demand_runtime(colocated=False)

    assert shared_gpu.requires_driver_model_offload is True
    assert separate_gpu.requires_driver_model_offload is False


def test_on_demand_factory_requires_on_demand_plan() -> None:
    config = _ray_config(
        sync_trainable_state=False,
        topology="resident",
    )

    with pytest.raises(ValueError, match="resolved on-demand"):
        _OnDemandRayGenerationRuntime(
            config,
            RayGenerationLaunchInputs(
                launch_contract=_launch_contract(),
                gatherer=_Gatherer(),
            ),
            launcher=_UnexpectedLauncher(),
            placement=SimpleNamespace(),
        )


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
    candidate = _FakeInner()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    runtime._launcher = _Launcher()
    await runtime.update_weights("W2", 2)

    _assert_desired_policy(runtime, "W2", 2)
    assert runtime._inner_runtime is None
    await runtime.activate()
    assert runtime._inner_runtime is candidate
    assert runtime._active_policy_version == 2
    assert candidate.calls == [("update", "W2", 2)]


@pytest.mark.asyncio
async def test_offload_keeps_workers_and_activation_wakes_latest_policy() -> None:
    runtime = _on_demand_runtime()
    inner = _FakeInner()
    runtime._inner_runtime = inner
    runtime._active_policy_version = 1

    await runtime.offload()
    await runtime.update_weights("W2", 2)

    assert runtime._inner_runtime is inner
    assert runtime._workers_offloaded is True
    _assert_desired_policy(runtime, "W2", 2)
    assert inner.calls == ["sleep"]

    await runtime.activate()
    assert runtime._inner_runtime is inner
    assert runtime._workers_offloaded is False
    assert runtime._active_policy_version == 2
    assert inner.calls == ["sleep", "wake", ("update", "W2", 2)]


@pytest.mark.asyncio
async def test_activation_does_not_reinstall_an_already_active_policy() -> None:
    runtime = _on_demand_runtime()
    inner = _FakeInner()
    runtime._inner_runtime = inner
    runtime._workers_offloaded = True
    runtime._active_policy_version = 2
    await _set_desired_policy(runtime, "W2", 2)

    await runtime.activate()

    assert inner.calls == ["wake"]
    assert runtime._active_policy_version == 2


@pytest.mark.asyncio
async def test_active_update_publishes_only_after_inner_ack() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W1", 1)
    inner = _BlockingRestoreInner()
    runtime._inner_runtime = inner
    runtime._active_policy_version = 1

    update = asyncio.create_task(runtime.update_weights("W2", 2))
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    _assert_desired_policy(runtime, "W1", 1)
    assert runtime._active_policy_version == 1
    assert runtime.current_policy_version == 1

    inner.finish_restore.set()
    await asyncio.wait_for(update, timeout=1)
    _assert_desired_policy(runtime, "W2", 2)
    assert runtime._active_policy_version == 2
    assert runtime.current_policy_version == 2


@pytest.mark.asyncio
async def test_active_update_failure_preserves_desired_policy_and_terminates() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W1", 1)
    update_error = RuntimeError("worker update failed")

    class _FailingInner(_FakeInner):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise update_error

    inner = _FailingInner()
    runtime._inner_runtime = inner

    with pytest.raises(RuntimeError, match="worker update failed") as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is update_error
    _assert_desired_policy(runtime, "W1", 1)
    assert runtime.current_policy_version == 1
    assert runtime.lifecycle.failure is update_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert inner.calls == ["shutdown"]


@pytest.mark.asyncio
async def test_active_update_health_race_does_not_publish_outer_version() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W1", 1)
    runtime._active_policy_version = 1
    health_failure = RolloutWorkerUnreachable(
        "rollout-1",
        0.5,
        TimeoutError("health probe timed out"),
    )

    class _HealthRaceInner(_FakeInner):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            self.lifecycle.fail(health_failure)

    inner = _HealthRaceInner()
    inner.current_policy_version = 1
    runtime._inner_runtime = inner

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is health_failure
    _assert_desired_policy(runtime, "W1", 1)
    assert runtime.current_policy_version == 1
    assert runtime._active_policy_version is None
    assert runtime._inner_runtime is None
    assert inner.current_policy_version == 2
    assert inner.calls == [("update", "W2", 2), "shutdown"]
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_wake_health_race_terminalizes_outer_before_active_publication() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W2", 2)
    runtime._active_policy_version = 1
    runtime._workers_offloaded = True
    health_failure = RolloutWorkerUnreachable(
        "rollout-1",
        0.5,
        TimeoutError("health probe timed out"),
    )

    class _WakeHealthRaceInner(_FakeInner):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            self.lifecycle.fail(health_failure)

        async def shutdown(self) -> None:
            await super().shutdown()
            self.lifecycle.finish_shutdown()

    inner = _WakeHealthRaceInner()
    inner.current_policy_version = 1
    runtime._inner_runtime = inner

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.activate()

    assert caught.value is health_failure
    assert inner.calls == ["wake", ("update", "W2", 2), "shutdown"]
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert runtime._active_policy_version is None
    with pytest.raises(RuntimeLifecycleError, match="terminated") as retry:
        await runtime.activate()
    assert retry.value.__cause__ is health_failure


@pytest.mark.asyncio
async def test_on_demand_active_health_race_propagates_inner_first_failure() -> None:
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
    inner: RayGenerationRuntime

    class _Executor:
        async def execute(self, _request: Any) -> None:
            inner.lifecycle.fail(health_failure)
            raise actor_error

    inner = RayGenerationRuntime(_Executor())
    runtime._inner_runtime = inner
    request = SimpleNamespace(
        request_id="health-race",
        sampling={},
        policy_version=None,
    )

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.generate(request)

    assert caught.value is health_failure
    assert failure_identity_cause(caught.value) is health_failure
    assert inner.lifecycle.failure is health_failure
    assert runtime.lifecycle.failure is health_failure
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None


@pytest.mark.asyncio
async def test_active_on_demand_timeout_force_kills_the_inner_owner(
    monkeypatch,
) -> None:
    import vrl.generation.ray.runtime as runtime_module

    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W1", 1)
    timeout = RayOperationTimeout("rollout.weight_sync", 1.0)
    inner, actor = _timeout_inner(timeout)
    runtime._inner_runtime = inner
    runtime._active_policy_version = 1

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, target: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    monkeypatch.setattr(runtime_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is timeout
    assert runtime.current_policy_version == 1
    assert inner.current_policy_version == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_on_demand_auto_probe_timeout_force_kills_the_inner_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.runtime as runtime_module
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
    worker = DistributedWorkerHandle(worker_id="w0", actor=actor)
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [worker],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=0.01,
    )
    inner = RayGenerationRuntime(
        executor,
        owned_workers=[worker],
    )
    runtime._inner_runtime = inner

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

    monkeypatch.setattr(runtime_module, "require_ray", lambda: _Ray)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    request = SimpleNamespace(
        request_id="req-auto-timeout",
        samples_per_prompt=2,
        sampling={"samples_per_chunk": "auto"},
        policy_version=None,
    )

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.generate(request)

    assert caught.value.operation == "rollout.generation.chunk_size_probe"
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert actor.release_calls == 0
    assert _Ray.cancelled == [probe_ref]
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_on_demand_submitted_cancellation_force_kills_the_inner_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.runtime as runtime_module

    runtime = _on_demand_runtime()
    terminal = RayOperationCancelled("rollout.generation.pipelined")
    cancellation = asyncio.CancelledError()
    cancellation.__cause__ = terminal

    class _Executor:
        async def execute(self, _request: Any) -> None:
            raise cancellation

    actor = _ReleaseActor()
    inner = RayGenerationRuntime(
        _Executor(),
        owned_workers=[DistributedWorkerHandle(worker_id="w0", actor=actor)],
    )
    runtime._inner_runtime = inner

    class _Ray:
        killed: ClassVar[list[Any]] = []

        @classmethod
        def kill(cls, target: Any, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    monkeypatch.setattr(runtime_module, "require_ray", lambda: _Ray)
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
    assert inner.lifecycle.failure is terminal
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_update_cleanup_failure_retains_inner_for_retry() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W1", 1)
    update_error = RuntimeError("worker update failed")
    cleanup_error = RuntimeError("worker cleanup failed")

    class _FailingInner(_FakeInner):
        shutdown_calls = 0

        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise update_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise cleanup_error
            self.calls.append("shutdown")

    inner = _FailingInner()
    runtime._inner_runtime = inner

    with pytest.raises(RuntimeError, match="worker update failed") as caught:
        await runtime.update_weights("W2", 2)
    assert caught.value is update_error
    assert runtime.lifecycle.failure is update_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._inner_runtime is inner

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert runtime._inner_runtime is None
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
    inner, actor = _failed_parking_runtime(health_failure)
    runtime = _on_demand_runtime()
    runtime._inner_runtime = inner

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.offload()

    assert caught.value is health_failure
    assert inner.lifecycle.failure is health_failure
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
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
    inner, actor = _failed_parking_runtime(health_failure)
    runtime = _on_demand_runtime()
    runtime._inner_runtime = inner

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.activate()

    assert caught.value is health_failure
    assert inner.lifecycle.failure is health_failure
    assert inner.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert actor.release_policy.calls == 0
    assert cleanup_ray.killed == [actor]


@pytest.mark.asyncio
async def test_offload_is_single_flight_and_idempotent() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepInner()
    runtime._inner_runtime = inner

    first = asyncio.create_task(runtime.offload())
    await asyncio.wait_for(inner.sleep_started.wait(), timeout=1)
    second = asyncio.create_task(runtime.offload())
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()

    inner.finish_sleep.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    await runtime.offload()
    assert inner.calls == ["sleep"]
    assert runtime._workers_offloaded is True


@pytest.mark.asyncio
async def test_terminal_failure_cancels_offload_with_stable_root() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepInner()
    runtime._inner_runtime = inner
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
async def test_terminal_failure_cancels_offload_waiting_for_activation_with_stable_root() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W", 3)
    candidate = _BlockingRestoreInner()
    timeout = RayOperationTimeout("rollout.generation.activation", 0.5)

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    runtime._launcher = _Launcher()
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)
    offload = asyncio.create_task(runtime.offload())
    await asyncio.sleep(0)
    terminalize = asyncio.create_task(runtime._terminalize_after_failure(timeout))

    with pytest.raises(asyncio.CancelledError) as caught:
        await offload
    with pytest.raises(asyncio.CancelledError):
        await activation
    assert await asyncio.wait_for(terminalize, timeout=1) is timeout

    assert caught.value.__cause__ is timeout
    assert failure_identity_cause(caught.value) is timeout
    assert candidate.calls == ["shutdown"]
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_waiter_cancellation_does_not_cancel_offload_or_forge_root() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepInner()
    runtime._inner_runtime = inner

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
    assert runtime._workers_offloaded is True
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


@pytest.mark.asyncio
async def test_offload_failure_runs_terminal_cleanup() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")

    class _FailingInner(_FakeInner):
        async def sleep_workers(self) -> None:
            raise offload_error

    inner = _FailingInner()
    runtime._inner_runtime = inner

    with pytest.raises(RuntimeError, match="sleep failed") as caught:
        await runtime.offload()
    assert caught.value is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert inner.calls == ["shutdown"]


@pytest.mark.asyncio
async def test_offload_cleanup_failure_retains_inner_for_shutdown_retry() -> None:
    runtime = _on_demand_runtime()
    offload_error = RuntimeError("sleep failed")
    cleanup_error = RuntimeError("worker cleanup failed")

    class _FailingInner(_FakeInner):
        shutdown_calls = 0

        async def sleep_workers(self) -> None:
            raise offload_error

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise cleanup_error
            self.calls.append("shutdown")

    inner = _FailingInner()
    runtime._inner_runtime = inner

    with pytest.raises(RuntimeError, match="worker cleanup failed") as caught:
        await runtime.offload()
    assert caught.value is cleanup_error
    assert caught.value.__cause__ is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert runtime._inner_runtime is inner

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert runtime._inner_runtime is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_concurrent_cold_activation_launches_and_restores_once() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W", 3)
    candidate = _BlockingRestoreInner()
    launch_calls = 0

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            nonlocal launch_calls
            del args, kwargs
            launch_calls += 1
            return candidate

    runtime._launcher = _Launcher()
    first = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)
    second = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)

    assert launch_calls == 1
    assert runtime._inner_runtime is None
    assert not first.done()
    assert not second.done()

    candidate.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert runtime._inner_runtime is candidate
    assert runtime._active_policy_version == 3
    assert candidate.calls == [("update", "W", 3)]


@pytest.mark.asyncio
async def test_concurrent_activation_wakes_and_restores_once() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingRestoreInner()
    runtime._inner_runtime = inner
    runtime._workers_offloaded = True
    await _set_desired_policy(runtime, "W", 3)

    first = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    assert runtime._workers_offloaded is False
    assert runtime._activation_task is not None
    second = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()

    inner.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert inner.calls == ["wake", ("update", "W", 3)]


@pytest.mark.asyncio
async def test_update_rejects_activation_overlap_while_offload_joins_it() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W1", 1)
    candidate = _BlockingRestoreInner()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    runtime._launcher = _Launcher()
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="activation to be idle"):
        await runtime.update_weights("W2", 2)
    offload = asyncio.create_task(runtime.offload())
    await asyncio.sleep(0)
    assert not offload.done()
    _assert_desired_policy(runtime, "W1", 1)

    candidate.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(activation, offload), timeout=1)
    assert candidate.calls == [("update", "W1", 1), "sleep"]
    assert runtime._workers_offloaded is True


@pytest.mark.asyncio
async def test_shutdown_joins_activation_after_waiter_cancellation() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W", 4)
    candidate = _BlockingRestoreInner()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    runtime._launcher = _Launcher()
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
    assert runtime._inner_runtime is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_external_terminal_shutdown_is_the_only_activation_cleanup_owner() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W", 4)
    candidate = _BlockingRestoreInner()
    timeout = RayOperationTimeout("rollout.generation.chunk", 0.5)
    finish_calls = 0
    finish_shutdown = runtime.lifecycle.finish_shutdown

    def count_finish_shutdown() -> None:
        nonlocal finish_calls
        finish_calls += 1
        finish_shutdown()

    runtime.lifecycle.finish_shutdown = count_finish_shutdown

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    runtime._launcher = _Launcher()
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)

    terminalize = asyncio.create_task(runtime._terminalize_after_failure(timeout))
    await asyncio.sleep(0)
    shutdown_waiter = asyncio.create_task(runtime.shutdown())

    assert await asyncio.wait_for(terminalize, timeout=1) is timeout
    await asyncio.wait_for(shutdown_waiter, timeout=1)
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await activation

    assert cancelled.value.__cause__ is timeout
    assert candidate.calls == ["shutdown"]
    assert finish_calls == 1
    assert runtime.lifecycle.failure is timeout
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime._inner_runtime is None
    assert not any("cleanup also failed" in note for note in getattr(timeout, "__notes__", ()))


@pytest.mark.asyncio
async def test_late_shutdown_joins_activation_owned_terminal_cleanup() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W2", 2)
    runtime._active_policy_version = 1
    runtime._workers_offloaded = True
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

    class _BlockingShutdownInner(_FakeInner):
        def __init__(self) -> None:
            super().__init__()
            self.shutdown_started = asyncio.Event()
            self.finish_shutdown = asyncio.Event()

        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            self.lifecycle.fail(health_failure)

        async def shutdown(self) -> None:
            self.calls.append("shutdown")
            self.shutdown_started.set()
            await self.finish_shutdown.wait()
            self.lifecycle.finish_shutdown()

    inner = _BlockingShutdownInner()
    inner.current_policy_version = 1
    runtime._inner_runtime = inner

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
    assert runtime._inner_runtime is None
    assert not any(
        "cleanup also failed" in note for note in getattr(health_failure, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_shutdown_joins_offload_before_teardown() -> None:
    runtime = _on_demand_runtime()
    inner = _BlockingSleepInner()
    runtime._inner_runtime = inner

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
    await _set_desired_policy(runtime, "W", 3)
    restore_error = RuntimeError("restore failed")

    class _Candidate(_FakeInner):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise restore_error

    candidate = _Candidate()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    runtime._launcher = _Launcher()
    with pytest.raises(RuntimeError, match="restore failed") as caught:
        await runtime.activate()

    assert caught.value is restore_error
    assert candidate.calls == ["shutdown"]
    assert runtime._inner_runtime is None
    assert runtime.lifecycle.failure is restore_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cold_restore_timeout_force_kills_unpublished_candidate(
    monkeypatch,
) -> None:
    import vrl.generation.ray.runtime as runtime_module

    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W2", 2)
    timeout = RayOperationTimeout("rollout.weight_sync", 1.0)
    candidate, actor = _timeout_inner(timeout)

    class _Launcher:
        async def launch_async(self, *_args, **_kwargs):
            return candidate

    class _Ray:
        killed: ClassVar[list[object]] = []

        @classmethod
        def kill(cls, target: object, *, no_restart: bool) -> None:
            assert no_restart is True
            cls.killed.append(target)

    runtime._launcher = _Launcher()
    monkeypatch.setattr(runtime_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout) as caught:
        await runtime.activate()

    assert caught.value is timeout
    assert runtime._inner_runtime is None
    assert runtime._active_policy_version is None
    assert runtime.current_policy_version == 2
    assert candidate.current_policy_version == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert candidate.lifecycle.phase is RuntimePhase.TERMINATED
    assert actor.release_calls == 0
    assert _Ray.killed == [actor]


@pytest.mark.asyncio
async def test_cold_candidate_health_failure_terminalizes_outer_before_publication() -> None:
    runtime = _on_demand_runtime()
    await _set_desired_policy(runtime, "W2", 2)
    health_failure = RolloutWorkerUnreachable(
        "rollout-0",
        0.5,
        TimeoutError("health probe timed out"),
    )
    launch_calls = 0

    class _ColdHealthRaceCandidate(_FakeInner):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            await super().update_weights(state_ref, version)
            self.lifecycle.fail(health_failure)

        async def shutdown(self) -> None:
            await super().shutdown()
            self.lifecycle.finish_shutdown()

    candidate = _ColdHealthRaceCandidate()

    class _Launcher:
        async def launch_async(self, *_args: Any, **_kwargs: Any):
            nonlocal launch_calls
            launch_calls += 1
            return candidate

    runtime._launcher = _Launcher()

    with pytest.raises(RolloutWorkerUnreachable) as caught:
        await runtime.activate()

    assert caught.value is health_failure
    assert candidate.calls == [("update", "W2", 2), "shutdown"]
    assert candidate.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.lifecycle.failure is health_failure
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    _assert_desired_policy(runtime, "W2", 2)
    assert runtime._inner_runtime is None
    assert runtime._active_policy_version is None
    with pytest.raises(RuntimeLifecycleError, match="terminated") as retry:
        await runtime.activate()
    assert retry.value.__cause__ is health_failure
    assert launch_calls == 1


@pytest.mark.asyncio
async def test_activation_launch_failure_terminalizes_without_publishing_candidate() -> None:
    runtime = _on_demand_runtime()
    launch_error = RuntimeError("rollout worker startup failed")

    class _Launcher:
        async def launch_async(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise launch_error

    runtime._launcher = _Launcher()

    with pytest.raises(RuntimeError, match="rollout worker startup failed") as caught:
        await runtime.activate()

    assert caught.value is launch_error
    assert runtime._inner_runtime is None
    assert runtime.lifecycle.failure is launch_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_on_demand_shutdown_is_idempotent_for_offloaded_workers() -> None:
    runtime = _on_demand_runtime()
    runtime._workers_offloaded = True
    inner = _FakeInner()
    runtime._inner_runtime = inner

    await runtime.shutdown()
    await runtime.shutdown()

    assert inner.calls == ["shutdown"]
    assert runtime._inner_runtime is None
    assert runtime._workers_offloaded is False
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_terminated_runtime_rejects_activation() -> None:
    runtime = _on_demand_runtime()
    await runtime.shutdown()

    with pytest.raises(RuntimeLifecycleError, match="terminated"):
        await runtime.activate()
