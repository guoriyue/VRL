"""Explicit activation/offload for shared-GPU Ray rollout workers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from vrl.generation.execution.types import (
    DistributedWorkerHandle,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.lifecycle_fsm import RuntimeLifecycleError, RuntimePhase
from vrl.generation.ray.runtime import (
    RayGenerationRuntime,
    _PolicySnapshot,
)
from vrl.trainers.weight_sync import build_runtime_weight_syncer


class _FakeInner:
    """Record ordered worker operations without requiring a Ray cluster."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.current_policy_version: int | None = 0

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


def _on_demand_resources(*, colocated: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        colocated=colocated,
        lifecycle=SimpleNamespace(
            rollout=SimpleNamespace(mode="on_demand"),
        ),
    )


def _launch_contract(*, policy_version: int = 0) -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="sd3_5",
        task="t2i",
        generation_kind="diffusion",
        policy_version=policy_version,
        runtime_builder="tests:runtime_builder",
        executor_cls="tests:executor_cls",
    )


def _on_demand_runtime(
    *,
    sync_trainable_state: bool = True,
    allow_driver_gpu_overlap: bool = True,
    gpus_per_worker: float = 0.0,
) -> RayGenerationRuntime:
    config = SimpleNamespace(
        sync_trainable_state=sync_trainable_state,
        gpus_per_worker=gpus_per_worker,
        allow_driver_gpu_overlap=allow_driver_gpu_overlap,
        resources=_on_demand_resources(colocated=allow_driver_gpu_overlap),
    )
    return RayGenerationRuntime.with_on_demand_activation(
        config,
        RayGenerationLaunchInputs(
            launch_contract=_launch_contract(),
            gatherer=SimpleNamespace(),
        ),
        placement=SimpleNamespace(),
    )


def _set_desired_policy(
    runtime: RayGenerationRuntime,
    state_ref: Any,
    policy_version: int,
) -> None:
    state = runtime._on_demand
    assert state is not None
    state.desired_policy = _PolicySnapshot(state_ref, policy_version)
    runtime.current_policy_version = policy_version


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


def _parking_runtime(*reports: Any) -> RayGenerationRuntime:
    workers = [
        DistributedWorkerHandle(
            worker_id=f"rollout-{index}",
            actor=_ParkingActor(report),
        )
        for index, report in enumerate(reports)
    ]
    return RayGenerationRuntime(SimpleNamespace(), owned_workers=workers)


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
async def test_runtime_rejects_mismatched_worker_parking_evidence() -> None:
    runtime = _parking_runtime(_parking_snapshot("another-worker"))

    with pytest.raises(RuntimeError, match="mismatched worker memory-parking report"):
        await runtime.sleep_workers()


@pytest.mark.asyncio
async def test_runtime_rejects_worker_gpu_residual() -> None:
    runtime = _parking_runtime(_parking_snapshot(residual_bytes=1))

    with pytest.raises(RuntimeError, match="incomplete cpu_offload memory parking"):
        await runtime.sleep_workers()


@pytest.mark.asyncio
async def test_runtime_requires_every_worker_parking_rpc_to_succeed() -> None:
    runtime = _parking_runtime(
        _parking_snapshot("rollout-0"),
        RuntimeError("worker sleep failed"),
    )

    with pytest.raises(RuntimeError, match="worker sleep failed"):
        await runtime.sleep_workers()


@pytest.mark.asyncio
async def test_runtime_rejects_missing_worker_actor_before_parking() -> None:
    runtime = RayGenerationRuntime(
        SimpleNamespace(),
        owned_workers=[DistributedWorkerHandle(worker_id="rollout-0")],
    )

    with pytest.raises(RuntimeError, match="generation workers have no actor"):
        await runtime.sleep_workers()


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


def test_on_demand_factory_stamps_contract_for_cumem_offload() -> None:
    runtime = _on_demand_runtime(gpus_per_worker=1.0)
    state = runtime._on_demand
    assert state is not None
    assert state.launch_inputs.launch_contract.extra["sleep_offload"] is True


def test_on_demand_weight_sync_capability_does_not_require_active_workers() -> None:
    runtime = _on_demand_runtime()
    disabled = _on_demand_runtime(sync_trainable_state=False)

    assert runtime.weight_sync is None
    assert runtime.supports_weight_sync is True
    assert build_runtime_weight_syncer(runtime) is not None
    assert disabled.supports_weight_sync is False
    assert build_runtime_weight_syncer(disabled) is None


def test_driver_model_offload_is_derived_from_actual_gpu_overlap() -> None:
    shared_gpu = _on_demand_runtime(
        allow_driver_gpu_overlap=True,
        gpus_per_worker=1.0,
    )
    separate_gpu = _on_demand_runtime(
        allow_driver_gpu_overlap=False,
        gpus_per_worker=1.0,
    )
    cpu_rollout = _on_demand_runtime(
        allow_driver_gpu_overlap=True,
        gpus_per_worker=0.0,
    )

    assert shared_gpu.requires_driver_model_offload is True
    assert separate_gpu.requires_driver_model_offload is False
    assert cpu_rollout.requires_driver_model_offload is False


@pytest.mark.parametrize("resources", [None, _on_demand_resources()])
def test_on_demand_factory_requires_on_demand_plan(resources) -> None:
    if resources is not None:
        resources.lifecycle.rollout.mode = "resident"
    config = SimpleNamespace(
        sync_trainable_state=False,
        gpus_per_worker=0.0,
        allow_driver_gpu_overlap=False,
        resources=resources,
    )

    with pytest.raises(ValueError, match="resolved on-demand"):
        RayGenerationRuntime.with_on_demand_activation(
            config,
            RayGenerationLaunchInputs(
                launch_contract=_launch_contract(),
                gatherer=SimpleNamespace(),
            ),
            placement=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_generate_requires_explicit_activation() -> None:
    runtime = _on_demand_runtime()
    request = SimpleNamespace(sampling={}, policy_version=None)

    with pytest.raises(RuntimeError, match=r"await activate\(\) first"):
        await runtime.generate(request)


@pytest.mark.asyncio
async def test_cold_weights_are_staged_then_applied_during_activation(monkeypatch) -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    candidate = _FakeInner()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    import vrl.generation.ray.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "RayGenerationLauncher", _Launcher)
    await runtime.update_weights("W2", 2)

    assert state.desired_policy == _PolicySnapshot("W2", 2)
    assert state.inner_runtime is None
    await runtime.activate()
    assert state.inner_runtime is candidate
    assert state.active_policy_version == 2
    assert candidate.calls == [("update", "W2", 2)]


@pytest.mark.asyncio
async def test_offload_keeps_workers_and_activation_wakes_latest_policy() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    inner = _FakeInner()
    state.inner_runtime = inner
    state.active_policy_version = 1

    await runtime.offload()
    await runtime.update_weights("W2", 2)

    assert state.inner_runtime is inner
    assert state.workers_offloaded is True
    assert state.desired_policy == _PolicySnapshot("W2", 2)
    assert inner.calls == ["sleep"]

    await runtime.activate()
    assert state.inner_runtime is inner
    assert state.workers_offloaded is False
    assert state.active_policy_version == 2
    assert inner.calls == ["sleep", "wake", ("update", "W2", 2)]


@pytest.mark.asyncio
async def test_activation_does_not_reinstall_an_already_active_policy() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    inner = _FakeInner()
    state.inner_runtime = inner
    state.workers_offloaded = True
    state.active_policy_version = 2
    _set_desired_policy(runtime, "W2", 2)

    await runtime.activate()

    assert inner.calls == ["wake"]
    assert state.active_policy_version == 2


@pytest.mark.asyncio
async def test_active_update_publishes_only_after_inner_ack() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W1", 1)
    inner = _BlockingRestoreInner()
    state.inner_runtime = inner
    state.active_policy_version = 1

    update = asyncio.create_task(runtime.update_weights("W2", 2))
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    assert state.desired_policy == _PolicySnapshot("W1", 1)
    assert state.active_policy_version == 1
    assert runtime.current_policy_version == 1

    inner.finish_restore.set()
    await asyncio.wait_for(update, timeout=1)
    assert state.desired_policy == _PolicySnapshot("W2", 2)
    assert state.active_policy_version == 2
    assert runtime.current_policy_version == 2


@pytest.mark.asyncio
async def test_active_update_failure_preserves_desired_policy_and_terminates() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W1", 1)
    update_error = RuntimeError("worker update failed")

    class _FailingInner(_FakeInner):
        async def update_weights(self, state_ref: Any, version: int) -> None:
            del state_ref, version
            raise update_error

    inner = _FailingInner()
    state.inner_runtime = inner

    with pytest.raises(RuntimeError, match="worker update failed") as caught:
        await runtime.update_weights("W2", 2)

    assert caught.value is update_error
    assert state.desired_policy == _PolicySnapshot("W1", 1)
    assert runtime.current_policy_version == 1
    assert runtime.lifecycle.failure is update_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert state.inner_runtime is None
    assert inner.calls == ["shutdown"]


@pytest.mark.asyncio
async def test_update_cleanup_failure_retains_inner_for_retry() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W1", 1)
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
    state.inner_runtime = inner

    with pytest.raises(RuntimeError, match="worker update failed") as caught:
        await runtime.update_weights("W2", 2)
    assert caught.value is update_error
    assert runtime.lifecycle.failure is update_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert state.inner_runtime is inner

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert state.inner_runtime is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_offload_is_single_flight_and_idempotent() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    inner = _BlockingSleepInner()
    state.inner_runtime = inner

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
    assert state.workers_offloaded is True


@pytest.mark.asyncio
async def test_offload_failure_runs_terminal_cleanup() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    offload_error = RuntimeError("sleep failed")

    class _FailingInner(_FakeInner):
        async def sleep_workers(self) -> None:
            raise offload_error

    inner = _FailingInner()
    state.inner_runtime = inner

    with pytest.raises(RuntimeError, match="sleep failed") as caught:
        await runtime.offload()
    assert caught.value is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert state.inner_runtime is None
    assert inner.calls == ["shutdown"]


@pytest.mark.asyncio
async def test_offload_cleanup_failure_retains_inner_for_shutdown_retry() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
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
    state.inner_runtime = inner

    with pytest.raises(RuntimeError, match="worker cleanup failed") as caught:
        await runtime.offload()
    assert caught.value is cleanup_error
    assert caught.value.__cause__ is offload_error
    assert runtime.lifecycle.failure is offload_error
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    assert state.inner_runtime is inner

    await runtime.shutdown()
    assert inner.shutdown_calls == 2
    assert state.inner_runtime is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_concurrent_cold_activation_launches_and_restores_once(monkeypatch) -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W", 3)
    candidate = _BlockingRestoreInner()
    launch_calls = 0

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            nonlocal launch_calls
            del args, kwargs
            launch_calls += 1
            return candidate

    import vrl.generation.ray.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "RayGenerationLauncher", _Launcher)
    first = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)
    second = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)

    assert launch_calls == 1
    assert state.inner_runtime is None
    assert not first.done()
    assert not second.done()

    candidate.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert state.inner_runtime is candidate
    assert state.active_policy_version == 3
    assert candidate.calls == [("update", "W", 3)]


@pytest.mark.asyncio
async def test_concurrent_activation_wakes_and_restores_once() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    inner = _BlockingRestoreInner()
    state.inner_runtime = inner
    state.workers_offloaded = True
    _set_desired_policy(runtime, "W", 3)

    first = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(inner.restore_started.wait(), timeout=1)
    assert state.workers_offloaded is False
    assert state.activation_task is not None
    second = asyncio.create_task(runtime.activate())
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()

    inner.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert inner.calls == ["wake", ("update", "W", 3)]


@pytest.mark.asyncio
async def test_update_rejects_activation_overlap_while_offload_joins_it(
    monkeypatch,
) -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W1", 1)
    candidate = _BlockingRestoreInner()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    import vrl.generation.ray.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "RayGenerationLauncher", _Launcher)
    activation = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="activation to be idle"):
        await runtime.update_weights("W2", 2)
    offload = asyncio.create_task(runtime.offload())
    await asyncio.sleep(0)
    assert not offload.done()
    assert state.desired_policy == _PolicySnapshot("W1", 1)

    candidate.finish_restore.set()
    await asyncio.wait_for(asyncio.gather(activation, offload), timeout=1)
    assert candidate.calls == [("update", "W1", 1), "sleep"]
    assert state.workers_offloaded is True


@pytest.mark.asyncio
async def test_shutdown_joins_activation_after_waiter_cancellation(monkeypatch) -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W", 4)
    candidate = _BlockingRestoreInner()

    class _Launcher:
        async def launch_async(self, *args, **kwargs):
            del args, kwargs
            return candidate

    import vrl.generation.ray.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "RayGenerationLauncher", _Launcher)
    waiter = asyncio.create_task(runtime.activate())
    await asyncio.wait_for(candidate.restore_started.wait(), timeout=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert "shutdown" not in candidate.calls

    candidate.finish_restore.set()
    await asyncio.wait_for(shutdown, timeout=1)
    assert candidate.calls == [("update", "W", 4), "shutdown"]
    assert state.inner_runtime is None
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_shutdown_joins_offload_before_teardown() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    inner = _BlockingSleepInner()
    state.inner_runtime = inner

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
async def test_activation_restore_failure_cleans_candidate_and_terminates(monkeypatch) -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    _set_desired_policy(runtime, "W", 3)
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

    import vrl.generation.ray.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "RayGenerationLauncher", _Launcher)
    with pytest.raises(RuntimeError, match="restore failed") as caught:
        await runtime.activate()

    assert caught.value is restore_error
    assert candidate.calls == ["shutdown"]
    assert state.inner_runtime is None
    assert runtime.lifecycle.failure is restore_error
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_on_demand_shutdown_is_idempotent_for_offloaded_workers() -> None:
    runtime = _on_demand_runtime()
    state = runtime._on_demand
    assert state is not None
    state.workers_offloaded = True
    inner = _FakeInner()
    state.inner_runtime = inner

    await runtime.shutdown()
    await runtime.shutdown()

    assert inner.calls == ["shutdown"]
    assert state.inner_runtime is None
    assert state.workers_offloaded is False
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_terminated_runtime_rejects_activation() -> None:
    runtime = _on_demand_runtime()
    await runtime.shutdown()

    with pytest.raises(RuntimeLifecycleError, match="terminated"):
        await runtime.activate()
