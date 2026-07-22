"""Physical GPU-memory parking owned by one generation worker."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, get_args

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.models.interfaces.runtime import PipelineOffloadMode
from vrl.utils.cuda_memory import (
    CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT,
    CumemPool,
    gpu_used_bytes,
    release_cuda_memory,
    release_cuda_memory_for_parking,
)
from vrl.utils.logging import init_logger

if TYPE_CHECKING:
    from vrl.families.registry import GenerationParkingProfile
    from vrl.generation.protocols import GenerationChunkExecutor

logger = init_logger(__name__)


class _ParkingPhase(Enum):
    ACTIVE = "active"
    PARKED = "parked"
    QUARANTINED = "quarantined"
    CUMEM_BROKEN = "cumem_broken"


@dataclass(slots=True)
class _ModelParking:
    restore_device: Any | None = None


@dataclass(frozen=True, slots=True)
class _CumemParking:
    pool: CumemPool


_ParkingBackend = _ModelParking | _CumemParking


@dataclass(frozen=True, slots=True)
class _ParkingPlan:
    required: bool
    profile: GenerationParkingProfile


@dataclass(slots=True)
class _ParkingSession:
    required: bool
    profile: GenerationParkingProfile
    baseline_gpu_used_bytes: int | None
    backend: _ParkingBackend


class WorkerMemoryParking:
    """Own one worker's mutually exclusive CUDA-memory parking backend.

    The Ray launch contract carries only the topology-derived requirement to
    yield the GPU. This owner combines it with the resolved rollout residency
    mode and family capabilities before model construction, then retains the
    backend-specific state required for idempotent sleep, wake, and teardown.
    """

    def __init__(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract,
    ) -> None:
        if not worker_id:
            raise ValueError("worker memory parking requires a non-empty worker_id")
        if not isinstance(launch_contract, GenerationRuntimeLaunchContract):
            raise TypeError(
                "worker memory parking requires a GenerationRuntimeLaunchContract",
            )
        from vrl.families.registry import (
            GenerationParkingProfile,
            get_model_family_entry,
        )

        parking_profile = get_model_family_entry(
            launch_contract.family,
        ).runtime_capabilities.memory_parking

        rollout = launch_contract.model_build.get("rollout")
        pipeline_offload_mode = "none"
        if isinstance(rollout, Mapping):
            pipeline_offload_mode = str(
                rollout.get("pipeline_offload_mode", "none"),
            )
        allowed_modes = get_args(PipelineOffloadMode)
        if pipeline_offload_mode not in allowed_modes:
            raise ValueError(
                "pipeline_offload_mode must be one of "
                f"{list(allowed_modes)}, got {pipeline_offload_mode!r}",
            )
        if pipeline_offload_mode != "none":
            # Accelerate already owns the model's active residency, so the
            # worker must not claim a CuMem build scope around CPU/meta weights.
            profile = GenerationParkingProfile.MODEL
        elif parking_profile in {
            GenerationParkingProfile.CUMEM,
            GenerationParkingProfile.MODEL,
        }:
            profile = parking_profile
        else:
            raise TypeError(
                f"unsupported generation memory-parking profile: {parking_profile!r}",
            )
        self.worker_id = worker_id
        self._parking: _ParkingPlan | _ParkingSession = _ParkingPlan(
            required=launch_contract.sleep_offload,
            profile=profile,
        )
        self._phase = _ParkingPhase.ACTIVE
        # Persist only a lightweight quarantine reason. Exception tracebacks can
        # retain the model and its CUDA tensors past terminal cleanup.
        self._failure_reason: str | None = None

    @property
    def _is_parked(self) -> bool:
        return self._phase is _ParkingPhase.PARKED

    def build(
        self,
        build_executor: Callable[[], GenerationChunkExecutor],
    ) -> GenerationChunkExecutor:
        """Build once, claiming a CuMem pool only for eager-resident weights."""

        self.require_active("policy build", executor=None)
        state = self._parking
        if isinstance(state, _ParkingSession):
            raise RuntimeError(
                f"generation worker {self.worker_id!r} already owns a loaded "
                "memory-parking backend",
            )
        baseline_gpu_used_bytes = None
        if state.required:
            # Device-wide rather than process-local: the topology must keep other
            # owners stable until the corresponding residual sample is taken.
            baseline_gpu_used_bytes = gpu_used_bytes()
        self._phase = _ParkingPhase.ACTIVE
        self._failure_reason = None

        from vrl.families.registry import GenerationParkingProfile

        if not (state.required and state.profile is GenerationParkingProfile.CUMEM):
            executor = build_executor()
            self._parking = _ParkingSession(
                required=state.required,
                profile=state.profile,
                baseline_gpu_used_bytes=baseline_gpu_used_bytes,
                backend=_ModelParking(),
            )
            return executor

        pool = CumemPool.try_create(tag=f"vrl:generation:{self.worker_id}:weights")
        if pool is None:
            executor = build_executor()
            self._parking = _ParkingSession(
                required=state.required,
                profile=state.profile,
                baseline_gpu_used_bytes=baseline_gpu_used_bytes,
                backend=_ModelParking(),
            )
            return executor
        backend = _CumemParking(pool)
        try:
            with pool.building():
                executor = build_executor()
        except BaseException as build_error:
            # Drop traceback-held builder locals before touching vLLM's retained
            # MemPool registry while preserving the diagnostic stack itself.
            if build_error.__traceback__ is not None:
                traceback.clear_frames(build_error.__traceback__)
            release_cuda_memory(gc_collect=True, ipc_collect=True)
            try:
                pool.close()
            except BaseException as close_error:
                self._parking = _ParkingSession(
                    required=state.required,
                    profile=state.profile,
                    baseline_gpu_used_bytes=baseline_gpu_used_bytes,
                    backend=backend,
                )
                self._quarantine(
                    f"CuMem cleanup failed after pooled build error: {close_error!r}",
                    cumem_broken=True,
                )
                raise RuntimeError(
                    "generation pooled build and cleanup both failed: "
                    f"build={build_error!r}; cleanup={close_error!r}",
                ) from close_error
            raise
        self._parking = _ParkingSession(
            required=state.required,
            profile=state.profile,
            baseline_gpu_used_bytes=baseline_gpu_used_bytes,
            backend=backend,
        )
        return executor

    def validate_loaded(self, executor: GenerationChunkExecutor) -> None:
        """Fail before serving when no complete backend owns the loaded model."""

        session = self._loaded_session()
        backend = session.backend
        if not session.required:
            return
        model = getattr(executor, "model", None)
        uses_pipeline_offload = bool(
            getattr(model, "uses_pipeline_cpu_offload", False),
        )
        if isinstance(backend, _CumemParking):
            if uses_pipeline_offload:
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} selected both CuMem and "
                    "pipeline CPU offload; residency mechanisms must be mutually exclusive",
                )
            return
        if uses_pipeline_offload:
            reset_offload = getattr(model, "reset_pipeline_cpu_offload", None)
            if not callable(reset_offload):
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} cannot completely park "
                    f"{type(model).__name__}: pipeline CPU offload requires "
                    "reset_pipeline_cpu_offload()",
                )
            if not bool(getattr(model, "pipeline_cpu_offload_healthy", False)):
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} cannot completely park "
                    f"{type(model).__name__}: pipeline CPU offload hooks are unhealthy",
                )
            return
        if model is None or not callable(getattr(model, "to", None)):
            raise RuntimeError(
                f"generation worker {self.worker_id!r} cannot completely park "
                f"{type(executor).__name__}: CPU fallback requires executor.model.to(...)"
            )
        from vrl.families.registry import GenerationParkingProfile

        if session.profile is GenerationParkingProfile.CUMEM and not callable(
            getattr(model, "move_frozen_components", None),
        ):
            raise RuntimeError(
                f"generation worker {self.worker_id!r} cannot completely park "
                f"{type(model).__name__}: this CPU fallback requires "
                "move_frozen_components(...)"
            )

    def sleep(
        self,
        executor: GenerationChunkExecutor | None,
        *,
        restore_device: Any,
    ) -> WorkerMemoryParkingSnapshot:
        """Park the loaded executor and return physical handoff evidence."""

        self.require_healthy("sleep", executor=executor)
        if executor is None:
            if self._parking.required:
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} cannot park before policy load",
                )
            used_bytes = gpu_used_bytes()
            snapshot = WorkerMemoryParkingSnapshot(
                worker_id=self.worker_id,
                backend="cpu_only",
                baseline_gpu_used_bytes=used_bytes,
                loaded_gpu_used_bytes=used_bytes,
                residual_gpu_used_bytes=used_bytes,
                residual_bytes_limit=0,
            )
            snapshot.validate()
            return snapshot

        self.validate_loaded(executor)
        session = self._loaded_session()
        loaded_bytes = gpu_used_bytes()
        model = executor.model
        backend = session.backend

        if isinstance(backend, _CumemParking):
            try:
                if not backend.pool.asleep:
                    backend.pool.sleep()
            except BaseException as error:
                self._quarantine(
                    f"CuMem sleep failed and may have partially unmapped allocations: {error!r}",
                    cumem_broken=True,
                )
                raise
            snapshot_backend = "cumem"
            residual_bytes_limit = CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
        elif bool(getattr(model, "uses_pipeline_cpu_offload", False)):
            if not self._is_parked:
                self._reset_pipeline_cpu_offload(model, operation="sleep")
            snapshot_backend = "cpu_offload"
            residual_bytes_limit = CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
        else:
            if backend.restore_device is None:
                self._park_model_on_cpu(
                    model,
                    backend=backend,
                    restore_device=restore_device,
                )
            snapshot_backend = (
                "cpu_offload" if str(backend.restore_device).startswith("cuda") else "cpu_only"
            )
            residual_bytes_limit = (
                CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT if snapshot_backend == "cpu_offload" else 0
            )

        try:
            release_cuda_memory_for_parking()
            residual_bytes = gpu_used_bytes()
            baseline_bytes = session.baseline_gpu_used_bytes
            if baseline_bytes is None:
                if session.required or residual_bytes:
                    raise RuntimeError(
                        f"generation worker {self.worker_id!r} has no pre-load GPU "
                        "parking baseline; load it with sleep_offload enabled",
                    )
                baseline_bytes = 0
            snapshot = WorkerMemoryParkingSnapshot(
                worker_id=self.worker_id,
                backend=snapshot_backend,
                baseline_gpu_used_bytes=baseline_bytes,
                loaded_gpu_used_bytes=loaded_bytes,
                residual_gpu_used_bytes=residual_bytes,
                residual_bytes_limit=residual_bytes_limit,
            )
            snapshot.validate()
        except BaseException as validation_error:
            self._quarantine(
                "physical GPU parking validation failed after backend "
                f"{snapshot_backend!r} completed: {validation_error!r}",
            )
            raise RuntimeError(
                f"generation worker {self.worker_id!r} failed physical GPU "
                f"parking validation after {snapshot_backend}: {validation_error}",
            ) from validation_error
        logger.info(
            "worker memory parking: worker=%s backend=%s loaded=%d "
            "residual=%d baseline=%d limit=%d",
            snapshot.worker_id,
            snapshot.backend,
            snapshot.loaded_gpu_used_bytes,
            snapshot.residual_gpu_used_bytes,
            snapshot.baseline_gpu_used_bytes,
            snapshot.residual_bytes_limit,
        )
        self._phase = _ParkingPhase.PARKED
        return snapshot

    def wake(self, executor: GenerationChunkExecutor) -> None:
        """Restore physical residency required before generation resumes."""

        self.require_healthy("wake", executor=executor)
        if not self._is_parked:
            return
        model = getattr(executor, "model", None)
        session = self._loaded_session()
        backend = session.backend
        if isinstance(backend, _CumemParking):
            try:
                backend.pool.wake()
            except BaseException as error:
                self._quarantine(
                    f"CuMem wake failed and may have partially remapped allocations: {error!r}",
                    cumem_broken=True,
                )
                raise
            self._phase = _ParkingPhase.ACTIVE
            return
        if bool(getattr(model, "uses_pipeline_cpu_offload", False)):
            # Accelerate hooks remain installed; the next forward performs the
            # actual per-layer or per-module onload.
            self._phase = _ParkingPhase.ACTIVE
            return
        device = backend.restore_device
        if device is None:
            reason = "module CPU parking lost its restore device"
            self._quarantine(reason)
            raise RuntimeError(
                f"generation worker {self.worker_id!r} {reason}",
            )
        move = getattr(model, "to", None)
        if callable(move):
            move(device)
        move_frozen = getattr(model, "move_frozen_components", None)
        if callable(move_frozen):
            move_frozen(device)
        # Preserve the target when either move raises so wake can be retried.
        backend.restore_device = None
        self._phase = _ParkingPhase.ACTIVE

    @contextmanager
    def release_scope(self) -> Iterator[None]:
        """Release state after the caller drops its executor reference."""

        state = self._parking
        backend = state.backend if isinstance(state, _ParkingSession) else None
        pool = backend.pool if isinstance(backend, _CumemParking) else None
        if self._phase is _ParkingPhase.CUMEM_BROKEN:
            raise RuntimeError(
                f"generation worker {self.worker_id!r} has an indeterminate CuMem "
                "mapping and must terminate instead of closing the pool in process",
            )
        if pool is not None and pool.asleep:
            try:
                pool.wake()
            except BaseException as error:
                self._quarantine(
                    f"CuMem wake during release failed: {error!r}",
                    cumem_broken=True,
                )
                raise
        yield
        release_cuda_memory(gc_collect=True, ipc_collect=True)
        if pool is not None:
            try:
                pool.close()
            except BaseException as error:
                self._quarantine(
                    f"CuMem terminal close failed: {error!r}",
                    cumem_broken=True,
                )
                raise
            release_cuda_memory(gc_collect=True, ipc_collect=True)
        self._parking = self._next_plan(state)
        self._phase = _ParkingPhase.ACTIVE
        self._failure_reason = None

    def require_healthy(
        self,
        operation: str,
        *,
        executor: GenerationChunkExecutor | None = None,
    ) -> None:
        """Reject operations after residency or policy state became unknowable."""

        if self._phase in {
            _ParkingPhase.QUARANTINED,
            _ParkingPhase.CUMEM_BROKEN,
        }:
            raise RuntimeError(
                f"generation worker {self.worker_id!r} is quarantined after a "
                f"memory-parking failure; refusing {operation}: "
                f"{self._failure_reason}",
            )
        model = getattr(executor, "model", None)
        if bool(getattr(model, "uses_pipeline_cpu_offload", False)) and not bool(
            getattr(model, "pipeline_cpu_offload_healthy", False),
        ):
            reason = f"{type(model).__name__} reports unhealthy pipeline CPU-offload hooks"
            self._quarantine(reason)
            raise RuntimeError(
                f"generation worker {self.worker_id!r} is quarantined; "
                f"refusing {operation}: {reason}",
            )

    def require_active(
        self,
        operation: str,
        *,
        executor: GenerationChunkExecutor | None,
    ) -> None:
        """Reject execution while this worker is logically parked."""

        self.require_healthy(operation, executor=executor)
        if self._is_parked:
            raise RuntimeError(
                f"generation worker {self.worker_id!r} is parked; refusing {operation} until wake",
            )

    def recover_after_execution_error(
        self,
        model: Any,
        execution_error: BaseException,
    ) -> None:
        """Re-arm pipeline hooks after a failed forward before retry."""

        if not bool(getattr(model, "uses_pipeline_cpu_offload", False)):
            return
        self._reset_pipeline_cpu_offload(
            model,
            operation="error recovery",
            execution_error=execution_error,
        )

    def record_model_failure(self, model: Any, error: BaseException) -> None:
        """Quarantine a model-reported partial hook or weight mutation."""

        if not bool(getattr(model, "uses_pipeline_cpu_offload", False)):
            return
        if bool(getattr(model, "pipeline_cpu_offload_healthy", False)):
            return
        self._quarantine(f"{type(error).__name__}: {error}")

    def _loaded_session(self) -> _ParkingSession:
        state = self._parking
        if isinstance(state, _ParkingSession):
            return state
        session = _ParkingSession(
            required=state.required,
            profile=state.profile,
            baseline_gpu_used_bytes=None,
            backend=_ModelParking(),
        )
        self._parking = session
        return session

    @staticmethod
    def _next_plan(state: _ParkingPlan | _ParkingSession) -> _ParkingPlan:
        if isinstance(state, _ParkingPlan):
            return state
        return _ParkingPlan(required=state.required, profile=state.profile)

    def _park_model_on_cpu(
        self,
        model: Any,
        *,
        backend: _ModelParking,
        restore_device: Any,
    ) -> None:
        move_frozen = getattr(model, "move_frozen_components", None)
        try:
            model.to("cpu")
            if callable(move_frozen):
                move_frozen("cpu")
        except BaseException as move_error:
            rollback_errors: list[BaseException] = []
            try:
                model.to(restore_device)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
            if callable(move_frozen):
                try:
                    move_frozen(restore_device)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                reason = (
                    "generation CPU parking and rollback both failed: "
                    f"move={move_error!r}; rollback={rollback_errors!r}"
                )
                self._quarantine(reason)
                raise RuntimeError(reason) from rollback_errors[0]
            raise
        backend.restore_device = restore_device

    def _reset_pipeline_cpu_offload(
        self,
        model: Any,
        *,
        operation: str,
        execution_error: BaseException | None = None,
    ) -> None:
        reset_offload = getattr(model, "reset_pipeline_cpu_offload", None)
        if not callable(reset_offload):
            reset_error: BaseException = RuntimeError(
                f"{type(model).__name__} does not implement reset_pipeline_cpu_offload()",
            )
        else:
            try:
                reset_offload()
                return
            except BaseException as error:
                reset_error = error

        reason = (
            f"pipeline CPU-offload {operation} failed; the worker is quarantined: {reset_error!r}"
        )
        self._quarantine(reason)
        failure = RuntimeError(
            f"generation worker {self.worker_id!r} {reason}",
        )
        if execution_error is not None:
            failure.add_note(f"original generation failure: {execution_error!r}")
        raise failure from reset_error

    def _quarantine(self, reason: str, *, cumem_broken: bool = False) -> None:
        self._failure_reason = self._failure_reason or str(reason)
        if cumem_broken or self._phase is _ParkingPhase.CUMEM_BROKEN:
            # CuMem mutates allocations one by one without rollback. An allocator
            # exception leaves mappings unknowable; only actor termination is safe.
            self._phase = _ParkingPhase.CUMEM_BROKEN
        else:
            self._phase = _ParkingPhase.QUARANTINED


__all__ = ["WorkerMemoryParking"]
