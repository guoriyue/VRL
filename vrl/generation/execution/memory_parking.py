"""Physical GPU-memory parking owned by one generation worker.

The steps every role shares (pooled build, park with rollback, restore,
release, the physical evidence) live in :class:`vrl.models.parking.ParkingSession`.
This owner adds what only a Ray generation worker needs: the mechanism choice
from residency, the quarantine phases that tell the driver to terminate the
actor instead of retrying, pipeline-CPU-offload hook health, and the
``WorkerMemoryParkingSnapshot`` the driver validates before handing the GPU to
the trainer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.models import parking
from vrl.models.interfaces.runtime import PipelineOffloadMode
from vrl.models.parking import CumemBroken, ModelParking, ParkingBroken, ParkingSession
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT, release_cuda_memory
from vrl.utils.logging import init_logger

if TYPE_CHECKING:
    from vrl.generation.protocols import GenerationBatchExecutor

logger = init_logger(__name__)


def _log_parking_diagnostics(model: Any, *, worker_id: str) -> None:
    """Report tensor residency without replacing the original handoff failure."""
    try:
        import torch

        components = getattr(getattr(model, "pipeline", None), "components", None)
        if not isinstance(components, Mapping):
            components = {"model": model}
        residency = {}
        for name, component in components.items():
            if isinstance(component, torch.nn.Module):
                tensors = (*component.parameters(), *component.buffers())
                residency[name] = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors
                    if tensor.device.type == "cuda"
                )
        logger.error(
            "parking diagnostics: worker=%s allocated=%d reserved=%d component_cuda_bytes=%s",
            worker_id,
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            residency,
        )
    except Exception:
        logger.exception("parking diagnostics unavailable: worker=%s", worker_id)


def _resident_on_cuda(executor: Any) -> bool:
    """Whether the built executor's model holds its weights on a CUDA device."""

    device = getattr(getattr(executor, "model", None), "device", None)
    return device is not None and str(device).startswith("cuda")


class _ParkingPhase(Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    CUMEM_BROKEN = "cumem_broken"


class GenerationWorkerParking:
    """Own one worker's parking session and the policy around it.

    The Ray launch contract carries only the topology-derived requirement to
    yield the GPU. This owner combines it with the resolved rollout residency
    mode before model construction, then keeps the session for idempotent
    sleep, wake, and teardown and quarantines the worker when a failure made
    its residency unknowable.
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
        rollout = launch_contract.model_build.get("rollout")
        pipeline_offload_mode = "none"
        if isinstance(rollout, Mapping):
            pipeline_offload_mode = str(
                rollout.get("pipeline_offload_mode", "none"),
            )
        # The wire mapping is read before anyone rebuilds the typed
        # RolloutBuildOptions from it, so share that type's validator rather than
        # writing the vocabulary check a second time.
        PipelineOffloadMode(pipeline_offload_mode)
        self.worker_id = worker_id
        self._required = launch_contract.sleep_offload
        # rollout.pipeline_offload_mode != "none": Accelerate already owns the
        # model's residency, so the worker must not claim a CuMem scope around it.
        self._pipeline_offload = pipeline_offload_mode != "none"
        self._session: ParkingSession | None = None
        self._phase = _ParkingPhase.ACTIVE
        # Persist only a lightweight quarantine reason. Exception tracebacks can
        # retain the model and its CUDA tensors past terminal cleanup.
        self._failure_reason: str | None = None

    @property
    def _is_parked(self) -> bool:
        return self._session is not None and self._session.parked

    def build(
        self,
        build_executor: Callable[[], GenerationBatchExecutor],
    ) -> GenerationBatchExecutor:
        """Build once; a parking-required, CUDA-resident model lives in a CuMem pool.

        The mechanism follows from residency, not from the family: whether the
        rank must yield its GPU (the launch contract), whether Accelerate
        already manages the pipeline's residency (``pipeline_offload_mode``),
        and whether the built model is resident on CUDA at all (a model that
        is not, such as an isolated-runtime wrapper, has nothing to unmap and
        parks by moving). The pool scope must wrap construction, so the third
        fact is read after the build and an unneeded pool is closed at once.
        """

        self.require_active("policy build", executor=None)
        if self._session is not None:
            raise RuntimeError(
                f"generation worker {self.worker_id!r} already owns a loaded "
                "memory-parking backend",
            )
        self._phase = _ParkingPhase.ACTIVE
        self._failure_reason = None
        session = ParkingSession(f"generation worker {self.worker_id!r}", required=self._required)
        try:
            executor = session.build(
                build_executor,
                cumem=self._required and not self._pipeline_offload,
                tag=f"vrl:generation:{self.worker_id}:weights",
                resident=_resident_on_cuda,
            )
        except CumemBroken as error:
            self._session = session
            self._quarantine(str(error), cumem_broken=True)
            raise
        self._session = session
        return executor

    def validate_loaded(self, executor: GenerationBatchExecutor) -> None:
        """Fail before serving when no complete backend owns the loaded model."""

        session = self._loaded_session()
        if not session.required:
            return
        model = getattr(executor, "model", None)
        uses_pipeline_offload = bool(
            getattr(model, "uses_pipeline_cpu_offload", False),
        )
        if session.mechanism == "cumem":
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
                f"{type(executor).__name__}: model parking requires executor.model.to(...)"
            )

    def sleep(
        self,
        executor: GenerationBatchExecutor | None,
        *,
        restore_device: Any,
    ) -> WorkerMemoryParkingSnapshot:
        """Park the loaded executor and return physical handoff evidence."""

        self.require_healthy("sleep", executor=executor)
        if executor is None:
            if self._required:
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} cannot park before policy load",
                )
            used_bytes = parking.gpu_process_used_bytes()
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
        loaded_bytes = session.gpu_used_bytes()
        model = executor.model

        try:
            if session.mechanism == "cumem":
                session.park()
                snapshot_backend = "cumem"
            elif bool(getattr(model, "uses_pipeline_cpu_offload", False)):
                # Accelerate hooks remain installed; resetting them drops the
                # resident layers, and the next forward onloads again.
                session.park(
                    move=lambda: self._reset_pipeline_cpu_offload(model, operation="sleep"),
                )
                snapshot_backend = "cpu_offload"
            else:
                ledger = session.backend
                assert isinstance(ledger, ModelParking)
                session.park(move=lambda: ledger.park(model, restore_device=restore_device))
                snapshot_backend = (
                    "cpu_offload" if str(ledger.restore_device).startswith("cuda") else "cpu_only"
                )
        except ParkingBroken as error:
            self._quarantine(str(error), cumem_broken=isinstance(error, CumemBroken))
            raise
        residual_bytes_limit = (
            CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT if snapshot_backend != "cpu_only" else 0
        )

        try:
            session.release_gpu()
            residual_bytes = session.gpu_used_bytes()
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
            _log_parking_diagnostics(model, worker_id=self.worker_id)
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
        return snapshot

    def wake(self, executor: GenerationBatchExecutor) -> None:
        """Restore physical residency required before generation resumes."""

        self.require_healthy("wake", executor=executor)
        if not self._is_parked:
            return
        try:
            self._loaded_session().restore()
        except CumemBroken as error:
            self._quarantine(str(error), cumem_broken=True)
            raise

    @contextmanager
    def release_scope(self) -> Iterator[None]:
        """Release state after the caller drops its executor reference."""

        if self._phase is _ParkingPhase.CUMEM_BROKEN:
            raise RuntimeError(
                f"generation worker {self.worker_id!r} has an indeterminate CuMem "
                "mapping and must terminate instead of closing the pool in process",
            )
        session = self._session
        if session is None:
            yield
            release_cuda_memory(ipc_collect=True)
        else:
            try:
                with session.release_scope():
                    yield
            except CumemBroken as error:
                self._quarantine(str(error), cumem_broken=True)
                raise
        self._session = None
        self._phase = _ParkingPhase.ACTIVE
        self._failure_reason = None

    def require_healthy(
        self,
        operation: str,
        *,
        executor: GenerationBatchExecutor | None = None,
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
        executor: GenerationBatchExecutor | None,
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

    def _loaded_session(self) -> ParkingSession:
        if self._session is not None:
            return self._session
        if self._required:
            # build() is the only place that commits a backend. Manufacturing one
            # here for a parking-required worker would silently reintroduce the
            # CuMem-to-model downgrade that build() refuses.
            raise RuntimeError(
                f"generation worker {self.worker_id!r} queried its parking backend "
                "before policy build committed one",
            )
        self._session = ParkingSession(
            f"generation worker {self.worker_id!r}",
            required=False,
            ledger=ModelParking(),
        )
        return self._session

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


__all__ = ["GenerationWorkerParking"]
