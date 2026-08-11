"""Thin Ray actor wrapper for generation worker execution."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import ray

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    ChunkProduceFence,
    ChunkSizeProbeResult,
    PipelinedRequestOutOfMemory,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.pipeline_protocol import PipelinedRequestProgress
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.ray.dependencies import current_gpu_ids, current_node_ip

# Ray binds methods to a concurrency group by name across two separate APIs --
# @ray.method here and ray.remote(concurrency_groups=...) at actor creation --
# so health/progress adapters share this protocol name; the group's thread
# count belongs to the creation site.
HEALTH_CONCURRENCY_GROUP = "health"


class RayGenerationWorker:
    """Ray actor adapter around ``GenerationWorkerCore``."""

    def __init__(
        self,
        worker_id: str,
        launch_inputs: RayGenerationLaunchInputs,
    ) -> None:
        if not isinstance(launch_inputs, RayGenerationLaunchInputs):
            raise TypeError(
                "launch_inputs must be RayGenerationLaunchInputs, "
                f"got {type(launch_inputs).__name__}",
            )
        self.core = GenerationWorkerCore(
            worker_id,
            launch_inputs.launch_contract,
            launch_inputs.gatherer,
        )
        self._pipelined_progress_lock = threading.Lock()
        self._pipelined_progress: PipelinedRequestProgress | None = None
        self._pipelined_completion_fences: deque[ChunkProduceFence] = deque()

    @ray.method(concurrency_group=HEALTH_CONCURRENCY_GROUP)
    def health(self) -> str:
        """Answer a liveness probe without touching model or GPU state.

        Runs in its own concurrency group so it never queues behind
        ``execute_chunk`` — a queued probe would measure queue depth, not
        liveness. The group is deliberately not a raw ``max_concurrency``
        bump: that would also let two chunks execute concurrently on one GPU
        worker.
        """

        return self.core.worker_id

    def load_policy(self) -> None:
        self.core.load_policy()

    def release_policy(self) -> None:
        self.core.release_policy()

    def sleep(self) -> WorkerMemoryParkingSnapshot:
        return self.core.sleep()

    def wake(self) -> None:
        self.core.wake()

    def update_weights(self, state_ref: Any, policy_version: int) -> int:
        return self.core.update_weights(state_ref, policy_version)

    def supports_versioned_trainable_state(self) -> bool:
        return self.core.supports_versioned_trainable_state()

    def worker_metadata(self) -> dict[str, Any]:
        """Return Ray placement metadata used during actor-group startup."""

        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "unknown"
            gpu_ids = []
        return {
            "worker_id": self.core.worker_id,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
        }

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        return self.core.execute_chunk(envelope)

    def probe_chunk_size(
        self,
        request: GenerationRequest,
        *,
        max_samples: int,
    ) -> ChunkSizeProbeResult:
        """Startup chunk-size probe; see GenerationWorkerCore.probe_chunk_size."""
        return self.core.probe_chunk_size(
            request,
            max_samples=max_samples,
        )

    def execute_request_pipelined(
        self,
        request: GenerationRequest,
        engine_plan: EnginePlan,
        sample_rows: Sequence[GenerationSampleRow],
    ) -> GenerationOutput | PipelinedRequestOutOfMemory:
        """Per-request software-pipelined execution (single-worker stage-overlap
        path); returns a gathered output or typed OOM retry. See
        GenerationWorkerCore.execute_request_pipelined."""
        request_id = str(request.request_id)
        total_chunks = len(engine_plan.chunks)
        with self._pipelined_progress_lock:
            if self._pipelined_progress is not None or self._pipelined_completion_fences:
                active_request_id = (
                    self._pipelined_progress.request_id
                    if self._pipelined_progress is not None
                    else "unknown"
                )
                raise RuntimeError(
                    "pipelined worker received overlapping requests "
                    f"{active_request_id!r} and {request_id!r}",
                )
            self._pipelined_progress = PipelinedRequestProgress(
                request_id=request_id,
                completed_chunks=0,
                total_chunks=total_chunks,
            )

        def record_completion(fence: ChunkProduceFence) -> None:
            with self._pipelined_progress_lock:
                current = self._pipelined_progress
                if current is None or current.request_id != request_id:
                    raise RuntimeError(
                        f"pipelined progress lost active request {request_id!r}",
                    )
                expected = current.completed_chunks + len(self._pipelined_completion_fences) + 1
                if fence.completed_chunks != expected:
                    raise RuntimeError(
                        "pipelined completion fences must register one chunk at a time "
                        f"(request_id={request_id!r}, previous="
                        f"{expected - 1}, actual={fence.completed_chunks})",
                    )
                if fence.completed_chunks > total_chunks:
                    raise RuntimeError(
                        "pipelined completion fence exceeds request chunk count "
                        f"(request_id={request_id!r}, total={total_chunks}, "
                        f"actual={fence.completed_chunks})",
                    )
                self._pipelined_completion_fences.append(fence)

        try:
            return self.core.execute_request_pipelined(
                request,
                engine_plan,
                sample_rows,
                completion_callback=record_completion,
            )
        finally:
            with self._pipelined_progress_lock:
                self._pipelined_completion_fences.clear()
                self._pipelined_progress = None

    @ray.method(concurrency_group=HEALTH_CONCURRENCY_GROUP)
    def pipelined_progress(
        self,
        request_id: str,
    ) -> PipelinedRequestProgress | None:
        """Report strict chunk progress without joining the busy default group."""

        with self._pipelined_progress_lock:
            progress = self._pipelined_progress
            if progress is None or progress.request_id != request_id:
                return None
            while self._pipelined_completion_fences:
                fence = self._pipelined_completion_fences[0]
                if not fence.query():
                    break
                self._pipelined_completion_fences.popleft()
                progress = PipelinedRequestProgress(
                    request_id=request_id,
                    completed_chunks=fence.completed_chunks,
                    total_chunks=progress.total_chunks,
                )
                self._pipelined_progress = progress
            return progress


__all__ = ["RayGenerationWorker"]
