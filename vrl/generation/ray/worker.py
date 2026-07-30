"""Thin Ray actor wrapper for generation worker execution."""

from __future__ import annotations

from typing import Any

import ray

from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    PipelinedRequestOutOfMemory,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.types import GenerationOutput
from vrl.ray.dependencies import current_gpu_ids, current_node_ip

# Ray binds a method to a concurrency group by name across two separate APIs --
# @ray.method here and ray.remote(concurrency_groups=...) at actor creation --
# so the name is shared; the group's thread count belongs to the creation site.
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
            metadata_provider=self._ray_metadata,
        )

    @ray.method(concurrency_group=HEALTH_CONCURRENCY_GROUP)
    def health(self) -> str:
        """Answer a liveness probe without touching model or GPU state.

        Runs in its own concurrency group so it never queues behind
        ``execute_chunk`` — a queued probe would measure queue depth, not
        liveness. The group is deliberately not a raw ``max_concurrency``
        bump: that would also let two chunks execute concurrently on one GPU
        worker whenever ``max_inflight_chunks_per_worker`` exceeds one.
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

    def worker_metadata(self, *, runtime_debug: bool = False) -> dict[str, Any]:
        return self.core.worker_metadata(runtime_debug=runtime_debug)

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        return self.core.execute_chunk(envelope)

    def probe_chunk_size(
        self,
        request: Any,
        *,
        max_samples: int,
    ) -> dict[str, Any]:
        """Startup chunk-size probe; see GenerationWorkerCore.probe_chunk_size."""
        return self.core.probe_chunk_size(
            request,
            max_samples=max_samples,
        )

    def execute_request_pipelined(
        self,
        request: Any,
        engine_plan: Any,
        sample_rows: Any,
    ) -> GenerationOutput | PipelinedRequestOutOfMemory:
        """Per-request software-pipelined execution (single-worker stage-overlap
        path); returns a gathered output or typed OOM retry. See
        GenerationWorkerCore.execute_request_pipelined."""
        return self.core.execute_request_pipelined(request, engine_plan, sample_rows)

    def _ray_metadata(self) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "unknown"
            gpu_ids = []
        return {"worker_id": self.core.worker_id, "node_ip": node_ip, "gpu_ids": gpu_ids}


__all__ = ["RayGenerationWorker"]
