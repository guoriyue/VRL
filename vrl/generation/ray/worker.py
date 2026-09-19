"""Thin Ray actor wrapper for generation worker execution."""

from __future__ import annotations

import threading
from typing import Any

import ray

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import (
    BatchSizeProbeResult,
    GenerationBatchEnvelope,
    GenerationBatchResult,
    RequestBatchOutOfMemory,
    StagedBatchRefs,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.pipeline_protocol import RequestBatchProgress
from vrl.generation.ray.reward_media import reference_reward_media
from vrl.generation.ray.tensor_wire import register_tensor_wire_serializer
from vrl.generation.types import GenerationRequest
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
        # Batch results and trajectories leave this process as byte views of
        # their pinned host buffers instead of pickled storages, so a request's
        # return does not stall the actor for a copy of every trajectory tensor.
        register_tensor_wire_serializer()
        self.core = GenerationWorkerCore(
            worker_id,
            launch_inputs.launch_contract,
            launch_inputs.gatherer,
            rank_group=launch_inputs.rank_group,
        )
        self._pipelined_progress_lock = threading.Lock()
        self._pipelined_progress: RequestBatchProgress | None = None

    @ray.method(concurrency_group=HEALTH_CONCURRENCY_GROUP)
    def health(self) -> str:
        """Answer a liveness probe without touching model or GPU state.

        Runs in its own concurrency group so it never queues behind
        ``execute_batch`` — a queued probe would measure queue depth, not
        liveness. The group is deliberately not a raw ``max_concurrency``
        bump: that would also let two batches execute concurrently on one GPU
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

    def begin_weight_transfer(self, manifest: Any, transfer_id: str, policy_version: int) -> int:
        return self.core.begin_weight_transfer(manifest, transfer_id, policy_version)

    def receive_weight_chunk(self, chunk: Any, transfer_id: str) -> int:
        return self.core.receive_weight_chunk(chunk, transfer_id)

    def receive_weight_bucket(self, chunks: Any, transfer_id: str) -> int:
        if not chunks:
            raise ValueError("weight bucket must not be empty")
        for chunk in chunks:
            version = self.receive_weight_chunk(chunk, transfer_id)
        return version

    def commit_weight_transfer(self, transfer_id: str, *, verify_content: bool = False) -> int:
        return self.core.commit_weight_transfer(transfer_id, verify_content=verify_content)

    def abort_weight_transfer(self, transfer_id: str) -> None:
        self.core.abort_weight_transfer(transfer_id)

    def verify_active_weights(self, trainable_state: Any, policy_version: int) -> int:
        """Read back the state already active on this rank for acceptance."""

        return self.core.verify_active_weights(trainable_state, policy_version)

    def update_weights(
        self, trainable_state: Any, policy_version: int, *, verify_content: bool = False
    ) -> int:
        if verify_content:
            return self.core.update_weights(trainable_state, policy_version, verify_content=True)
        return self.core.update_weights(trainable_state, policy_version)

    def supports_versioned_trainable_state(self) -> bool:
        return self.core.supports_versioned_trainable_state()

    def worker_metadata(self) -> dict[str, Any]:
        """Return Ray placement metadata used during actor-group startup."""

        node_ip = current_node_ip()
        gpu_ids = current_gpu_ids()
        return {
            "worker_id": self.core.worker_id,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
        }

    def execute_batch(self, envelope: GenerationBatchEnvelope) -> GenerationBatchResult:
        result = self.core.execute_batch(envelope)
        if result.output is not None and not result.error:
            reference_reward_media(result.output, envelope.request, primary=self._is_primary_rank)
        return result

    @property
    def _is_primary_rank(self) -> bool:
        rank_group = self.core.rank_group_spec
        return rank_group is None or rank_group.group_rank == 0

    def probe_batch_size(
        self,
        request: GenerationRequest,
        *,
        max_samples: int,
    ) -> BatchSizeProbeResult:
        """Startup batch-size probe; see GenerationWorkerCore.probe_batch_size."""
        return self.core.probe_batch_size(
            request,
            max_samples=max_samples,
        )

    def execute_request_batches(
        self,
        request: GenerationRequest,
        engine_plan: EnginePlan,
    ) -> StagedBatchRefs | RequestBatchOutOfMemory:
        """Run all of the plan's batches on this rank in one call.

        Each batch payload is staged into the object store right after its host
        copy, so the returned value carries only references and this rank is
        free for the next request the moment its last batch is staged. A
        non-primary rank of a multi-rank engine stages nothing. See
        GenerationWorkerCore.execute_request_batches for version safety and the
        typed OOM retry.
        """

        request_id = str(request.request_id)
        total_batches = len(engine_plan.sample_batches)
        with self._pipelined_progress_lock:
            if self._pipelined_progress is not None:
                raise RuntimeError(
                    "pipelined worker received overlapping requests "
                    f"{self._pipelined_progress.request_id!r} and {request_id!r}",
                )
            self._pipelined_progress = RequestBatchProgress(
                request_id=request_id,
                completed_batches=0,
                total_batches=total_batches,
            )

        def record_completion(completed_batches: int) -> None:
            with self._pipelined_progress_lock:
                current = self._pipelined_progress
                if current is None or current.request_id != request_id:
                    raise RuntimeError(
                        f"pipelined progress lost active request {request_id!r}",
                    )
                expected = current.completed_batches + 1
                if completed_batches != expected:
                    raise RuntimeError(
                        "batch completion notifications must register one batch at a time "
                        f"(request_id={request_id!r}, previous="
                        f"{expected - 1}, actual={completed_batches})",
                    )
                if completed_batches > total_batches:
                    raise RuntimeError(
                        "batch completion exceeds request batch count "
                        f"(request_id={request_id!r}, total={total_batches}, "
                        f"actual={completed_batches})",
                    )
                self._pipelined_progress = RequestBatchProgress(
                    request_id=request_id,
                    completed_batches=completed_batches,
                    total_batches=total_batches,
                )

        primary = self._is_primary_rank
        try:
            staged = self.core.execute_request_batches(
                request,
                engine_plan,
                completion_callback=record_completion,
                stage_batch_result=ray.put if primary else _discard_payload,
            )
        finally:
            with self._pipelined_progress_lock:
                self._pipelined_progress = None
        if isinstance(staged, RequestBatchOutOfMemory):
            return staged
        if len(staged) != total_batches:
            raise RuntimeError(
                f"pipelined worker {self.core.worker_id} staged {len(staged)} batches "
                f"for a {total_batches}-batch plan",
            )
        if not primary:
            return StagedBatchRefs(
                request_id=request_id,
                worker_id=self.core.worker_id,
                batch_keys=(),
                batch_refs=(),
                policy_version=request.policy_version,
            )
        return StagedBatchRefs(
            request_id=request_id,
            worker_id=self.core.worker_id,
            batch_keys=tuple(batch.batch_key for batch in engine_plan.sample_batches),
            batch_refs=tuple(staged),
            policy_version=request.policy_version,
        )

    @ray.method(concurrency_group=HEALTH_CONCURRENCY_GROUP)
    def pipelined_progress(
        self,
        request_id: str,
    ) -> RequestBatchProgress | None:
        """Report strict batch progress without joining the busy default group."""

        with self._pipelined_progress_lock:
            progress = self._pipelined_progress
            if progress is None or progress.request_id != request_id:
                return None
            return progress


def _discard_payload(payload: Any) -> None:
    """Non-primary ranks keep nothing: the engine result is the primary's."""

    del payload
    return None


__all__ = ["RayGenerationWorker"]
