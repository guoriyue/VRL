"""Thin Ray actor wrapper for generation worker execution."""

from __future__ import annotations

from typing import Any

import ray

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import (
    GenerationBatchEnvelope,
    GenerationBatchResult,
    RequestBatchOutOfMemory,
    StagedBatchRefs,
    StaleSlotDiscard,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.reward_media import reference_reward_media
from vrl.generation.ray.tensor_wire import register_tensor_wire_serializer
from vrl.generation.types import GenerationRequest
from vrl.ray.dependencies import current_gpu_ids, current_node_ip


class RayGenerationWorker:
    """Ray actor adapter around ``GenerationWorkerCore``."""

    def __init__(
        self,
        worker_id: str,
        launch_inputs: RayGenerationLaunchInputs,
    ) -> None:
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

    def load_policy(self) -> None:
        self.core.load_policy()

    def release_policy(self) -> None:
        self.core.release_policy()

    def sleep(self) -> WorkerMemoryParkingSnapshot:
        return self.core.sleep()

    def wake(self) -> None:
        self.core.wake()

    def update_weights(self, trainable_state: Any, policy_version: int) -> int:
        return self.core.update_weights(trainable_state, policy_version)

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

    def execute_request_batches(
        self,
        request: GenerationRequest,
        engine_plan: EnginePlan,
    ) -> StagedBatchRefs | RequestBatchOutOfMemory | StaleSlotDiscard:
        """Run all of the plan's batches on this rank in one call.

        Each batch payload is staged into the object store right after its host
        copy, so the returned value carries only references and this rank is
        free for the next request the moment its last batch is staged. A
        non-primary rank of a multi-rank engine stages nothing. A stale policy
        slot comes back as a ``StaleSlotDiscard`` value, not a raised error, so
        the driver releases the actor slot and counts a graceful discard. See
        GenerationWorkerCore.execute_request_batches for version safety and the
        typed OOM retry.
        """

        request_id = str(request.request_id)
        total_batches = len(engine_plan.sample_batches)
        primary = self._is_primary_rank
        try:
            staged = self.core.execute_request_batches(
                request,
                engine_plan,
                stage_batch_result=ray.put if primary else _discard_payload,
            )
        except StaleSlotDiscard as discard:
            return discard
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
            )
        return StagedBatchRefs(
            request_id=request_id,
            worker_id=self.core.worker_id,
            batch_keys=tuple(batch.batch_key for batch in engine_plan.sample_batches),
            batch_refs=tuple(staged),
        )


def _discard_payload(payload: Any) -> None:
    """Non-primary ranks keep nothing: the engine result is the primary's."""

    del payload
    return None


__all__ = ["RayGenerationWorker"]
