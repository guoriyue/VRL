"""CPU actor that merges a request's staged batch payloads into one output.

The GPU rank stages each batch into the object store as it is produced and
returns only references. Merging (the gatherer's concatenation and trajectory
build) and boxing the reward media then run here, in a process without a model,
so the rank is free for the next request while this one is assembled. One
finalizer is launched per engine, pinned to the engine's primary bundle; the
executor hands each request to whichever finalizer is free, so with several
engines the merge reads part of its input from other nodes' object stores.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.generation.protocols import GenerationBatchGatherer
from vrl.generation.ray.reward_media import reference_reward_media
from vrl.generation.ray.tensor_wire import register_tensor_wire_serializer
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.ray.dependencies import current_node_ip


class RayGenerationFinalizer:
    """Ray actor adapter around a registry-owned batch gatherer."""

    def __init__(self, finalizer_id: str, gatherer: GenerationBatchGatherer) -> None:
        if not finalizer_id:
            raise ValueError("finalizer_id must be non-empty")
        if not isinstance(gatherer, GenerationBatchGatherer):
            raise TypeError(
                "RayGenerationFinalizer requires a GenerationBatchGatherer, "
                f"got {type(gatherer).__name__}",
            )
        # The merged trajectory leaves this process the same way batch results
        # leave the rank: as byte views of its host buffers.
        register_tensor_wire_serializer()
        self.finalizer_id = finalizer_id
        self.gatherer = gatherer

    def health(self) -> str:
        return self.finalizer_id

    def worker_metadata(self) -> dict[str, Any]:
        """Return Ray placement metadata used during actor-group startup."""

        return {
            "worker_id": self.finalizer_id,
            "node_ip": current_node_ip(),
            "gpu_ids": [],
        }

    def merge_request(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batch_refs: Sequence[Any],
    ) -> GenerationOutput:
        """Fetch the staged batch payloads and assemble the request's output.

        ``batch_refs`` arrives as a list so Ray hands over the references rather
        than resolving them at call time; this process fetches them itself.
        """

        import ray

        if not batch_refs:
            raise ValueError(
                f"finalizer {self.finalizer_id!r} received no batch references "
                f"for request {request.request_id!r}",
            )
        payloads = ray.get(list(batch_refs))
        output = self.gatherer.merge_generation_batches(request, list(sample_rows), payloads)
        if not isinstance(output, GenerationOutput):
            raise TypeError(
                f"{type(self.gatherer).__name__}.merge_generation_batches returned "
                f"{type(output).__name__}, expected GenerationOutput",
            )
        reference_reward_media(output, request, primary=True)
        return output


__all__ = ["RayGenerationFinalizer"]
