"""Request-level sample batch planning."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """Public execution-plan envelope shared by direct and Ray runtimes."""

    sample_batches: tuple[GenerationSampleBatch, ...]

    @classmethod
    def from_request(
        cls,
        request: GenerationRequest,
        *,
        max_samples_per_batch: int | None = None,
    ) -> EnginePlan:
        """Plan the batches consumed by direct and distributed executors.

        THE single batch-width resolution: explicit ``max_samples_per_batch``
        argument, then the request's ``samples_per_generation_batch``,
        then ``samples_per_prompt`` (the whole group in one batch). Every
        planner — Ray placement and in-process alike — goes through this one
        fallback.
        """

        from vrl.utils.profiling import profile_range

        if max_samples_per_batch is not None:
            batch_size = max(1, int(max_samples_per_batch))
        else:
            raw = request.samples_per_generation_batch
            if raw is None:
                raw = request.samples_per_prompt
            if raw == "auto":
                # Resolved to an int by the Ray runtime's startup probe before
                # a request reaches planning; seeing it here means the request
                # bypassed that runtime (e.g. a local/direct executor).
                raise ValueError(
                    "rollout.samples_per_generation_batch: auto requires the Ray "
                    "generation runtime (startup batch-size probe); set an "
                    "explicit int here",
                )
            batch_size = max(1, int(raw))
        with profile_range("engine.plan"):
            return cls(
                sample_batches=GenerationSampleBatch.plan(
                    len(request.inputs),
                    samples_per_prompt=request.samples_per_prompt,
                    max_samples_per_batch=batch_size,
                ),
            )


__all__ = ["EnginePlan"]
