"""Request-level batch execution shared by every generation binding."""

from __future__ import annotations

from collections.abc import Sequence

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import execute_generation_batches
from vrl.generation.protocols import BatchPayload, GenerationBatchGatherer
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)


class BatchExecutorBase:
    """Drive a planned request through the family batch step and gather it.

    The two binding bases (full-sequence and chunk-autoregressive denoise)
    differ in how ONE batch is produced, never
    in how a request's batches are driven or assembled, so that half lives here:

    - ``forward_plan`` is the in-process request entry (local tools, family
      tests, single-process e2e): the same ``forward_batch`` batch step
      and the same gather as the Ray dispatch, with a local OOM-split retry.
      Production drives batches through the Ray dispatcher instead; planning is
      shared via ``EnginePlan.from_request``'s single width fallback.
    - ``merge_generation_batches`` delegates to the gatherer injected by the composition
      root that already owns the family registry entry. The neutral execution
      layer never looks family identity up again.

    Families own ``forward_batch``; overriding ``merge_generation_batches`` is only
    for payloads that never reach the registry (test doubles).
    """

    family: str

    def __init__(self, *, gatherer: GenerationBatchGatherer | None = None) -> None:
        self._gatherer = gatherer

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        plan: EnginePlan,
    ) -> GenerationOutput:
        batches = execute_generation_batches(
            plan.sample_batches,
            lambda batch: self.forward_batch(request, batch),
        )
        return self.merge_generation_batches(request, list(sample_rows), batches)

    def merge_generation_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        if self._gatherer is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an injected batch gatherer "
                "for request-level execution",
            )
        return self._gatherer.merge_generation_batches(request, sample_rows, batches)


__all__ = ["BatchExecutorBase"]
