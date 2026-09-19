"""Request-level batch execution shared by every generation binding."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import (
    GenerationSampleBatch,
    execute_generation_batches,
)
from vrl.generation.execution.types import BatchCompletion, BatchCompletionCallback
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
    - ``execute_request_batches`` runs ALL of a request's batches on one
      worker in one call, copying each result to pinned CPU memory (and
      staging it, when asked) before the next batch is produced. Ray uses
      this entry directly; ``forward_plan_pipelined`` wraps it only for local
      execution and merge equivalence tests.
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

    def execute_request_batches(
        self,
        request: GenerationRequest,
        batches: Sequence[GenerationSampleBatch],
        *,
        completion_callback: BatchCompletionCallback | None = None,
        stage_batch_result: Callable[[BatchPayload], Any] | None = None,
    ) -> list[Any]:
        """Produce a request's batches in order on this worker, one RPC for all.

        What this saves over per-batch dispatch is the per-batch Ray round trip,
        result pickling, and worker prologue, during which the GPU sat idle. Each
        batch's result is copied to pinned CPU memory (one synchronize) before
        the next batch is produced, so at most one batch's payload occupies the
        GPU at a time and the copied results are readable when the loop returns.

        The copy itself is not overlapped with the next batch's compute: measured
        on Cosmos 240p (SPRINT_diffusion_rollout_stage_pipeline, 2026-06-27) the
        device-to-host copy is a few percent of denoise time and a side-stream
        overlap recovered nothing (6735 ms serial vs 6729 ms overlapped), so the
        loop keeps the plain synchronous copy.

        ``stage_batch_result`` runs on each copied batch result and its return value is
        what the loop keeps: the Ray rank passes ``ray.put`` so the loop returns
        object-store references instead of payloads. Staging is NOT overlapped
        with the next batch's compute either; the pinned copy synchronizes the
        device first, so the order within one request is strictly compute, copy,
        stage, next batch. What staging buys is that the rank's return value is
        small and the merge runs elsewhere, so THIS request's finalize overlaps
        the NEXT request's generation. ``completion_callback`` receives one completion
        per batch, after that batch's result is staged. Results stay in batch
        order.
        """

        from vrl.trajectory.device import copy_tensor_tree_to_pinned_cpu

        results: list[Any] = []
        for idx, batch in enumerate(batches):
            result = self.forward_batch(request, batch)
            if result is not None:
                result = copy_tensor_tree_to_pinned_cpu(result)
                if stage_batch_result is not None:
                    result = stage_batch_result(result)
            results.append(result)
            if completion_callback is not None:
                completion_callback(BatchCompletion(completed_batches=idx + 1))
        return results

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
