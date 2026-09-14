"""Request-level batch execution shared by every generation binding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import (
    GenerationSampleBatch,
    execute_generation_batches,
)
from vrl.generation.execution.types import BatchCompletionCallback, BatchProduceFence
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
    - ``forward_batches_pipelined`` is the single-worker overlap of batch N+1's
      produce with batch N's GPU->CPU teardown; bindings that expose it wrap
      it in their own ``forward_plan_pipelined``.
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

    def forward_batches_pipelined(
        self,
        request: GenerationRequest,
        batches: Sequence[GenerationSampleBatch],
        *,
        completion_callback: BatchCompletionCallback | None = None,
    ) -> list[BatchPayload]:
        """In-process software pipeline over a request's batches: while batch N+1's
        PRODUCE (encode->prepare->denoise->decode, GPU compute on the default stream)
        runs, batch N's TEARDOWN (the GPU->CPU result copy + host packing, on a copy
        stream) drains — hiding the copy+CPU boundary behind the next batch's denoise.

        Before producing batch N+1, enqueue batch N's copy on the copy stream.
        That stream waits for batch N's produce event before reading its tensors;
        compute does not wait for the copy to finish. Copies preserve tensor values,
        and results are appended in batch order regardless of copy completion.

        Compute uses the executor's canonical ``forward_batch`` implementation;
        teardown is a stream-scoped GPU-to-CPU copy. Results remain in batch order.
        """

        import torch

        cuda = torch.cuda.is_available()
        copy_stream = torch.cuda.Stream() if cuda else None

        results: list[BatchPayload] = []
        pending_events: list[torch.cuda.Event] = []

        failed = False
        try:
            for idx, batch in enumerate(batches):
                result = self.forward_batch(request, batch)
                produce_done = None
                if cuda:
                    produce_done = torch.cuda.Event()
                    produce_done.record()  # This batch's compute has been enqueued.
                if completion_callback is not None:
                    # Registration happens only after the CUDA event is recorded.
                    # The callback retains this fence; it does not claim completion
                    # until a later non-blocking query observes the event.
                    completion_callback(
                        BatchProduceFence(
                            completed_batches=idx + 1,
                            event=produce_done,
                        ),
                    )

                # Queue this batch's copy before starting the next batch's compute.
                # Only the copy stream waits for production; the host keeps advancing.
                if result is not None and copy_stream is not None:
                    copy_stream.wait_event(produce_done)
                    results.append(_enqueue_cpu_copies(result, copy_stream))
                    copy_done = torch.cuda.Event()
                    copy_done.record(copy_stream)
                    pending_events.append(copy_done)
                else:
                    results.append(result)
        except BaseException:
            failed = True
            raise
        finally:
            # A later produce can OOM while the previous batch's side-stream D2H is
            # still reading its source tensors. Join every submitted copy before the
            # worker clears exception frames and releases those tensors for retry.
            try:
                for ev in pending_events:
                    ev.synchronize()
            finally:
                # If teardown itself failed after submitting a copy but before its
                # Event was appended, the stream is the only complete barrier.
                if failed and copy_stream is not None:
                    copy_stream.synchronize()
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


def _enqueue_cpu_copies(value: Any, stream: Any) -> Any:
    """Enqueue CUDA tensor copies into pinned CPU buffers on ``stream``.

    Return the rebuilt tensor tree without waiting for the copies. The caller
    must order this stream after production and wait for copy completion before
    reading the CPU buffers. Non-CUDA tensors are returned unchanged.
    """

    import torch

    from vrl.trajectory.device import map_tensor_tree

    def _leaf(t: Any) -> Any:
        if not t.is_cuda:
            return t
        source = t.detach()
        host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
        with torch.cuda.stream(stream):
            host.copy_(source, non_blocking=True)
        # The pending Event protects consumers of ``host``, but it does not keep
        # ``source`` alive. Tell the caching allocator that the source storage is
        # still read by the copy stream so a short next batch cannot recycle it
        # before D2H completes.
        source.record_stream(stream)
        return host

    return map_tensor_tree(
        value,
        _leaf,
        is_leaf=lambda candidate: isinstance(candidate, torch.Tensor),
    )


__all__ = ["BatchExecutorBase"]
