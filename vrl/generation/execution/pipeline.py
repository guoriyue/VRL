"""Single-worker overlap between chunk compute and result teardown."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _move_tree_to_cpu_async(value: Any, stream: Any) -> Any:
    """Move every CUDA tensor in a (possibly nested) structure to pinned CPU on
    ``stream`` with a non-blocking copy — NO global sync. Mirrors worker._to_cpu
    but stream-scoped + event-based, so the D2H drains concurrently with the next
    chunk's denoise instead of blocking it (the copy engine ≠ the tensor cores).
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
        # still read by the copy stream so a short next chunk cannot recycle it
        # before D2H completes.
        source.record_stream(stream)
        return host

    return map_tensor_tree(
        value,
        _leaf,
        is_leaf=lambda candidate: isinstance(candidate, torch.Tensor),
    )


def forward_chunks_pipelined(
    executor: Any,
    request: Any,
    chunks: Any,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> list:
    """In-process software pipeline over a request's chunks: while chunk N+1's
    PRODUCE (encode->prepare->denoise->decode, GPU compute on the default stream)
    runs, chunk N's TEARDOWN (the GPU->CPU result copy + host packing, on a copy
    stream) drains — hiding the copy+CPU boundary behind the next chunk's denoise.

    BIT-EXACT by construction: the side-stream copy never changes values, and the
    only ordering it introduces (teardown(N) before produce(N+1) is launched) is
    guarded by a default-stream event so the copy never reads tensors a denoise
    kernel is still writing (torn-read safety). Chunks are independent and gather
    re-orders by ordered_chunks, so completion order is irrelevant to the output.

    Compute uses the executor's canonical ``forward_chunk_plan`` implementation;
    teardown is a stream-scoped GPU-to-CPU copy. Results remain in chunk order.
    """

    import torch

    cuda = torch.cuda.is_available()
    copy_stream = torch.cuda.Stream() if cuda else None

    def _teardown(result: Any) -> Any:
        if copy_stream is None:
            return result
        return _move_tree_to_cpu_async(result, copy_stream)

    chunk_list = list(chunks)
    results: list = [None] * len(chunk_list)
    prev_idx = -1
    prev_result = None
    prev_done = None  # default-stream event marking prev produce complete
    pending_events: list = []

    failed = False
    try:
        for idx, chunk in enumerate(chunk_list):
            # Start the PREVIOUS chunk's teardown on the copy stream BEFORE producing
            # this chunk, so the D2H overlaps this chunk's denoise. Wait on prev_done so
            # the copy never reads tensors a denoise kernel is still writing.
            if prev_result is not None and copy_stream is not None:
                copy_stream.wait_event(prev_done)
                results[prev_idx] = _teardown(prev_result)
                ev = torch.cuda.Event()
                ev.record(copy_stream)
                pending_events.append(ev)
            elif prev_result is not None:
                results[prev_idx] = _teardown(prev_result)

            prev_result = executor.forward_chunk_plan(request, chunk)
            prev_idx = idx
            if progress_callback is not None:
                progress_callback(idx + 1)
            if cuda:
                prev_done = torch.cuda.Event()
                prev_done.record()  # default stream: this chunk's produce is enqueued

        # Flush the final chunk's teardown.
        if prev_result is not None:
            if copy_stream is not None:
                copy_stream.wait_event(prev_done)
                results[prev_idx] = _teardown(prev_result)
                ev = torch.cuda.Event()
                ev.record(copy_stream)
                pending_events.append(ev)
            else:
                results[prev_idx] = _teardown(prev_result)
    except BaseException:
        failed = True
        raise
    finally:
        # A later produce can OOM while the previous chunk's side-stream D2H is
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


__all__ = [
    "forward_chunks_pipelined",
]
