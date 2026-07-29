"""CUDA memory helpers shared across engine, rollout, and trainer code."""

from __future__ import annotations

import gc
import itertools
import os
from typing import Any

# CUDA loads device code and library state on first real kernel execution. Those
# process-lifetime pages are not user tensors, the caching allocator, or a tagged
# CuMem model pool, so neither CPU offload nor pool.sleep() can release them. RTX
# 5090 probes measured 120--210 MiB after deleting all user tensors; production
# SANA generation retained 154 MiB and a correctly pooled CLIP-L score retained
# 42 MiB (126 MiB in a fresh process). Keep one bounded backend protocol limit;
# callers still reject a single byte beyond it and CPU-only paths use zero.
#
# The default is calibrated for fp16 rollout. fp32 (or any doubled-precision)
# generation leaves a larger, fragmentation-dependent residual after cumem
# sleep — measured 0.2--0.9 GiB above baseline on single-card colocated SANA.
# On a high-headroom card that residual is harmless (rollout parks while the
# trainer's few-GiB step runs, far under the 32 GiB ceiling), so the tolerance
# is overridable via VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB for such runs. The
# default stays strict so a real leak in the tight fp16 case still fails loud.
CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT = (
    int(os.environ.get("VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB", "256")) * 1024 * 1024
)


def _cumem_allocator() -> Any | None:
    """Return the process-wide vLLM CuMemAllocator, or None when unavailable.

    None on a CPU box or when vLLM is not importable. The allocator is a
    per-process singleton; callers hold a :class:`CumemPool` tagged handle
    rather than the raw allocator. Tags control backup/discard during one
    process-wide sleep; they do not isolate sleep operations. Single mock point
    for tests.
    """

    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except Exception:
        return None
    try:
        return CuMemAllocator.get_instance()
    except Exception:
        return None


class CumemPool:
    """Tagged handle over the process-wide vLLM CuMemAllocator.

    A tag selects which pages receive a CPU backup; it is not an independently
    sleepable allocator slice. ``CuMemAllocator.sleep`` walks and unmaps every
    registered pointer, discarding pages outside ``offload_tags``. Therefore a
    process may have only one independently parked owner unless a higher-level
    coordinator performs one process-wide sleep with all backup tags together.
    Build the owner's model inside :meth:`building` so its CUDA allocations get
    the backup tag; :meth:`sleep` then copies those pages to pinned host RAM and
    unmaps the process-wide allocator while virtual addresses stay valid.

    Note: pooled physical pages come from CUDA virtual memory, NOT torch's
    caching allocator — a co-resident phase must ``empty_cache`` before this
    pool wakes, or its cached-but-free blocks starve the remap.
    """

    _tags = itertools.count()

    def __init__(self, allocator: Any, tag: str) -> None:
        self._allocator = allocator
        self.tag = tag
        self.asleep = False
        self._building_claimed = False
        self._closed = False

    @classmethod
    def try_create(cls, tag: str | None = None) -> CumemPool | None:
        """Availability probe: pool handle, or None on a CPU box / without vLLM.

        Callers that need a pool must use :meth:`require`. Branching on this
        None to build unpooled is how the deleted CPU-parking fallback came
        back once already: it turns a misconfigured box into a silent 6x
        slowdown that only appears to release GPU memory.
        """

        allocator = _cumem_allocator()
        if allocator is None:
            return None
        return cls(allocator, tag if tag else f"cumem-{next(cls._tags)}")

    @classmethod
    def require(cls, tag: str | None = None) -> CumemPool:
        """Pool handle; raise when cumem is unavailable (no silent fallback)."""

        pool = cls.try_create(tag)
        if pool is None:
            raise RuntimeError(
                "vLLM's CuMemAllocator is required here but unavailable — "
                "install vLLM and run on a CUDA device.",
            )
        return pool

    def building(self) -> Any:
        """Return the tag's one-shot model-construction allocation scope.

        vLLM creates a new ``torch.cuda.MemPool`` on every
        ``use_memory_pool`` call. Re-entering the same tag while tensors from its
        first pool are alive can abort the process inside PyTorch rather than
        raise Python. Keep the dependency's supported shape: one scope for model
        construction, then normal execution plus a physical residual check.
        """

        if self._closed:
            raise RuntimeError(f"CuMemPool tag {self.tag!r} is closed")
        if self._building_claimed:
            raise RuntimeError(
                f"CuMemPool tag {self.tag!r} model-building scope is one-shot",
            )
        self._building_claimed = True
        return self._allocator.use_memory_pool(tag=self.tag)

    def sleep(self) -> None:
        """Sleep the process-wide allocator, backing up only this handle's tag."""

        if self._closed:
            raise RuntimeError(f"CuMemPool tag {self.tag!r} is closed")
        self._allocator.sleep(offload_tags=(self.tag,))
        self.asleep = True

    def wake(self) -> None:
        """Remap this handle's backed-up tag (no-op when not asleep)."""

        if self._closed:
            return
        if not self.asleep:
            return
        self._allocator.wake_up(tags=[self.tag])
        self.asleep = False

    def close(self) -> None:
        """Drop vLLM's retained MemPool after all tagged tensors are gone.

        This is a terminal-only adapter for the installed vLLM API. Its public
        ``use_memory_pool`` context retains the created ``torch.cuda.MemPool`` in
        ``allocator_and_pools`` but exposes no close operation; leaving that
        registry entry alive keeps freed model pages mapped indefinitely. Guard
        the one required internal seam explicitly so a vLLM layout change fails
        closed instead of silently leaking GPU ownership.
        """

        if self._closed:
            return
        if self.asleep:
            raise RuntimeError(
                f"CuMemPool tag {self.tag!r} must be awake before terminal close",
            )
        retained_pools = getattr(self._allocator, "allocator_and_pools", None)
        if not isinstance(retained_pools, dict):
            raise RuntimeError(
                "installed vLLM CuMemAllocator does not expose the retained-pool "
                "registry required for terminal release",
            )
        retained = retained_pools.pop(self.tag, None)
        if retained is None:
            raise RuntimeError(
                f"vLLM retained no CuMem pool for terminal tag {self.tag!r}",
            )
        self._closed = True
        del retained
        gc.collect()


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Return whether an exception looks like a CUDA OOM failure."""

    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def empty_cuda_cache() -> None:
    """Empty PyTorch CUDA cache when CUDA is available."""

    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def release_cuda_memory(*, ipc_collect: bool = False) -> None:
    """Release best-effort CUDA memory after large runtime objects are dropped."""

    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch
    except Exception:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        if ipc_collect:
            torch.cuda.ipc_collect()
    except Exception:
        return


def reset_cuda_peak() -> None:
    """Reset the process CUDA peak counters at a phase boundary.

    Pairs with :func:`cuda_peak_allocated_bytes`: resetting at each phase
    boundary is what makes the readback phase-scoped instead of a
    process-lifetime high-water mark.
    """

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def cuda_peak_allocated_bytes() -> int | None:
    """Peak CUDA bytes allocated since the last reset (None without CUDA)."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def cuda_peak_allocated_mb() -> float | None:
    """:func:`cuda_peak_allocated_bytes` in MiB, for debug metric payloads."""

    peak_bytes = cuda_peak_allocated_bytes()
    return None if peak_bytes is None else peak_bytes / (1024 * 1024)


def gpu_used_bytes(device: str | None = None) -> int:
    """Driver-level physical bytes in use on a CUDA device (0 without CUDA).

    ``device=None`` measures the process's current CUDA device. A non-CUDA
    ``device`` string (or a CPU-only box) reads as 0 — memory-parking proofs
    on CPU components are trivially satisfied.
    """

    if device is not None and not str(device).startswith("cuda"):
        return 0
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    target = None if device is None else torch.device(device)
    torch.cuda.synchronize(target)
    free_bytes, total_bytes = torch.cuda.mem_get_info(target)
    return int(total_bytes - free_bytes)


def release_cuda_memory_for_parking(device: str | None = None) -> None:
    """Strict CUDA cleanup before publishing a memory-parking proof.

    Unlike :func:`release_cuda_memory` this path must not swallow failures:
    the caller is about to certify physical GPU release to a phase handoff,
    so any error here invalidates the handoff and propagates.
    """

    gc.collect()
    if device is not None and not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        return
    target = torch.device(device)
    with torch.cuda.device(target):
        torch.cuda.synchronize(target)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize(target)


__all__ = [
    "CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT",
    "CumemPool",
    "cuda_peak_allocated_bytes",
    "cuda_peak_allocated_mb",
    "empty_cuda_cache",
    "gpu_used_bytes",
    "is_cuda_out_of_memory",
    "release_cuda_memory",
    "release_cuda_memory_for_parking",
    "reset_cuda_peak",
]
