"""CUDA memory helpers shared across engine, rollout, and trainer code."""

from __future__ import annotations

import gc
import itertools
from typing import Any


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
        """Pool handle, or None when cumem is unavailable (CPU box / no vLLM)."""

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


def release_cuda_memory(
    *,
    gc_collect: bool = False,
    ipc_collect: bool = False,
) -> None:
    """Release best-effort CUDA memory after large runtime objects are dropped."""

    if gc_collect:
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


__all__ = [
    "CumemPool",
    "empty_cuda_cache",
    "is_cuda_out_of_memory",
    "release_cuda_memory",
]
