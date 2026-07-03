"""CUDA memory helpers shared across engine, rollout, and trainer code."""

from __future__ import annotations

import itertools
from typing import Any


def _cumem_allocator() -> Any | None:
    """Return the process-wide vLLM CuMemAllocator, or None when unavailable.

    None on a CPU box or when vLLM is not importable. The allocator is a
    per-process singleton; callers hold a :class:`CumemPool` tag slice of it
    rather than the raw allocator. Single mock point for tests.
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
    """One tagged slice of the process-wide vLLM CuMemAllocator.

    The allocator is a per-process singleton whose sleep/wake act on TAGS, so
    every owner (the rollout executor, each heavyweight reward) holds its own
    tag: waking one owner's pages never drags another owner's back onto the
    GPU. Build the owner's model inside :meth:`building` so every CUDA
    allocation it makes lands in the tagged pool; :meth:`sleep` then copies
    the physical pages to pinned host RAM and unmaps them while the virtual
    addresses stay valid — the model needs no ``.to()``, and :meth:`wake`
    remaps without cudaMalloc, immune to fragmentation (probe-measured ~6x
    faster than a naive ``.to()`` round trip at 14GB scale).

    Note: pooled physical pages come from CUDA virtual memory, NOT torch's
    caching allocator — a co-resident phase must ``empty_cache`` before this
    pool wakes, or its cached-but-free blocks starve the remap.
    """

    _tags = itertools.count()

    def __init__(self, allocator: Any, tag: str) -> None:
        self._allocator = allocator
        self.tag = tag
        self.asleep = False

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
        """Context manager: allocations made inside land in this pool's tag."""

        return self._allocator.use_memory_pool(tag=self.tag)

    def sleep(self) -> None:
        """Offload this tag's physical pages to pinned host RAM, free the GPU."""

        self._allocator.sleep(offload_tags=(self.tag,))
        self.asleep = True

    def wake(self) -> None:
        """Remap this tag's pages back onto the GPU (no-op when not asleep)."""

        if not self.asleep:
            return
        self._allocator.wake_up(tags=[self.tag])
        self.asleep = False


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Return whether an exception looks like a CUDA OOM failure."""

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


def cap_cuda_memory_fraction(fraction: float | None) -> None:
    """Hard-cap this process's CUDA allocator to ``fraction`` of the current device.

    Bounds the caching allocator so a colocated rollout worker cannot grow its
    reservation into memory the trainer needs on a shared GPU (the shared-GPU
    ``gpu_memory_utilization`` role). ``None`` leaves the allocator uncapped, which
    is correct for a worker that owns a dedicated GPU. Best-effort: a CPU-only or
    torch-less worker is a no-op.
    """

    if fraction is None:
        return
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {fraction}")
    try:
        import torch
    except Exception:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.set_per_process_memory_fraction(float(fraction), torch.cuda.current_device())
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
    "cap_cuda_memory_fraction",
    "empty_cuda_cache",
    "is_cuda_out_of_memory",
    "release_cuda_memory",
]
