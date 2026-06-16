"""CUDA memory helpers shared across engine, rollout, and trainer code."""

from __future__ import annotations


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
    reservation into memory the trainer needs on a shared GPU (the vLLM/cosmos-rl
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
    "cap_cuda_memory_fraction",
    "empty_cuda_cache",
    "is_cuda_out_of_memory",
    "release_cuda_memory",
]
