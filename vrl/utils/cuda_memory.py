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
    "empty_cuda_cache",
    "is_cuda_out_of_memory",
    "release_cuda_memory",
]
