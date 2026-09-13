"""CUDA memory helpers shared across engine, rollout, and trainer code."""

from __future__ import annotations

import gc
import os

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


def is_cuda_out_of_memory(exc: BaseException | str) -> bool:
    """Recognize CUDA/HIP allocator failures from local exceptions or remote text."""

    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "hip" in message)


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


def gpu_process_used_bytes(device: str | None = None) -> int:
    """Physical CUDA memory attributed to this process, including CuMem pools.

    Whole-device usage includes unrelated processes; Torch allocated bytes include
    unmapped CuMem virtual tensors. Neither can prove this owner's physical release.
    Missing process accounting (for example unsupported MPS/PID namespaces) must
    fail closed, never silently report zero or fall back to whole-device usage.
    """
    if device is not None and not str(device).startswith("cuda"):
        return 0
    import torch

    if not torch.cuda.is_available():
        return 0
    import pynvml

    target = torch.device(device) if device is not None else torch.device("cuda")
    torch.cuda.synchronize(target)
    # CUDA ordinals may be reordered by CUDA_VISIBLE_DEVICES; NVML ordinals are
    # physical. UUID addresses the actual device (or MIG instance) without guessing.
    uuid = getattr(torch.cuda.get_device_properties(target), "uuid", None)
    if not uuid:
        raise RuntimeError("CUDA device UUID is required for process memory accounting")
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        matches = [entry for entry in processes if entry.pid == os.getpid()]
        if len(matches) != 1:
            raise RuntimeError("NVML cannot identify this CUDA process unambiguously")
        used = matches[0].usedGpuMemory
        if type(used) is not int or not 0 <= used < 2**64 - 1:
            raise RuntimeError("NVML physical memory accounting is unavailable for this process")
        return used
    finally:
        pynvml.nvmlShutdown()


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


def validate_parking_residual(
    *,
    residual_bytes: int,
    baseline_bytes: int,
    limit_bytes: int,
    context: str,
) -> None:
    """One invariant behind every GPU-parking release check: residual <= baseline + limit.

    Shared by the generation worker parking snapshot and the reward inference
    runtime so the two engines cannot drift on what a complete release means.
    """

    if min(residual_bytes, baseline_bytes, limit_bytes) < 0:
        raise ValueError(f"{context} byte counts must be >= 0")
    if residual_bytes > baseline_bytes + limit_bytes:
        raise RuntimeError(
            f"incomplete {context}: residual={residual_bytes} "
            f"baseline={baseline_bytes} limit={limit_bytes}",
        )


__all__ = [
    "CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT",
    "cuda_peak_allocated_bytes",
    "cuda_peak_allocated_mb",
    "empty_cuda_cache",
    "gpu_process_used_bytes",
    "is_cuda_out_of_memory",
    "release_cuda_memory",
    "release_cuda_memory_for_parking",
    "reset_cuda_peak",
    "validate_parking_residual",
]
