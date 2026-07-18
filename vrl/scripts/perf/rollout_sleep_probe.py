"""Does level-1 sleep free enough GPU for a colocated trainer, and how much does
vLLM's CuMemAllocator cut the wake cost vs a naive to(cpu)/to(gpu) round trip?

SPRINT_frozen_component_preservation defect A turns on-demand rollout handoff's
per-cycle teardown (level-2: kill the worker, ``from_pretrained`` re-reads
the frozen VAE / text-encoders from disk) into level-1 offload-and-restore. Two
backends implement the offload:

  - ``naive``: ``model.to("cpu")`` + ``empty_cache()`` + ``model.to(device)`` —
    pays a full cudaFree/cudaMalloc round trip and fragments the pool.
  - ``cumem``: allocate the model inside vLLM's CuMemAllocator pool, then
    ``sleep``/``wake_up`` release+restore the *physical* pages while keeping the
    virtual addresses mapped — no re-malloc, no fragmentation. This is the
    mechanism verl-omni gets for free because its rollout already runs on vLLM.

It reports, per backend:
  1. RESIDUAL after sleep = reserved GPU a slept worker still holds (the headroom a
     colocated trainer loses vs a full teardown).
  2. sleep+wake wall time, against the cold ``from_build`` reload it replaces.

This is the GPU-side acceptance probe for the sprint; one-shot, not imported by
runtime code. Needs a free GPU (the model is ~11 GB resident).

Run: python -m vrl.scripts.perf.rollout_sleep_probe --backend both
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from typing import Any

import torch

from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.sd3_5.model import SD3_5Model

_WEIGHTS_TAG = "weights"


def _gpu_used_mb() -> float:
    """Driver-level GPU memory in use (total - free).

    Authoritative for BOTH backends: ``memory_reserved()`` only sees PyTorch's
    caching-allocator pool, NOT cumem's CUDA virtual-memory pages, so it reports a
    sleeping cumem worker as still-resident. ``mem_get_info`` is the physical truth
    a colocated trainer actually competes for.
    """
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def _build(model_name: str, device: str, dtype: torch.dtype) -> SD3_5Model:
    build = SimpleNamespace(
        model_name_or_path=model_name,
        device=device,
        parameter_dtype=dtype,
        rollout=None,  # let from_build apply the Flow-GRPO frozen-dtype default
    )
    return SD3_5Model.from_build(build)


def _cumem():
    from vllm.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance()


def _do_sleep(alloc: Any, model: SD3_5Model) -> None:
    """Offload the model: cumem releases pooled physical pages; naive moves to CPU."""
    if alloc is not None:
        alloc.sleep(offload_tags=(_WEIGHTS_TAG,))
        return
    model.to("cpu")
    model.move_frozen_components("cpu")
    torch.cuda.empty_cache()


def _do_wake(alloc: Any, model: SD3_5Model, device: str) -> None:
    """Restore the model: cumem remaps physical pages; naive moves back to GPU."""
    if alloc is not None:
        alloc.wake_up(tags=[_WEIGHTS_TAG])
        return
    model.to(device)
    model.move_frozen_components(device)


def _stats_ms(durations: list[float]) -> str:
    ms = sorted(d * 1e3 for d in durations)
    median = ms[len(ms) // 2]
    return f"min {ms[0]:7.1f} | median {median:7.1f} | max {ms[-1]:7.1f} ms"


def _measure(
    backend: str,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    cycles: int,
    warmup: int = 2,
) -> None:
    print(f"\n===== backend: {backend} =====")
    torch.cuda.empty_cache()
    torch.zeros(1, device=device)  # ensure CUDA context exists
    torch.cuda.synchronize()
    base = _gpu_used_mb()
    print(f"baseline GPU used (context only)    = {base:8.1f} MB")

    alloc = _cumem() if backend == "cumem" else None
    if alloc is not None:
        with alloc.use_memory_pool(tag=_WEIGHTS_TAG):
            model = _build(model_name, device, dtype)
    else:
        model = _build(model_name, device, dtype)

    torch.cuda.synchronize()
    loaded = _gpu_used_mb()
    print(f"GPU used after load                    = {loaded:8.1f} MB  (+{loaded - base:.1f})")

    _do_sleep(alloc, model)
    torch.cuda.synchronize()
    slept = _gpu_used_mb()
    print(
        f"GPU used after sleep (RESIDUAL)    = {slept:8.1f} MB  "
        f"(trainer reclaims {loaded - slept:.1f} MB)"
    )

    _do_wake(alloc, model, device)
    torch.cuda.synchronize()
    woken = _gpu_used_mb()
    frag = woken - loaded
    print(
        f"GPU used after wake                    = {woken:8.1f} MB  "
        f"({'+' if frag >= 0 else ''}{frag:.1f} vs load — fragmentation)"
    )

    # Warm up so the first-touch allocator/cudaMalloc costs don't skew the stats.
    for _ in range(warmup):
        _do_sleep(alloc, model)
        torch.cuda.synchronize()
        _do_wake(alloc, model, device)
        torch.cuda.synchronize()

    sleep_s, wake_s, total_s = [], [], []
    for _ in range(cycles):
        t0 = time.perf_counter()
        _do_sleep(alloc, model)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _do_wake(alloc, model, device)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sleep_s.append(t1 - t0)
        wake_s.append(t2 - t1)
        total_s.append(t2 - t0)
    print(f"sleep      ({cycles} cyc, {warmup} warmup): {_stats_ms(sleep_s)}")
    print(f"wake       ({cycles} cyc, {warmup} warmup): {_stats_ms(wake_s)}")
    print(f"sleep+wake ({cycles} cyc, {warmup} warmup): {_stats_ms(total_s)}")

    del model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--backend", default="both", choices=["naive", "cumem", "both"])
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("rollout_sleep_probe needs a CUDA device")

    dtype = resolve_torch_dtype(args.dtype)
    torch.cuda.set_device(args.device)

    # Cold reload baseline (what sleep replaces), measured once with a warm/cold OS
    # page cache as-is on this box. Excludes the Ray actor respawn + CUDA context
    # init that the real teardown path also pays.
    t0 = time.perf_counter()
    cold_model = _build(args.model, args.device, dtype)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1e3
    del cold_model
    torch.cuda.empty_cache()
    print(f"cold from_build reload (replaced by sleep) = {cold_ms:8.1f} ms")

    backends = ["naive", "cumem"] if args.backend == "both" else [args.backend]
    for backend in backends:
        _measure(backend, args.model, args.device, dtype, args.cycles, args.warmup)


if __name__ == "__main__":
    main()
