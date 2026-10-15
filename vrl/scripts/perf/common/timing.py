"""CUDA-event timing helpers shared by perf probe CLIs."""

from __future__ import annotations

from collections.abc import Callable

import torch


def cuda_mean_ms(
    step_fn: Callable[[], object],
    *,
    iters: int,
    warmup: int = 3,
) -> float:
    """Mean CUDA-event wall time for ``step_fn`` after warmup."""

    for _ in range(max(0, warmup)):
        step_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        step_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(1, iters)


def cuda_step_times_ms(
    step_fn: Callable[[], object],
    *,
    iters: int,
) -> list[float]:
    """Per-call CUDA-event wall times for ``iters`` already-warm steps, in call order."""

    times: list[float] = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def cuda_median_ms(
    step_fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> float:
    """Median CUDA-event wall time for one steady-state step."""

    for _ in range(max(0, warmup)):
        step_fn()
    torch.cuda.synchronize()
    times = sorted(cuda_step_times_ms(step_fn, iters=iters))
    return times[len(times) // 2]
