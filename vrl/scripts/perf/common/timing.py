"""Timing and profiler helpers shared by perf probe CLIs."""

from __future__ import annotations

from collections.abc import Callable

import torch


def cuda_mean_ms(
    step_fn: Callable[[], object],
    *,
    iters: int,
    warmup: int = 3,
    device: torch.device | str | None = None,
) -> float:
    """Mean CUDA-event wall time for ``step_fn`` after warmup."""

    for _ in range(max(0, warmup)):
        step_fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        step_fn()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / max(1, iters)


def cuda_median_ms(
    step_fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    device: torch.device | str | None = None,
) -> float:
    """Median CUDA-event wall time for one steady-state step."""

    for _ in range(max(0, warmup)):
        step_fn()
    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def kernel_launches_per_step(step_fn: Callable[[], object], *, steps: int = 3) -> float:
    """Host-side CUDA kernel launch calls issued per step.

    Counts both ``cudaLaunchKernel`` and ``cuLaunchKernel`` so eager and inductor
    paths are comparable after compilation warmup.
    """

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(max(1, steps)):
            step_fn()
        torch.cuda.synchronize()

    launches = 0
    for evt in prof.key_averages():
        if "launchkernel" in str(evt.key).lower():
            launches += int(getattr(evt, "count", 0) or 0)
    return launches / float(max(1, steps))
