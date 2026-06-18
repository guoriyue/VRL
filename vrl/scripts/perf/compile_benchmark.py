"""torch.compile A/B for the diffusion DiT forward (and train step).

WHY: SPRINT_gemm_utilization.md P2 asks one cheap question -- is flipping
``model.torch_compile.enable:true`` on cosmos-predict2 (already full-param, no
LoRA fragments) worth it? The sprint's prior is "near-zero-cost, low payoff":
the load is launch-bound, and the runtime compiles with ``fullgraph=False`` while
training runs with gradient checkpointing, both of which block the CUDA-graph
launch-elimination that would make compile a big win. This tool turns that prior
into a measured number instead of a guess.

WHAT it measures, for compile OFF vs ON, on two paths that mirror the real code:
  - rollout: forward-only under ``no_grad`` (the 35-step x CFG denoise loop is
    just this forward repeated -- the dominant rollout cost).
  - train:   forward + backward with gradient checkpointing enabled, matching
    ``vrl/scripts/diffusion/cosmos/train.py:131`` (default True).
Per cell: median step latency, CUDA kernel launches issued per step (the
launch-bound signal -- compile fuses elementwise epilogues into fewer kernels),
and peak memory.

FAITHFULNESS: this compiles the bare diffusers transformer exactly as the runtime
does -- ``torch.compile(transformer, mode=mode, fullgraph=False)`` (see
``vrl/models/diffusion/base.py:200``). Compile's fusion/guard decisions depend on
the op graph and tensor shapes, both reproduced exactly by the config-init
synthetic model (``build_synthetic_inputs``); weight VALUES never affect compile
behavior or launch structure, so no checkpoint download is needed. The synthetic
forward uses the cosmos Text2Image backbone dims at a modest video latent grid --
the per-step compile EFFECT (launch reduction %, latency delta) is structural and
representative; absolute ms scales with the real grid.

Usage:
    python -m vrl.scripts.perf.compile_benchmark --family cosmos-predict2 --device cuda
    # fast smoke (few layers, still exercises both paths + compile):
    python -m vrl.scripts.perf.compile_benchmark --layers 4 --iters 5 --warmup 3
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vrl.scripts.perf.gemm_projection_breakdown import build_synthetic_inputs
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class CellResult:
    """One (path, compile) measurement cell."""

    path: str  # "rollout" | "train"
    compiled: bool
    latency_ms: float
    launches_per_step: float
    peak_mem_mb: float


def _median_latency_ms(step_fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    """Median wall time of one step via CUDA events (ms), after warmup.

    Warmup absorbs the first compiled call (graph capture + guard install), which
    is one-time and not part of steady-state throughput.
    """

    for _ in range(max(0, warmup)):
        step_fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _launches_per_step(step_fn: Callable[[], None], *, steps: int = 3) -> float:
    """Host-side kernel launches issued per step (the launch-bound metric).

    Counts every kernel-launch runtime/driver call -- ``cudaLaunchKernel`` (eager
    aten) AND ``cuLaunchKernel`` (inductor's Triton kernels) -- so a compile that
    fuses N elementwise ops into one Triton kernel shows up as fewer launches.
    Caller must warm up first so compilation isn't counted.
    """

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(steps):
            step_fn()
        torch.cuda.synchronize()

    launches = 0
    for evt in prof.key_averages():
        if "launchkernel" in str(evt.key).lower():
            launches += int(getattr(evt, "count", 0) or 0)
    return launches / float(steps)


def _maybe_compile(model: torch.nn.Module, *, compiled: bool, mode: str) -> torch.nn.Module:
    """Apply torch.compile exactly as the runtime does, or return eager."""

    if not compiled:
        return model
    # Mirrors vrl/models/diffusion/base.py:200 (the only place the runtime compiles).
    return torch.compile(model, mode=mode, fullgraph=False)


def _run_cell(
    *,
    path: str,
    compiled: bool,
    family: str,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    layers: int | None,
    mode: str,
    concat_padding_mask: bool,
    warmup: int,
    iters: int,
) -> CellResult:
    """Build a fresh model, configure the path, then time + count launches."""

    # Fresh dynamo state so a prior compiled cell's cache/guards don't bleed in.
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model, kwargs = build_synthetic_inputs(
        family, batch=batch, device=device, dtype=dtype, layers=layers,
        concat_padding_mask=concat_padding_mask,
    )

    if path == "rollout":
        model.eval()
        runner = _maybe_compile(model, compiled=compiled, mode=mode)

        # Bind the closure vars as defaults (early binding): the enclosing scope
        # `del`s them below, which makes ruff flag them as possibly-unbound inside
        # the closure (F821). Defaults pin the same objects and read as defined.
        def step_fn(runner=runner, kwargs=kwargs) -> None:
            with torch.no_grad():
                runner(**kwargs)

    elif path == "train":
        # Match the real train path: full-param grads + gradient checkpointing
        # (vrl/scripts/diffusion/cosmos/train.py:131, default True). Configure the
        # eager module fully, THEN compile, so the OptimizedModule wraps the final
        # state (grad-ckpt is a forward-time flag; pre- vs post-compile is equivalent).
        model.train()
        model.requires_grad_(True)
        model.enable_gradient_checkpointing()
        runner = _maybe_compile(model, compiled=compiled, mode=mode)

        # Bind closure vars as defaults (early binding) — see rollout branch.
        def step_fn(runner=runner, kwargs=kwargs, model=model) -> None:
            out = runner(**kwargs)
            sample = out[0] if isinstance(out, tuple) else out
            loss = sample.float().pow(2).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown path {path!r}")

    latency = _median_latency_ms(step_fn, warmup=warmup, iters=iters)
    launches = _launches_per_step(step_fn)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    del model, runner, kwargs
    torch.cuda.empty_cache()
    return CellResult(
        path=path, compiled=compiled, latency_ms=latency,
        launches_per_step=launches, peak_mem_mb=peak_mb,
    )


def format_report(results: list[CellResult], *, mode: str) -> str:
    """Render cells grouped by path with the eager->compiled speedup per path."""

    lines = [
        f"torch.compile A/B  (mode={mode}, fullgraph=False; matches base.py:200)",
        f"{'path':<9}{'compile':<9}{'latency ms':>12}{'launches/step':>15}{'peak MB':>10}{'speedup':>10}",
        "-" * 65,
    ]
    by_path: dict[str, dict[bool, CellResult]] = {}
    for r in results:
        by_path.setdefault(r.path, {})[r.compiled] = r

    for path in ("rollout", "train"):
        cells = by_path.get(path)
        if not cells:
            continue
        eager = cells.get(False)
        for compiled in (False, True):
            r = cells.get(compiled)
            if r is None:
                continue
            speedup = ""
            if compiled and eager is not None and r.latency_ms > 0:
                speedup = f"{eager.latency_ms / r.latency_ms:.2f}x"
            lines.append(
                f"{path:<9}{('on' if compiled else 'off'):<9}"
                f"{r.latency_ms:>12.3f}{r.launches_per_step:>15.1f}"
                f"{r.peak_mem_mb:>10.0f}{speedup:>10}"
            )
        lines.append("-" * 65)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="cosmos-predict2",
        choices=["sd3_5", "cosmos-predict2", "cosmos-predict2.5", "wan_2_1"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch", type=int, default=2, help="2 == cond+uncond CFG pair")
    parser.add_argument(
        "--layers", type=int, default=None,
        help="override transformer depth (default: production; small value = fast smoke)",
    )
    parser.add_argument("--mode", default="default", help="torch.compile mode (runtime uses 'default')")
    parser.add_argument("--warmup", type=int, default=8, help="steps before timing (absorbs compilation)")
    parser.add_argument("--iters", type=int, default=20, help="timed steps (median taken)")
    parser.add_argument(
        "--paths", default="rollout,train",
        help="comma list of paths to measure: rollout,train",
    )
    parser.add_argument(
        "--no-concat-padding-mask", dest="concat_padding_mask", action="store_false",
        help="cosmos only: skip the torchvision padding-mask resize (no GEMM impact)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("compile_benchmark needs --device cuda (latency/launch counts are GPU-only)")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]

    logger.info(
        "compile A/B: family=%s device=%s dtype=%s batch=%d layers=%s mode=%s paths=%s",
        args.family, device, dtype, args.batch, args.layers, args.mode, paths,
    )

    results: list[CellResult] = []
    for path in paths:
        for compiled in (False, True):
            logger.info("running cell: path=%s compile=%s", path, compiled)
            results.append(
                _run_cell(
                    path=path, compiled=compiled, family=args.family, batch=args.batch,
                    device=device, dtype=dtype, layers=args.layers, mode=args.mode,
                    concat_padding_mask=args.concat_padding_mask,
                    warmup=args.warmup, iters=args.iters,
                )
            )

    print(format_report(results, mode=args.mode))


if __name__ == "__main__":
    main()
