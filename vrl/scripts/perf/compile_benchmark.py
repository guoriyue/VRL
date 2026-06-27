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
from dataclasses import dataclass
from typing import Any

import torch

from vrl.scripts.perf.common.synthetic_diffusion import build_synthetic_inputs
from vrl.scripts.perf.common.timing import cuda_median_ms, kernel_launches_per_step
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


@dataclass(slots=True)
class ParityResult:
    """One eager-vs-compiled numeric parity cell.

    The launch-bound speedup says compile is *worth it*; this says compile is
    *safe* -- fusing kernels reorders reductions, so the open question before
    flipping ``torch_compile.enable`` is whether the forward (rollout) and
    forward+backward under grad-ckpt (train) stay numerically faithful to eager.
    """

    path: str  # "rollout" | "train"
    max_abs_out: float
    max_rel_out: float
    max_abs_grad: float | None  # train path only (None for rollout)
    params_checked: int | None


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

    latency = cuda_median_ms(step_fn, warmup=warmup, iters=iters)
    launches = kernel_launches_per_step(step_fn)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    del model, runner, kwargs
    torch.cuda.empty_cache()
    return CellResult(
        path=path, compiled=compiled, latency_ms=latency,
        launches_per_step=launches, peak_mem_mb=peak_mb,
    )


def _extract_sample(out: Any) -> torch.Tensor:
    """Pull the prediction tensor out of the transformer return (mirror _run_cell)."""

    return out[0] if isinstance(out, tuple) else out


def _abs_rel(eager: torch.Tensor, compiled: torch.Tensor) -> tuple[float, float]:
    """Max absolute and max relative difference between two tensors (in fp32)."""

    diff = (eager - compiled).abs()
    return diff.max().item(), (diff / eager.abs().clamp_min(1e-8)).max().item()


def _run_parity_cell(
    *,
    path: str,
    family: str,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    layers: int | None,
    mode: str,
    concat_padding_mask: bool,
) -> ParityResult:
    """Run identical inputs through eager then compiled (same weights, same RNG)
    and report the divergence. Compile wraps the *same* module, so any delta is
    pure kernel-fusion/reduction-order, not a weight difference. Use --dtype fp32
    for the faithfulness bound (epsilon-level == compile-safe); bf16 shows the
    production drift the rollout/train logprob guard must absorb."""

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    model, kwargs = build_synthetic_inputs(
        family, batch=batch, device=device, dtype=dtype, layers=layers,
        concat_padding_mask=concat_padding_mask,
    )

    if path == "rollout":
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            out_eager = _extract_sample(model(**kwargs)).float().clone()
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        torch.manual_seed(0)
        with torch.no_grad():
            out_comp = _extract_sample(compiled(**kwargs)).float().clone()
        max_abs, max_rel = _abs_rel(out_eager, out_comp)
        result = ParityResult(path, max_abs, max_rel, None, None)

    elif path == "train":
        # Full-param grads + grad-ckpt -- the exact train replay path (and the one
        # the LoRA+grad-ckpt parity worry is about). Snapshot eager grads, then run
        # the compiled module on the same inputs and diff both output and grads.
        model.train()
        model.requires_grad_(True)
        model.enable_gradient_checkpointing()

        torch.manual_seed(0)
        out_eager = _extract_sample(model(**kwargs))
        out_eager.float().pow(2).mean().backward()
        # Offload the eager-grad snapshot to CPU so it does not co-reside on GPU
        # with the compiled backward's activations+grads (heavy grids OOM otherwise).
        eager_grads = {
            name: p.grad.detach().float().cpu().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        out_eager_t = out_eager.detach().float().cpu().clone()
        model.zero_grad(set_to_none=True)

        compiled = torch.compile(model, mode=mode, fullgraph=False)
        torch.manual_seed(0)
        out_comp = _extract_sample(compiled(**kwargs))
        out_comp.float().pow(2).mean().backward()

        max_abs_grad = 0.0
        checked = 0
        for name, p in model.named_parameters():
            ref = eager_grads.get(name)
            if p.grad is None or ref is None:
                continue
            max_abs_grad = max(
                max_abs_grad, (p.grad.detach().float().cpu() - ref).abs().max().item()
            )
            checked += 1
        max_abs, max_rel = _abs_rel(out_eager_t, out_comp.detach().float().cpu())
        result = ParityResult(path, max_abs, max_rel, max_abs_grad, checked)

    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown path {path!r}")

    del model
    torch.cuda.empty_cache()
    return result


def format_parity_report(results: list[ParityResult], *, dtype: str) -> str:
    """Render eager-vs-compiled max |Δ| per path (output, and grad for train)."""

    lines = [
        f"torch.compile parity  (eager vs compiled, dtype={dtype}, fullgraph=False)",
        f"{'path':<9}{'max|Δ out|':>14}{'max rel out':>14}{'max|Δ grad|':>14}{'params':>9}",
        "-" * 60,
    ]
    for r in results:
        grad = f"{r.max_abs_grad:.3e}" if r.max_abs_grad is not None else "n/a"
        params = str(r.params_checked) if r.params_checked is not None else "n/a"
        lines.append(
            f"{r.path:<9}{r.max_abs_out:>14.3e}{r.max_rel_out:>14.3e}{grad:>14}{params:>9}"
        )
    lines.append("-" * 60)
    return "\n".join(lines)


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
    parser.add_argument(
        "--parity", action="store_true",
        help="numeric eager-vs-compiled parity (max |Δ| output/grad) instead of timing",
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

    if args.parity:
        parity: list[ParityResult] = []
        for path in paths:
            logger.info("running parity cell: path=%s", path)
            parity.append(
                _run_parity_cell(
                    path=path, family=args.family, batch=args.batch, device=device,
                    dtype=dtype, layers=args.layers, mode=args.mode,
                    concat_padding_mask=args.concat_padding_mask,
                )
            )
        print(format_parity_report(parity, dtype=args.dtype))
        return

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
