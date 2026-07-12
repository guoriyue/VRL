"""Generation-forward bottleneck profiler — is it compute- or memory-bound, and which kernels?

WHY: GPU util=100% does NOT prove peak FLOPS — a bandwidth-bound kernel (attention
materializing an O(n^2) matrix, an oversized elementwise/copy) can pin util at 100%
while wasting the SMs. ``gemm_projection_breakdown.py`` splits *GEMM* time only; this
tool profiles the WHOLE denoise forward and answers two questions directly:

  1. Which kernels own the device time? (full top-N by self CUDA time, GEMM + attention
     + norm + elementwise + copy — so a hidden bandwidth-bound kernel shows up.)
  2. Is the forward compute- or memory-bound? (nvidia-smi dmon samples SM% vs MEM%
     hardware counters during the profiled window; MEM% >> SM% == bandwidth-bound.)

HOW: reuses the repo torch.profiler pattern (ProfilerActivity.CUDA + key_averages,
same as gemm_projection_breakdown / compile_benchmark). Kernels are bucketed by name
into gemm / attention / norm-elementwise / reduction / copy-memset / other so the
compute-vs-bandwidth split is legible without Nsight Compute (which isn't installed).

Usage:
    python -m vrl.scripts.perf.generation_bottleneck_profile \
        --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
        --steps 6 --warmup 2 --trace outputs/perf/gen_trace.json
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections import defaultdict

import torch

from vrl.config.loading import load_config
from vrl.config.precision import normalize_precision
from vrl.generation.diffusion.teacache import TeaCacheConfig
from vrl.models.dtypes import resolve_torch_dtype
from vrl.scripts.perf.common.diffusion_runtime import build_model, make_step_fn, run_e2e

# Kernel-name -> bucket. First substring match wins; lowercased CUDA kernel name.
# gemm/attention are compute-heavy; norm/elementwise/reduction/copy are typically
# bandwidth-bound. A bandwidth bucket dominating == the "slow 访存 kernel" smell.
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gemm", ("gemm", "cutlass", "sgemm", "ampere", "wgrad", "implicit", "cublas", "matmul")),
    ("attention", ("flash", "fmha", "attention", "scaled_dot", "mha", "softmax")),
    (
        "norm/elementwise",
        ("norm", "elementwise", "vectorized", "mul", "add", "silu", "gelu", "activation", "scale"),
    ),
    ("reduction", ("reduce", "sum", "mean", "var")),
    ("copy/memset", ("copy", "memcpy", "memset", "cat", "transpose", "permute", "contiguous")),
)


def _bucket(name: str) -> str:
    n = name.lower()
    for label, keys in _BUCKETS:
        if any(k in n for k in keys):
            return label
    return "other"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--e2e",
        action="store_true",
        help="measure full end-to-end image latency (encode+denoise+decode), "
        "matched scope to an inference engine's generate(); skips kernel profiling",
    )
    p.add_argument("--steps", type=int, default=6, help="profiled denoise steps")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--trace", default="outputs/perf/gen_trace.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the transformer before profiling (measures fusion effect)",
    )
    # Profiler shorthand, not a second public precision schema. Plain tokens select
    # the rollout base dtype; quantized tokens mean a BF16 base plus selective
    # rollout GEMMs.
    p.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "fp16", "bf16", "fp8", "nvfp4"],
        help=("profiler shorthand: fp32/fp16/bf16 base dtype, bf16+fp8, or bf16+nvfp4 rollout"),
    )
    p.add_argument(
        "--fp8-recipe",
        default="rowwise",
        choices=["rowwise", "tensorwise", "blockwise"],
        help="fp8 quant recipe (only with --precision fp8); blockwise reuses vLLM's "
        "1x128 triton block GEMM",
    )
    p.add_argument(
        "--teacache",
        type=float,
        default=None,
        help="enable rollout TeaCache at this rel-L1 threshold (e.g. 0.15); "
        "measures the skip speedup vs the full-forward baseline",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device)
    precision = normalize_precision(args.precision)
    fp8 = precision == "fp8"
    nvfp4 = precision == "nvfp4"
    dtype = torch.bfloat16 if fp8 or nvfp4 else resolve_torch_dtype(precision)
    label = f"bf16+fp8/{args.fp8_recipe}" if fp8 else "bf16+nvfp4" if nvfp4 else precision
    teacache_cfg = (
        TeaCacheConfig.from_sampling({"threshold": args.teacache})
        if args.teacache is not None
        else None
    )
    if teacache_cfg is not None:
        label = f"{label}+teacache(thr={args.teacache})"
    _nf = cfg.sampling.get("num_frames", cfg.sampling.get("frame_count", 1))
    print(
        f"shape {cfg.sampling.width}x{cfg.sampling.height}x{_nf}, "
        f"{cfg.sampling.num_steps} steps; precision={label}; profiling {args.steps} steps",
        flush=True,
    )

    model = build_model(cfg, device, dtype)
    if fp8:
        swapped = model.quantize_rollout_fp8(recipe=args.fp8_recipe)
        if not swapped:
            raise RuntimeError("FP8 profiling matched 0 quantizable policy linears")
        print(f"fp8/{args.fp8_recipe}: swapped {len(swapped)} transformer linears", flush=True)
    elif nvfp4:
        from vrl.nn.quantization import nvfp4_available

        if not nvfp4_available(device):
            raise SystemExit("nvfp4 profiling requires a Blackwell-class CUDA target")
        swapped = model.quantize_rollout_nvfp4()
        if not swapped:
            raise RuntimeError("NVFP4 profiling matched 0 quantizable policy MLP linears")
        print(f"nvfp4: swapped {len(swapped)} transformer MLP linears", flush=True)
    if args.compile:
        # use the model's own helper so BOTH self.transformer and pipeline.transformer
        # (forward_step reads self.transformer) point at the compiled module.
        print("torch.compile(default) the transformer ...", flush=True)
        model.torch_compile_transformer("default")
    if args.e2e:
        run_e2e(model, cfg, device, dtype)
        return
    step_fn, teacache_state = make_step_fn(model, cfg, device, dtype, teacache=teacache_cfg)

    # extra warmup when compiling so the (slow) first compiled call is excluded
    for i in range(args.warmup + (3 if args.compile else 0)):
        step_fn(i)
    torch.cuda.synchronize(device)
    # Reset peak after warmup so the reported peak reflects the profiled window's
    # steady-state forward, not the (larger) one-time load/warmup allocations.
    torch.cuda.reset_peak_memory_stats(device)

    # Sample SM% vs MEM% hardware counters during the profiled window (compute vs bandwidth).
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-c", str(max(2, args.steps))],
        stdout=subprocess.PIPE,
        text=True,
    )

    t0 = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(args.steps):
            step_fn(i)
        torch.cuda.synchronize(device)
    wall = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    dmon_out, _ = dmon.communicate(timeout=30)

    # Per-kernel device self time, bucketed. Count ONLY raw device kernels: skip the
    # CPU-side ``aten::`` dispatcher wrappers and ``cuda*`` runtime calls, which carry
    # the SAME device time as the kernel they launch (summing both double-counts —
    # that earlier made total CUDA time exceed wall time). Profiler markers excluded too.
    def _is_device_kernel(key: str) -> bool:
        k = key.strip()
        if (
            k.startswith("aten::")
            or k.startswith("cuda")
            or k.startswith("Command Buffer")
            or k.startswith("ProfilerStep")
            or k.startswith("Memcpy")
            or k.startswith("Memset")
        ):
            return key.startswith(("Memcpy", "Memset"))  # keep mem ops, drop aten/cuda/markers
        return True

    bucket_us: dict[str, float] = defaultdict(float)
    bucket_calls: dict[str, int] = defaultdict(int)
    top: list[tuple[float, str]] = []
    total_cuda_us = 0.0
    n_launches = 0
    for evt in prof.key_averages():
        if not _is_device_kernel(str(evt.key)):
            continue
        dev_us = float(
            getattr(evt, "self_device_time_total", 0)
            or getattr(evt, "self_cuda_time_total", 0)
            or 0
        )
        if dev_us <= 0:
            continue
        calls = int(getattr(evt, "count", 0) or 0)
        total_cuda_us += dev_us
        n_launches += calls
        b = _bucket(str(evt.key))
        bucket_us[b] += dev_us
        bucket_calls[b] += calls
        top.append((dev_us, str(evt.key)))

    total = total_cuda_us or 1.0
    print(
        f"\n=== wall {wall:.1f}s for {args.steps} steps ({wall / args.steps:.2f}s/step), "
        f"device-kernel time {total / 1e6:.2f}s, {n_launches} kernel launches "
        f"({n_launches / args.steps:.0f}/step) ==="
    )
    print(f"  peak GPU memory (profiled window): {peak_mb:.0f} MiB", flush=True)
    if teacache_state is not None:
        c = teacache_state.counters()
        print(
            f"  teacache: {c['teacache_skips']} skips / {c['teacache_runs']} runs "
            f"(skip ratio {c['teacache_skip_ratio']:.0%}, incl. warmup) ==="
        )
    print(
        "\n--- device time by bucket (compute = gemm+attention; bandwidth = norm/copy/reduction) ---"
    )
    comp = bucket_us.get("gemm", 0) + bucket_us.get("attention", 0)
    band = (
        bucket_us.get("norm/elementwise", 0)
        + bucket_us.get("copy/memset", 0)
        + bucket_us.get("reduction", 0)
    )
    for b in ("gemm", "attention", "norm/elementwise", "reduction", "copy/memset", "other"):
        v = bucket_us.get(b, 0.0)
        if v > 0:
            print(f"  {b:18s} {v / total * 100:5.1f}%  ({v / 1e6:.2f}s)")
    print(
        f"  → compute(gemm+attn) {comp / total * 100:.1f}%  vs  bandwidth(norm/copy/reduce) {band / total * 100:.1f}%"
    )

    print("\n--- top 15 kernels by self device time ---")
    for dev_us, name in sorted(top, reverse=True)[:15]:
        print(
            f"  {dev_us / total * 100:5.1f}%  {dev_us / 1e3:8.1f}ms  [{_bucket(name):16s}] {name[:70]}"
        )

    print("\n--- nvidia-smi dmon (sm% = compute util, mem% = memory-controller util) ---")
    print("    bandwidth-bound smell: mem% sustained near or above sm%")
    for line in dmon_out.strip().splitlines()[-(args.steps + 2) :]:
        print("   ", line)

    import os

    os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
    prof.export_chrome_trace(args.trace)
    print(f"\nchrome trace -> {args.trace} (open in chrome://tracing or perfetto.dev)")


if __name__ == "__main__":
    main()
