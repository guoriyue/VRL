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
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.math.diffusion.flow_matching import sde_step_with_logprob

# Kernel-name -> bucket. First substring match wins; lowercased CUDA kernel name.
# gemm/attention are compute-heavy; norm/elementwise/reduction/copy are typically
# bandwidth-bound. A bandwidth bucket dominating == the "slow 访存 kernel" smell.
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gemm", ("gemm", "cutlass", "sgemm", "ampere", "wgrad", "implicit", "cublas", "matmul")),
    ("attention", ("flash", "fmha", "attention", "scaled_dot", "mha", "softmax")),
    ("norm/elementwise", ("norm", "elementwise", "vectorized", "mul", "add", "silu", "gelu", "activation", "scale")),
    ("reduction", ("reduce", "sum", "mean", "var")),
    ("copy/memset", ("copy", "memcpy", "memset", "cat", "transpose", "permute", "contiguous")),
)


def _bucket(name: str) -> str:
    n = name.lower()
    for label, keys in _BUCKETS:
        if any(k in n for k in keys):
            return label
    return "other"


def build_model(cfg, device, dtype):
    from vrl.models.diffusion.cosmos.predict2_5.runtime import (
        build_cosmos_predict25_runtime_bundle,
        extract_cosmos_predict25_runtime_spec,
    )
    cfg.model.use_lora = True
    spec = extract_cosmos_predict25_runtime_spec(cfg, device, dtype)
    return build_cosmos_predict25_runtime_bundle(spec).model


def make_step_fn(model, cfg, device, dtype):
    s = cfg.sampling
    enc = model.encode_prompt(["a physical scene, high quality"], None,
                              guidance_scale=float(s.guidance_scale),
                              max_sequence_length=int(s.max_sequence_length))
    req = VideoGenerationRequest(prompt="a physical scene, high quality", negative_prompt=None,
            width=int(s.width), height=int(s.height), frame_count=int(s.num_frames),
            num_steps=int(s.num_steps), guidance_scale=float(s.guidance_scale), seed=0,
            extra={"max_sequence_length": int(s.max_sequence_length)})
    state = model.prepare_sampling(req, enc)

    def one_step(idx: int):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward_step(state, idx % int(s.num_steps))
            r = sde_step_with_logprob(state.scheduler, out["noise_pred"].float(),
                    state.timesteps[idx % int(s.num_steps)].unsqueeze(0), state.latents.float(),
                    generator=None, deterministic=True, sde_type="cps")
            state.latents = r.prev_sample
    return one_step


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=6, help="profiled denoise steps")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--trace", default="outputs/perf/gen_trace.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer before profiling (measures fusion effect)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    print(f"shape {cfg.sampling.width}x{cfg.sampling.height}x{cfg.sampling.num_frames}, "
          f"{cfg.sampling.num_steps} steps; profiling {args.steps} steps", flush=True)

    model = build_model(cfg, device, dtype)
    if args.compile:
        # use the model's own helper so BOTH self.transformer and pipeline.transformer
        # (forward_step reads self.transformer) point at the compiled module.
        print("torch.compile(default) the transformer ...", flush=True)
        model.torch_compile_transformer("default")
    step_fn = make_step_fn(model, cfg, device, dtype)

    # extra warmup when compiling so the (slow) first compiled call is excluded
    for i in range(args.warmup + (3 if args.compile else 0)):
        step_fn(i)
    torch.cuda.synchronize(device)

    # Sample SM% vs MEM% hardware counters during the profiled window (compute vs bandwidth).
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-c", str(max(2, args.steps))],
        stdout=subprocess.PIPE, text=True)

    t0 = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(args.steps):
            step_fn(i)
        torch.cuda.synchronize(device)
    wall = time.time() - t0
    dmon_out, _ = dmon.communicate(timeout=30)

    # Per-kernel device self time, bucketed. Count ONLY raw device kernels: skip the
    # CPU-side ``aten::`` dispatcher wrappers and ``cuda*`` runtime calls, which carry
    # the SAME device time as the kernel they launch (summing both double-counts —
    # that earlier made total CUDA time exceed wall time). Profiler markers excluded too.
    def _is_device_kernel(key: str) -> bool:
        k = key.strip()
        if k.startswith("aten::") or k.startswith("cuda") or k.startswith("Command Buffer") \
                or k.startswith("ProfilerStep") or k.startswith("Memcpy") or k.startswith("Memset"):
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
        dev_us = float(getattr(evt, "self_device_time_total", 0) or
                       getattr(evt, "self_cuda_time_total", 0) or 0)
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
    print(f"\n=== wall {wall:.1f}s for {args.steps} steps ({wall/args.steps:.2f}s/step), "
          f"device-kernel time {total/1e6:.2f}s, {n_launches} kernel launches "
          f"({n_launches/args.steps:.0f}/step) ===")
    print("\n--- device time by bucket (compute = gemm+attention; bandwidth = norm/copy/reduction) ---")
    comp = bucket_us.get("gemm", 0) + bucket_us.get("attention", 0)
    band = bucket_us.get("norm/elementwise", 0) + bucket_us.get("copy/memset", 0) + bucket_us.get("reduction", 0)
    for b in ("gemm", "attention", "norm/elementwise", "reduction", "copy/memset", "other"):
        v = bucket_us.get(b, 0.0)
        if v > 0:
            print(f"  {b:18s} {v/total*100:5.1f}%  ({v/1e6:.2f}s)")
    print(f"  → compute(gemm+attn) {comp/total*100:.1f}%  vs  bandwidth(norm/copy/reduce) {band/total*100:.1f}%")

    print("\n--- top 15 kernels by self device time ---")
    for dev_us, name in sorted(top, reverse=True)[:15]:
        print(f"  {dev_us/total*100:5.1f}%  {dev_us/1e3:8.1f}ms  [{_bucket(name):16s}] {name[:70]}")

    print("\n--- nvidia-smi dmon (sm% = compute util, mem% = memory-controller util) ---")
    print("    bandwidth-bound smell: mem% sustained near or above sm%")
    for line in dmon_out.strip().splitlines()[-(args.steps + 2):]:
        print("   ", line)

    import os
    os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
    prof.export_chrome_trace(args.trace)
    print(f"\nchrome trace -> {args.trace} (open in chrome://tracing or perfetto.dev)")


if __name__ == "__main__":
    main()
