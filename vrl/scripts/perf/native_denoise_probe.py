"""Native denoise s/step probe — pure wall-clock, no torch.profiler.

This measures VRL's denoise loop only. ``vllm_omni_diffusion_profile.py`` measures
the full ``Omni.generate`` request, including engine and worker boundaries, so the
two tools share a prompt protocol but their latency numbers are not directly
comparable. Use this probe for native before/after measurements; a fair cross-engine
comparison needs equivalent stage-level timing from both engines.

WHY a separate script from ``generation_bottleneck_profile.py``: that profiler wraps
the timed loop in ``torch.profiler.profile(record_shapes=True)`` to get the kernel
breakdown, which adds large CPU dispatch overhead for launch-bound diffusion forwards
(thousands of kernel launches/step) and inflates the reported s/step. For a clean
engine-vs-engine throughput number we must time WITHOUT the profiler attached.

Matches the omni profiler's scope: base weights (no LoRA, like the omni side), frozen
encoders/VAE parked on CPU (denoise-only window), torch.compile on (omni auto-compiles).

Usage (main env):
    HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
      python -m vrl.scripts.perf.native_denoise_probe \\
        --config experiment/flux/online_grpo_smoke_single_gpu \\
        --steps 20 --compile --out /mnt/nvme/perf/flux_native.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.scripts.perf.common.diffusion_runtime import build_runtime, make_step_fn


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=20, help="timed denoise steps")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="outputs/perf/native_denoise.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    # Match the omni side: base weights, no LoRA adapter.
    cfg.model.use_lora = False
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    device = torch.device(args.device)
    runtime = build_runtime(root, device, precision=precision)
    model = runtime.model
    if args.compile:
        print("torch.compile(default) the transformer ...", flush=True)
        model.torch_compile_transformer("default")

    # make_step_fn parks frozen encoders/VAE on CPU and returns a closure that runs
    # one denoise forward + SDE step (the exact rollout inner loop).
    step_fn = make_step_fn(runtime, cfg)

    # Extra warmup when compiling so the (slow) first compiled call is excluded.
    for i in range(args.warmup + (4 if args.compile else 0)):
        step_fn(i)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    for i in range(args.steps):
        step_fn(i)
    torch.cuda.synchronize(device)
    wall = time.time() - t0
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    s = cfg.sampling
    nf = int(s.get("num_frames", s.get("frame_count", 1)))
    result = {
        "backend": "native(vrl)",
        "config": args.config,
        "compiled": bool(args.compile),
        "shape": [int(s.width), int(s.height), nf],
        "steps_timed": args.steps,
        "wall_s": wall,
        "s_per_step": wall / args.steps,
        "denoise_peak_mib": peak_mib,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        f"native denoise [{args.config}] compiled={args.compile}: "
        f"{args.steps} steps, {wall:.2f}s ({wall / args.steps * 1000:.1f} ms/step), "
        f"denoise peak {peak_mib:.0f} MiB -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
