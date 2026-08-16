"""Profile end-to-end diffusion requests through the real vLLM-Omni engine.

This times ``Omni.generate`` including engine and worker boundaries. The closest
native scope is ``generation_bottleneck_profile.py --e2e``;
``native_denoise_probe.py`` times only VRL's denoise loop and is not a direct
cross-engine latency counterpart. All three tools share the same prompt protocol.

NOT imported by any long-term code — it is a standalone CLI. It needs a Python env
where ``import vllm_omni`` works (vllm and vllm-omni versions must match; this repo
ships such an env at ``.venvs/vllm-omni`` = vllm 0.22.0 + vllm_omni 0.22.0). The
naive profiler runs in the main env; this one runs in that venv.

vLLM-Omni API (entrypoints/omni.py, diffusion/{diffusion_engine,registry,request}.py,
inputs/data.py):
  Omni(model=<hf id or path>, dtype=..., enable_cpu_offload=..., **stage_kwargs)
    .generate(prompt, OmniDiffusionSamplingParams(...)) -> list[OmniRequestOutput].
  OmniDiffusionSamplingParams(height,width,num_inference_steps,guidance_scale,
  true_cfg_scale,seed,num_outputs_per_prompt).

Example (FLUX is bf16-native; fp16/Half errors with a dtype mismatch — use bf16):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \\
  .venvs/vllm-omni/bin/python vrl/scripts/perf/vllm_omni_diffusion_profile.py \\
      --model black-forest-labs/FLUX.1-dev --precision bf16 \\
      --height 256 --width 256 --steps 4 --out outputs/perf/vllm_omni_flux_results.json

NOTE on memory: the model runs in a separate vLLM worker process, so this script's
``torch.cuda.max_memory_allocated()`` reports ~0 — the real peak must be read from
vLLM's own logs or by sampling ``nvidia-smi`` during the run.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from vrl.scripts.perf.common.diffusion_benchmark import DIFFUSION_BENCHMARK_PROMPT

# vLLM-Omni dtype tokens. fp8 is requested via quantization_config, not dtype, so
# its storage dtype stays bf16 (mirrors the naive profiler's fp8 = bf16 master + swap).
_DTYPE = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "fp8": "bfloat16"}


def _build_omni(model: str, precision: str, offload: bool):
    from vllm_omni.entrypoints.omni import Omni

    kwargs = {"dtype": _DTYPE[precision]}
    if offload:
        # FLUX (12B) / Qwen-Image (20B) full pipelines exceed 32 GB if all-on-GPU
        # (vanilla vLLM-Omni OOMs at orchestrator init). enable_cpu_offload is
        # vLLM-Omni's own offload knob (OmniDiffusionConfig) — the fair counterpart
        # to the naive path's CPU-encoder offload.
        kwargs["enable_cpu_offload"] = True
    if precision == "fp8":
        # fp8 weight-only quantization through the diffusion stage config; the exact
        # token may differ across vllm-omni builds — record the failure honestly if so.
        kwargs["quantization"] = "fp8"
    return Omni(model=model, **kwargs)


def _sampling_params(height: int, width: int, steps: int, guidance: float, is_qwen: bool):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    kwargs = dict(
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=0,
        num_outputs_per_prompt=1,
    )
    # Qwen-Image uses true_cfg_scale; FLUX.1-dev uses the distilled guidance_scale.
    if is_qwen:
        kwargs["true_cfg_scale"] = guidance
    else:
        kwargs["guidance_scale"] = guidance
    return OmniDiffusionSamplingParams(**kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--precision", default="bf16", choices=list(_DTYPE))
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--out", default="results_vllm_omni.json")
    p.add_argument(
        "--no-offload",
        dest="offload",
        action="store_false",
        help="disable vLLM-Omni CPU offload (will OOM the big models on 32GB)",
    )
    p.set_defaults(offload=True)
    args = p.parse_args(argv)

    is_qwen = "qwen" in args.model.lower()
    record: dict = {
        "engine": "vllm-omni",
        "model": args.model,
        "precision": args.precision,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "cpu_offload": args.offload,
        "status": "pending",
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        omni = _build_omni(args.model, args.precision, args.offload)
        params = _sampling_params(
            args.height,
            args.width,
            args.steps,
            args.guidance,
            is_qwen,
        )
        for _ in range(args.warmup):
            omni.generate(DIFFUSION_BENCHMARK_PROMPT, params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        last_metrics = None
        for _ in range(args.iters):
            t0 = time.perf_counter()
            outputs = omni.generate(DIFFUSION_BENCHMARK_PROMPT, params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            last_metrics = getattr(outputs[0], "metrics", None)

        latencies.sort()
        record.update(
            status="ok",
            forward_ms_median=latencies[len(latencies) // 2],
            forward_ms_min=latencies[0],
            throughput_img_per_s=1000.0 / latencies[len(latencies) // 2],
            peak_mem_mib=(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
            engine_metrics=_jsonable(last_metrics),
        )
        print(
            f"[ok] {args.model} {args.precision}: "
            f"{record['forward_ms_median']:.1f} ms/img, "
            f"peak {record['peak_mem_mib']:.0f} MiB"
        )
    except Exception as exc:
        peak = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else None
        )
        record.update(
            status="blocked",
            error_type=type(exc).__name__,
            error=str(exc)[:600],
            peak_mem_mib_at_failure=peak,
            traceback=traceback.format_exc()[-2000:],
        )
        print(f"[blocked] {args.model} {args.precision}: {type(exc).__name__}: {exc}")

    out = Path(args.out)
    existing = []
    if out.exists():
        existing = json.loads(out.read_text())
    existing.append(record)
    out.write_text(json.dumps(existing, indent=2))
    print(f"-> appended to {out}")


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {k: float(v) for k, v in dict(obj).items()} if obj else None


if __name__ == "__main__":
    main()
