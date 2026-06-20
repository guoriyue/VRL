"""SD3.5 vLLM-omni denoise forward probe — the vLLM-omni half of the compare.

STANDALONE (no vrl imports): runs ONLY in the isolated `.venv-vllm-omni`
(vllm 0.22 + vllm-omni 0.22). Loads sd3.5 through vLLM-omni's own
`StableDiffusion3Pipeline` and times a denoise forward, dumping the SAME JSON
shape as `sd3_forward_probe.py` (native) so `compare` can diff throughput.

Usage (in the venv):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
      .venv-vllm-omni/bin/python vrl/scripts/perf/sd3_forward_probe_vllm_omni.py \
      --cache none --out outputs/perf/sd3_vllm_omni.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

_MODEL = "stabilityai/stable-diffusion-3.5-medium"
_PROMPT = "a photo of a red apple on a wooden table, high quality"
_SEED = 0
_W = _H = 512
_STEPS = 10
_GUIDANCE = 4.5
_MAX_SEQ = 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="none", help="vllm-omni cache_backend: none | tea_cache")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="outputs/perf/sd3_vllm_omni.json")
    args = p.parse_args()

    # Single-GPU worker bootstrap: DiffusionWorker.__init__ sets up the whole vLLM
    # engine context (device, distributed TP=1, VllmConfig, forward_context, weight
    # load) that the SD3 pipeline's vLLM CustomOps/TP layers require. Standalone
    # pipeline instantiation can't satisfy these — vllm-omni diffusion IS an engine.
    for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29577",
                 "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        os.environ.setdefault(k, v)

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import (
        OmniDiffusionRequest,
        OmniDiffusionSamplingParams,
    )
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    od = OmniDiffusionConfig(
        model=_MODEL,
        dtype=torch.bfloat16,
        output_type="latent",
        cache_backend=args.cache,
    )
    od.enrich_config()  # populate tf_model_config (num_layers etc.) from transformer/config.json
    print(f"building vllm-omni SD3 worker (cache={args.cache}) ...", flush=True)
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=od)

    # --- precision introspection (is vllm-omni running the SAME precision as native bf16?) ---
    tf = worker.model_runner.pipeline.transformer
    dtypes = {}
    cls_names = set()
    for n, p in tf.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    for m in tf.modules():
        cn = type(m).__name__
        if any(k in cn.lower() for k in ("fp8", "quant", "int8", "linear")):
            cls_names.add(cn)
    print(f"[precision] od.dtype={od.dtype} quantization_config={getattr(od, 'quantization_config', None)}",
          flush=True)
    print(f"[precision] transformer param dtypes={dtypes}", flush=True)
    print(f"[precision] linear/quant module classes={sorted(cls_names)}", flush=True)

    def build_req():
        sp = OmniDiffusionSamplingParams(
            height=_H, width=_W, num_inference_steps=_STEPS,
            guidance_scale=_GUIDANCE, max_sequence_length=_MAX_SEQ, seed=_SEED,
        )
        return OmniDiffusionRequest(request_id="probe-0", prompts=[_PROMPT], sampling_params=sp)

    def run_once():
        with torch.no_grad():
            return worker.execute_model(build_req(), od)

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    t0 = time.time()
    out = run_once()
    torch.cuda.synchronize()
    wall = time.time() - t0

    # Pull the output latent for a coarse consistency stat.
    lat = getattr(out, "latents", None)
    if lat is None:
        lat = getattr(out, "images", None)
    fl = {}
    if isinstance(lat, torch.Tensor):
        f = lat.detach().float()
        fl = {"mean": float(f.mean()), "absmean": float(f.abs().mean()),
              "l2": float(f.norm()), "shape": list(f.shape)}

    result = {
        "backend": f"vllm_omni(cache={args.cache})",
        "model": _MODEL, "prompt": _PROMPT, "seed": _SEED, "num_steps": _STEPS,
        "shape": [_W, _H], "wall_s": wall, "s_per_step": wall / _STEPS,
        "final_latent": fl,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"vllm-omni sd3.5 (cache={args.cache}): {_STEPS} steps, {wall:.2f}s "
          f"({wall/_STEPS:.3f}s/step) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
