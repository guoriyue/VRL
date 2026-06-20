"""SD3.5 native denoise forward probe — the baseline half of the vLLM-omni compare.

Runs our native sd3.5 rollout denoise (the `vrl` path) on a fixed prompt+seed and
dumps per-step noise_pred summaries + s/step to JSON. The vLLM-omni half
(`sd3_forward_probe_vllm_omni.py`, run in the isolated `.venv-vllm-omni`) produces
the SAME JSON shape; `compare` diffs them for throughput + noise_pred consistency.

WHY a separate vLLM-omni script + venv: vllm-omni 0.22 needs vllm 0.22 (CUDA 13),
which conflicts with the main env's vllm 0.21 (our fp8/paged). So the two backends
run in two envs and meet at the JSON. See SPRINT_vllm_omni_rollout_perf.md.

Usage (main env):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.perf.sd3_forward_probe \
        --config experiment/diffusion/sd3_5/online_grpo_geneval --out outputs/perf/sd3_native.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from vrl.config.loading import load_config
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.math.diffusion.flow_matching import sde_step_with_logprob

_PROMPT = "a photo of a red apple on a wooden table, high quality"
_SEED = 0
_SDE_TYPE = "sde"


def build_sd3(cfg, device, dtype):
    from vrl.models.diffusion.sd3_5.runtime import (
        build_sd3_5_runtime_bundle,
        extract_sd3_5_runtime_spec,
    )
    # Base weights (no LoRA) to match the vLLM-omni side (no active adapter) and to
    # allow torch.compile (PEFT-wrapped transformers reject the compiled-module swap).
    cfg.model.use_lora = False
    spec = extract_sd3_5_runtime_spec(cfg, device, dtype)
    return build_sd3_5_runtime_bundle(spec).model


def _noise_pred_summary(noise_pred: torch.Tensor) -> dict:
    f = noise_pred.detach().float()
    return {
        "mean": float(f.mean()),
        "absmean": float(f.abs().mean()),
        "std": float(f.std()),
        "l2": float(f.norm()),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer (fair compare vs vllm-omni's auto-compile)")
    p.add_argument("--out", default="outputs/perf/sd3_native.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    s = cfg.sampling
    model = build_sd3(cfg, device, dtype)
    if args.compile:
        model.torch_compile_transformer("default")

    enc = model.encode_prompt([_PROMPT], None, guidance_scale=float(s.guidance_scale),
                              max_sequence_length=int(s.max_sequence_length))
    req = VideoGenerationRequest(
        prompt=_PROMPT, negative_prompt=None, width=int(s.width), height=int(s.height),
        frame_count=int(getattr(s, "num_frames", 1) or 1), num_steps=int(s.num_steps),
        guidance_scale=float(s.guidance_scale), seed=_SEED,
        extra={"max_sequence_length": int(s.max_sequence_length)},
    )
    num_steps = int(s.num_steps)

    def run_once(record: bool):
        state = model.prepare_sampling(req, enc)
        gen = torch.Generator(device=device).manual_seed(_SEED)
        per_step = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            for i in range(num_steps):
                noise_pred = model.forward_step(state, i)["noise_pred"]
                if record:
                    per_step.append(_noise_pred_summary(noise_pred))
                r = sde_step_with_logprob(
                    state.scheduler, noise_pred.float(), state.timesteps[i].unsqueeze(0),
                    state.latents.float(), generator=gen, deterministic=False,
                    sde_type=_SDE_TYPE, step_index=i)
                state.latents = r.prev_sample
        return per_step, state

    for _ in range(args.warmup):
        run_once(record=False)
    torch.cuda.synchronize(device)

    t0 = time.time()
    per_step, state = run_once(record=True)
    torch.cuda.synchronize(device)
    wall = time.time() - t0

    final_latent = state.latents.detach().float()
    out = {
        "backend": "native",
        "config": args.config,
        "prompt": _PROMPT,
        "seed": _SEED,
        "num_steps": num_steps,
        "shape": [int(s.width), int(s.height)],
        "wall_s": wall,
        "s_per_step": wall / num_steps,
        "per_step_noise_pred": per_step,
        "final_latent": {
            "mean": float(final_latent.mean()),
            "absmean": float(final_latent.abs().mean()),
            "l2": float(final_latent.norm()),
            "shape": list(final_latent.shape),
        },
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"native sd3.5: {num_steps} steps, {wall:.2f}s ({wall/num_steps:.3f}s/step) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
