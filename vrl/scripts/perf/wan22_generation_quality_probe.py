"""Wan 2.2 T2V generation-setting probe: is the 320x320x17 / 10-step geometry usable?

Fixed-prompt base videos at the accepted lifecycle geometry show a corrupted
first frame (blocky, wrong content) that clears by frame ~8. Before spending
GPU-hours on GRPO at that geometry this probe generates the same prompt/seed
under controlled variants and saves BOTH the mp4 (what the reward scores) and
lossless PNG frames (what the model produced), so codec blockiness and model
artifacts can be told apart, and the first-frame defect can be attributed to
step count, sampler, VAE tiling, resolution or the missing negative prompt.

    python -m vrl.scripts.perf.wan22_generation_quality_probe \
        --run-dir <training run with resolved_config.yaml> --output-dir <dir> \
        --prompt "..." --prompt "..." --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.types import DenoiseRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.registry import get_model_family_entry
from vrl.scripts.eval._device import resolve_eval_device, resolve_eval_dtype
from vrl.scripts.eval._sampling import resolve_eval_sampling
from vrl.scripts.eval.denoise_generation import _denoise_native, video_to_cthw
from vrl.trainers.checkpointing import load_resolved_run_config
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.json_files import write_json
from vrl.utils.media import video_tensor_to_uint8_frames, write_mp4

logger = logging.getLogger(__name__)

# The negative prompt every official Wan example passes (WanPipeline docs).
WAN_NEGATIVE_PROMPT = (  # noqa: RUF001
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

VARIANTS: dict[str, dict[str, Any]] = {
    "a_base_10step_ode": {},
    "b_sde_cps_10step": {"denoise_mode": "sde"},
    "c_20step_ode": {"num_steps": 20},
    "d_40step_ode": {"num_steps": 40},
    "e_no_vae_tiling_10step": {"vae_tiling": False},
    "f_480p_10step": {"width": 480, "height": 480},
    "g_negative_prompt_10step": {"negative_prompt": WAN_NEGATIVE_PROMPT},
    "h_negative_prompt_20step": {"negative_prompt": WAN_NEGATIVE_PROMPT, "num_steps": 20},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", type=int, default=2_026_091_400)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated subset.")
    return parser


def _generate(
    model: Any,
    *,
    prompt: str,
    negative_prompt: str | None,
    seed: int,
    sampling: dict[str, Any],
) -> torch.Tensor:
    encoded = model.encode_prompt(
        prompt,
        negative_prompt,
        max_sequence_length=int(sampling["max_sequence_length"]),
        guidance_scale=float(sampling["guidance_scale"]),
    )
    request = DenoiseRequest(
        negative_prompt=negative_prompt or "",
        width=int(sampling["width"]),
        height=int(sampling["height"]),
        frame_count=int(sampling["num_frames"]),
        num_steps=int(sampling["num_steps"]),
        guidance_scale=float(sampling["guidance_scale"]),
        seed=int(seed),
        fps=int(sampling["fps"]),
    )
    state = model.prepare_sampling(request, encoded)
    generator = torch.Generator(device=state.latents.device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        if str(sampling["denoise_mode"]) == "native":
            _denoise_native(model, state)
        else:
            for step_idx, timestep in enumerate(state.timesteps):
                step_output = model.forward_step(state, step_idx)
                state.latents = sde_step_with_logprob(
                    state.scheduler,
                    step_output["noise_pred"].float(),
                    timestep.unsqueeze(0),
                    state.latents.float(),
                    generator=generator,
                    deterministic=False,
                    return_dt=False,
                    noise_level=float(sampling["noise_level"]),
                    sde_type=str(sampling["sde_type"]),
                    step_index=step_idx,
                ).prev_sample
        decoded = model.decode_latents(state.latents)
    return video_to_cthw(decoded.detach().cpu())


def _frame_stats(frames: np.ndarray) -> dict[str, Any]:
    diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean(
        axis=(1, 2, 3)
    )
    # Blockiness: energy of 8-pixel-period edges vs. all edges (VAE latent = 8x8 px).
    gray = frames.astype(np.float32).mean(axis=-1)
    dx = np.abs(np.diff(gray, axis=2))
    grid = dx[:, :, 7::8].mean(axis=(1, 2))
    total = dx.mean(axis=(1, 2)) + 1e-6
    return {
        "frame_std": [round(float(f.std()), 2) for f in frames],
        "consecutive_abs_diff": [round(float(d), 2) for d in diffs],
        "blockiness_8px_ratio": [round(float(g / t), 3) for g, t in zip(grid, total, strict=True)],
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {sorted(VARIANTS)}")

    cfg, _ = load_resolved_run_config(
        args.run_dir,
        overrides=[
            "model.lora.path=",
            "model.torch_compile.enable=false",
            f"sampling.fps={int(args.fps)}",
        ],
    )
    root = parse_config(cfg)
    device = resolve_eval_device(args.device)
    precision = PrecisionPolicy.from_section(root.precision)
    dtype = resolve_eval_dtype(
        "auto", root, precision=precision, device=device, requires_trainer="Wan 2.2 quality probe"
    )
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(
        root, device, precision=precision, parameter_dtype_override=dtype
    )
    base_sampling = resolve_eval_sampling(root)
    base_sampling.setdefault("denoise_mode", "native")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = entry.build_rollout(build)
    model = bundle.model.eval()
    vae = model.pipeline.vae
    results: list[dict[str, Any]] = []
    try:
        with model.disable_adapter():
            for name in selected:
                variant = dict(VARIANTS[name])
                vae_tiling = variant.pop("vae_tiling", True)
                negative_prompt = variant.pop("negative_prompt", None)
                sampling = {**base_sampling, **variant}
                if vae_tiling:
                    vae.enable_tiling()
                    vae.enable_slicing()
                else:
                    vae.disable_tiling()
                    vae.disable_slicing()
                for prompt_index, prompt in enumerate(args.prompt):
                    started = time.perf_counter()
                    video = _generate(
                        model,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=args.seed + prompt_index,
                        sampling=sampling,
                    )
                    wall = time.perf_counter() - started
                    frames = video_tensor_to_uint8_frames(video)
                    stem = args.output_dir / f"{name}_p{prompt_index}"
                    write_mp4(video, stem.with_suffix(".mp4"), fps=float(sampling["fps"]))
                    for index in (0, 1, 2, 8, 16):
                        if index < len(frames):
                            Image.fromarray(frames[index]).save(f"{stem}_f{index:02d}.png")
                    row = {
                        "variant": name,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "settings": {
                            k: sampling[k]
                            for k in ("width", "height", "num_frames", "num_steps", "denoise_mode")
                        },
                        "vae_tiling": vae_tiling,
                        "negative_prompt": bool(negative_prompt),
                        "wall_s": round(wall, 1),
                        "finite": bool(torch.isfinite(video).all()),
                        "raw": _frame_stats(frames),
                    }
                    results.append(row)
                    logger.info("%s", json.dumps(row))
                    write_json(args.output_dir / "results.json", results)
    finally:
        del model, bundle
        release_cuda_memory()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
