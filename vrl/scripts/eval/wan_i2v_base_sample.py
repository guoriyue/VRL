"""Base (untrained) image-to-video samples from the local Wan 2.2 I2V A14B model.

Why this exists: the repo's wan_2_1 family is text-to-video only; there is no
I2V runtime wired into the engine yet. This is a standalone eval probe that
drives diffusers' ``WanImageToVideoPipeline`` directly against the locally
cached ``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` checkpoint, so we can eyeball base
I2V quality before investing in a real runtime.

Conditioning frames: by default we take the *first frame* of the existing T2V
baseline clips under a directory you provide via --baseline-dir and reuse the same 5
physics prompts. That needs zero new input assets and makes the I2V output
directly comparable to the T2V baseline.

Memory: A14B is a Mixture-of-Experts video model with two ~14B transformers
(high-noise ``transformer`` + low-noise ``transformer_2``). Neither fits twice
on a 32 GB card, so we always CPU-offload. ``--offload model`` keeps the active
expert resident (fast, ~28 GB peak); on CUDA OOM we auto-fall back to
``sequential`` (per-submodule streaming — slow but fits in a few GB).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

DEFAULT_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Same 5 prompts as the T2V A/B probe so I2V output is comparable.
PROMPTS = [
    "A whisk spins in the egg mixture, mixing it thoroughly.",
    "A spoon scoops creamy soup from a pot.",
    "Water dripping from a faucet into a sink.",
    "Frog leaping from one lilypad to another.",
    "Whipped cream pouring onto hot chocolate.",
]

# Standard Wan negative prompt (quality + anatomy guardrails). The fullwidth
# commas are the model's expected Chinese punctuation — keep them verbatim.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"  # noqa: RUF001
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"  # noqa: RUF001
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"  # noqa: RUF001
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: RUF001
)


def _first_frame(video_path: Path) -> Image.Image:
    """Return frame 0 of an mp4 as a PIL image."""
    frame = iio.imread(video_path, index=0)
    return Image.fromarray(np.asarray(frame)).convert("RGB")


def _fit_resolution(
    image: Image.Image, pipe: WanImageToVideoPipeline, max_area: int
) -> Image.Image:
    """Resize keeping aspect so H*W ~= max_area and both are model-valid multiples.

    Mirrors the diffusers Wan I2V recipe: dims must be divisible by
    vae_scale_factor_spatial * transformer patch size.
    """
    patch = pipe.transformer.config.patch_size
    patch_w = patch[1] if isinstance(patch, (list, tuple)) else patch
    mod = pipe.vae_scale_factor_spatial * patch_w
    aspect = image.height / image.width
    height = max(mod, round((max_area * aspect) ** 0.5 / mod) * mod)
    width = max(mod, round((max_area / aspect) ** 0.5 / mod) * mod)
    return image.resize((width, height), Image.LANCZOS)


def _load_pipe(model_path: str, offload: str) -> WanImageToVideoPipeline:
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # VAE tiling/slicing trade latency for peak decode memory — always on here.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "none":
        pipe.to("cuda")
    else:
        raise ValueError(f"unknown offload mode: {offload!r}")
    return pipe


def _call_kwargs(pipe: WanImageToVideoPipeline, **base: object) -> dict[str, object]:
    """Keep only kwargs this pipeline both accepts AND actually supports.

    ``guidance_scale_2`` (the low-noise expert's guidance) appears in the call
    signature on both Wan 2.1 and 2.2, but 2.2's MoE pipeline rejects it at
    runtime unless ``boundary_ratio`` is set (i.e. only the dual-expert 2.2
    checkpoint honors it). Single-tower 2.1 has no ``boundary_ratio`` and raises
    if it is passed, so we drop the second-expert knob whenever the pipeline is
    not configured as a two-expert model.
    """
    import inspect

    allowed = set(inspect.signature(pipe.__call__).parameters)
    out = {k: v for k, v in base.items() if k in allowed}
    if getattr(pipe.config, "boundary_ratio", None) is None:
        out.pop("guidance_scale_2", None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="outputs/wan_i2v_a14b_base_samples")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/wan_phys_eval_AB/baseline",
        help="T2V baseline clips whose first frame seeds each I2V sample.",
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--guidance-2", type=float, default=3.0)
    parser.add_argument("--max-area", type=int, default=480 * 832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--offload", choices=["model", "sequential", "none"], default="model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir)

    # Pair each prompt with the first frame of its baseline clip.
    seeds: list[tuple[int, str, Path]] = []
    for i, prompt in enumerate(PROMPTS[: args.num_samples]):
        clip = baseline_dir / f"prompt{i:02d}_seed{i}.mp4"
        if not clip.exists():
            print(f"[skip] missing baseline clip {clip}", flush=True)
            continue
        seeds.append((i, prompt, clip))
    if not seeds:
        raise SystemExit(f"no baseline clips found under {baseline_dir}")

    print(f"[load] {args.model_path} (offload={args.offload})", flush=True)
    t0 = time.time()
    pipe = _load_pipe(args.model_path, args.offload)
    print(f"[load] done in {time.time() - t0:.0f}s", flush=True)

    manifest: list[dict[str, object]] = []
    offload = args.offload
    for idx, prompt, clip in seeds:
        image = _first_frame(clip)
        image = _fit_resolution(image, pipe, args.max_area)
        cond_path = out_dir / f"prompt{idx:02d}_input.png"
        image.save(cond_path)

        kwargs = _call_kwargs(
            pipe,
            image=image,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=image.height,
            width=image.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            guidance_scale_2=args.guidance_2,
            generator=torch.Generator(device="cpu").manual_seed(idx),
        )

        print(
            f"[gen] prompt{idx:02d} {image.width}x{image.height} "
            f"{args.num_frames}f/{args.steps}steps :: {prompt}",
            flush=True,
        )
        t1 = time.time()
        try:
            frames = pipe(**kwargs).frames[0]
        except torch.cuda.OutOfMemoryError:
            if offload != "sequential":
                print("[oom] retrying with sequential cpu offload (slow)", flush=True)
                del pipe
                torch.cuda.empty_cache()
                offload = "sequential"
                pipe = _load_pipe(args.model_path, offload)
                kwargs = _call_kwargs(pipe, **kwargs)  # re-filter for safety
                frames = pipe(**kwargs).frames[0]
            else:
                raise
        dt = time.time() - t1

        video_path = out_dir / f"prompt{idx:02d}_i2v.mp4"
        export_to_video(frames, str(video_path), fps=args.fps)
        rec = {
            "prompt_index": idx,
            "prompt": prompt,
            "input_image": str(cond_path),
            "input_clip": str(clip),
            "video": str(video_path),
            "height": image.height,
            "width": image.width,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "offload": offload,
            "seconds": round(dt, 1),
        }
        manifest.append(rec)
        print(f"[done] prompt{idx:02d} -> {video_path} in {dt:.0f}s", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {out_dir / 'manifest.json'} ({len(manifest)} clips)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
