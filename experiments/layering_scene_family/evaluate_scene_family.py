"""Native 50-step Heun decomposition of cached scenes, scored on every known-layer axis.

Arms: ``--adapter`` (a trained LoRA or the official adapter) or ``--zero-b``
(the official adapter with every lora_B zeroed, i.e. the original
Qwen-Image-Layered base). The layer budget is each scene's truth layer count,
so every arm generates the same number of layers. Writes per-scene layer PNGs
and ``scores.json`` with the layered_reference axes (inferable and unknowable
hidden regions scored separately) plus split means. Run with the layered_rl
interpreter; takes the GPU lock (non-blocking).
"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.layered_reference import SCORE_KEYS, LayeredReferenceRewardModel
from vrl.scripts.generation.qwen_layered_rl_probe import (
    load_transformer,
    load_upstream,
    predict_velocity,
)

TRUTH_FREE_KEYS = (
    "nonempty_layers",
    "composite_coverage",
    "reconstruction_quality",
    "hide_without_holes",
)


def background_recovery(stack, scene):
    """RORD: how much of the true background the back-most layer recovers under the foreground.

    1 - err(layer0, truth) / err(photo, truth), inside the foreground polygon:
    1 means the background was recovered exactly, 0 means no better than
    leaving the foreground in place, negative means worse.
    """
    truth = np.asarray(Image.open(scene / "layer_0.png").convert("RGB"), dtype=np.float32) / 255.0
    photo = np.asarray(Image.open(scene / "source.png").convert("RGB"), dtype=np.float32) / 255.0
    region = np.asarray(Image.open(scene / "foreground_mask.png").convert("L")) > 127
    layer = stack[0]
    rgb = layer[..., :3] * layer[..., 3:4] + (1 - layer[..., 3:4])
    e_pred = float(np.abs(rgb - truth)[region].mean())
    e_photo = float(np.abs(photo - truth)[region].mean())
    return {
        "background_recovery": 1.0 - e_pred / max(e_photo, 1e-6),
        "background_error": e_pred,
        "photo_error": e_photo,
    }


def composite_alpha(stack):
    alpha = np.zeros(stack.shape[1:3], dtype=np.float32)
    for layer in stack:
        alpha = layer[..., 3] + alpha * (1 - layer[..., 3])
    return alpha


def truth_free_metrics(stack, source):
    """Layer count, recomposition coverage and fidelity, and holes left after hiding each object layer."""
    pixels = stack.shape[1] * stack.shape[2]
    nonempty = [i for i, layer in enumerate(stack) if (layer[..., 3] > 0.5).sum() > 0.003 * pixels]
    rgb = np.ones((*stack.shape[1:3], 3), dtype=np.float32)
    for layer in stack:
        a = layer[..., 3:4]
        rgb = layer[..., :3] * a + rgb * (1 - a)
    holes = []
    for i in nonempty[1:]:  # keep the back-most layer: hiding the background is not an edit
        kept = np.delete(stack, i, axis=0)
        region = stack[i, ..., 3] > 0.5
        holes.append(float((composite_alpha(kept)[region] < 0.99).mean()))
    return {
        "nonempty_layers": float(len(nonempty)),
        "composite_coverage": float((composite_alpha(stack) >= 0.99).mean()),
        "reconstruction_quality": float(1 - np.abs(rgb - source).mean()),
        "hide_without_holes": float(1 - np.mean(holes)) if holes else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--zero-b", action="store_true", help="original base: official adapter with lora_B zeroed"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[4401, 4402])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--no-truth", action="store_true", help="real photos: truth-free metrics only"
    )
    args = parser.parse_args()
    assert args.adapter is not None
    plan = json.loads(args.plan.read_text())
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    scenes = sorted((args.dataset / args.split).glob("*/cache.pt"))[: args.max_scenes]
    assert scenes
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = LayeredReferenceRewardModel({})
    rows = []
    with Path(os.environ.get("VRL_GPU_LOCK", "/tmp/vrl-preservation-gpu0.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler

        upstream = load_upstream(upstream_path)
        model = load_transformer(plan)
        saved = torch.load(args.adapter, weights_only=True, map_location="cpu")
        parameters = {
            name: value for name, value in model.named_parameters() if value.requires_grad
        }
        assert saved.keys() == parameters.keys()
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(
                    torch.zeros_like(saved[name])
                    if args.zero_b and "lora_B" in name
                    else saved[name]
                )
        model.requires_grad_(False).eval()
        weight_sha = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
        vae = (
            AutoencoderKLQwenImage.from_pretrained(
                plan["base_path"],
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .eval()
            .cuda()
        )
        vae.requires_grad_(False)
        vae.enable_slicing()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            plan["base_path"], subfolder="scheduler", local_files_only=True
        )
        started = time.perf_counter()
        for cache_path in scenes:
            scene = cache_path.parent
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            condition = {
                k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                for k, v in cache.items()
                if k in ("condition", "prompt", "mask", "layers", "width", "height")
            }
            layer_count = int(cache["layers"])
            scheduler.set_timesteps(
                sigmas=np.linspace(1, 0, args.steps + 1)[:-1],
                device="cuda",
                mu=(condition["condition"].shape[1] / 256) ** 0.5,
            )
            for seed in args.seeds:
                out = args.output / scene.name / f"seed{seed}"
                if (out / "layer_0.png").exists():
                    layers = [
                        Image.open(out / f"layer_{i}.png").convert("RGBA")
                        for i in range(layer_count)
                    ]
                else:
                    out.mkdir(parents=True, exist_ok=True)
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    latent = torch.randn(
                        1,
                        (layer_count + 1) * condition["condition"].shape[1],
                        model.config.in_channels,
                        generator=generator,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    with torch.no_grad():
                        for index, timestep in enumerate(scheduler.timesteps):
                            dt = scheduler.sigmas[index + 1] - scheduler.sigmas[index]
                            velocity = predict_velocity(
                                model, latent, timestep, condition, upstream
                            )
                            midpoint = latent + dt * velocity
                            if scheduler.sigmas[index + 1] > 0:
                                corrected = predict_velocity(
                                    model,
                                    midpoint,
                                    scheduler.sigmas[index + 1] * 1000,
                                    condition,
                                    upstream,
                                )
                                latent = latent + dt * (velocity + corrected) / 2
                            else:
                                latent = midpoint
                        decoded = upstream.decode_layers(
                            vae, latent, condition["height"], condition["width"], layer_count
                        )
                    layers = [
                        upstream.layer_to_pil_rgba(decoded[0, i]) for i in range(layer_count)
                    ]
                    for i, layer in enumerate(layers):
                        layer.save(out / f"layer_{i}.png")
                stack = np.stack(
                    [np.asarray(layer, dtype=np.float32) / 255.0 for layer in layers]
                )  # [L, H, W, 4]
                if args.no_truth:
                    source = (
                        np.asarray(
                            Image.open(scene / "source.png").convert("RGB"), dtype=np.float32
                        )
                        / 255.0
                    )
                    rows.append(
                        {
                            "scene": scene.name,
                            "seed": seed,
                            "layers": layer_count,
                            **truth_free_metrics(stack, source),
                        }
                    )
                    print(
                        json.dumps(
                            {**rows[-1], "elapsed": round(time.perf_counter() - started, 1)}
                        ),
                        flush=True,
                    )
                    (args.output / "scores.json").write_text(
                        json.dumps({"rows": rows}, indent=1) + "\n"
                    )
                    continue
                media = torch.from_numpy(stack).permute(0, 3, 1, 2)
                axes = scorer(
                    RewardInferenceArtifact(
                        "x",
                        "x",
                        "",
                        media=media,
                        metadata={
                            "layered_reference": {
                                "layers": str(scene),
                                **{
                                    k: str(scene / f"{k}.png")
                                    for k in ("inferable_mask", "unknowable_mask")
                                    if (scene / f"{k}.png").exists()
                                },
                            }
                        },
                    )
                )
                extra = {}
                if (scene / "foreground_mask.png").exists():
                    extra = background_recovery(stack, scene)
                rows.append(
                    {
                        "scene": scene.name,
                        "seed": seed,
                        "layers": layer_count,
                        **{k: float(v) for k, v in axes.items()},
                        **extra,
                    }
                )
                print(
                    json.dumps({**rows[-1], "elapsed": round(time.perf_counter() - started, 1)}),
                    flush=True,
                )
                (args.output / "scores.json").write_text(
                    json.dumps({"rows": rows}, indent=1) + "\n"
                )
    keys = (
        TRUTH_FREE_KEYS
        if args.no_truth
        else SCORE_KEYS
        + (("background_recovery",) if rows and "background_recovery" in rows[0] else ())
    )
    means = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    summary = {
        "arm": "zero_b" if args.zero_b else str(args.adapter),
        "weight_sha256": weight_sha,
        "zero_b": args.zero_b,
        "split": args.split,
        "scenes": len(scenes),
        "seeds": args.seeds,
        "steps": args.steps,
        "sampler": "Heun; fp32 states; bf16 blocks streamed from host; CFG1",
        "means": means,
        "rows": rows,
        "scope": "Native inference on cached scenes; axes from the layered_reference reward model (not qualified for a training key); not real-photo evidence.",
    }
    (args.output / "scores.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        json.dumps({"arm": summary["arm"], "means": {k: round(v, 4) for k, v in means.items()}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
