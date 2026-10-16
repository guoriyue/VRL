"""Decompose cached layered scenes with one adapter and score every known-layer axis.

Inputs are ``cache.pt`` samples from ``vrl.scripts.denoise.encode_layered_scenes``
(rendered scenes, RORD samples or real photos). The model is the family build
of ``--experiment``; the arm is ``--adapter`` (a PEFT directory, or a training
checkpoint directory whose ``lora_weights/`` is taken) or ``--base`` (the
original Qwen-Image-Layered without any adapter). The layer budget is each
sample's cached layer count, so every arm generates the same number of layers,
with the layered release's sampler: Heun over ``--steps`` flow-matching sigmas,
CFG 1, fp32 states.

Writes per-sample layer PNGs (reused on rerun, so an interrupted arm resumes and a
new scorer can rescore stored layers) and ``scores.json``:

- scenes with truth: the ``layered_metrics`` axes (inferable and unknowable
  hidden regions scored separately), plus ``background_recovery`` on RORD
  samples that carry a ``foreground_mask.png``;
- ``--no-truth`` (real photos): layer count, recomposition coverage and fidelity,
  and the holes left after hiding each object layer.

The same adapter and seed give different per-scene samples on different GPU
models (bf16 kernel differences grow through sampling), so compare arms evaluated
on one machine. On a 32 GB card add ``precision.rollout.quantization.format=fp8``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vrl.scripts.eval.layered_metrics import SCORE_KEYS, composite, score_layers

logger = logging.getLogger(__name__)

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


def truth_free_metrics(stack, source):
    """Layer count, recomposition coverage and fidelity, and holes left after hiding a layer.

    ``stack`` is ``[L, H, W, 4]`` RGBA in ``[0, 1]``, back to front; ``source`` is the
    ``[H, W, 3]`` photo. A layer counts when more than 0.3% of its pixels are opaque.
    """
    pixels = stack.shape[1] * stack.shape[2]
    nonempty = [i for i, layer in enumerate(stack) if (layer[..., 3] > 0.5).sum() > 0.003 * pixels]
    recomposed = composite(stack)
    rgb_on_white = recomposed[..., :3] + (1 - recomposed[..., 3:])
    holes = []
    for i in nonempty[1:]:  # keep the back-most layer: hiding the background is not an edit
        kept = composite(np.delete(stack, i, axis=0))
        region = stack[i, ..., 3] > 0.5
        holes.append(float((kept[..., 3][region] < 0.99).mean()))
    return {
        "nonempty_layers": float(len(nonempty)),
        "composite_coverage": float((recomposed[..., 3] >= 0.99).mean()),
        "reconstruction_quality": float(1 - np.abs(rgb_on_white - source).mean()),
        "hide_without_holes": float(1 - np.mean(holes)) if holes else 1.0,
    }


def heun_decompose(model: Any, state: Any) -> Any:
    """The layered release's sampler: Heun over the state's sigmas, Euler on the last step.

    ``state`` comes from ``prepare_sampling`` (fp32 initial noise, the shifted
    sigma grid); the transformer runs at its own dtype through ``forward_step``.
    Returns the final packed latents.
    """
    import torch

    sigmas = state.scheduler.sigmas
    steps = len(state.timesteps)
    with torch.no_grad():
        for index in range(steps):
            dt = sigmas[index + 1] - sigmas[index]
            velocity = model.forward_step(state, index)["noise_pred"].float()
            midpoint = state.latents + dt * velocity
            if index + 1 < steps:
                corrected = model.forward_step(replace(state, latents=midpoint), index + 1)
                state.latents = (
                    state.latents + dt * (velocity + corrected["noise_pred"].float()) / 2
                )
            else:
                state.latents = midpoint
    return state.latents


def decompose(model: Any, cache: dict[str, Any], *, steps: int, seed: int) -> list[Image.Image]:
    """Decompose one cached sample into its cached layer budget of RGBA PIL layers."""
    import torch

    from vrl.generation.types import DenoiseRequest
    from vrl.utils.media import to_pil_image

    device = model.device
    encoded = {
        "prompt_embeds": cache["prompt_embeds"].to(device),
        "prompt_embeds_mask": (
            None if cache["prompt_embeds_mask"] is None else cache["prompt_embeds_mask"].to(device)
        ),
        "condition_latents": cache["condition_latents"].to(device),
    }
    request = DenoiseRequest(
        width=int(cache["width"]),
        height=int(cache["height"]),
        frame_count=1,
        num_steps=steps,
        guidance_scale=1.0,
        seed=seed,
    )
    state = model.prepare_sampling(request, encoded, layers=int(cache["layers"]))
    latents = heun_decompose(model, state)
    with torch.no_grad():
        decoded = model.decode_latents(latents)  # [1, layers, 4, H, W] in [0, 1]
    return [to_pil_image(layer, preserve_alpha=True) for layer in decoded[0].cpu()]


def _score(stack, scene):
    truth = np.stack(
        [
            np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
            for path in sorted(scene.glob("layer_*.png"))
        ]
    )
    masks = {
        name: np.asarray(Image.open(scene / f"{name}_mask.png").convert("L")) > 127
        if (scene / f"{name}_mask.png").exists()
        else None
        for name in ("inferable", "unknowable")
    }
    scores = score_layers(truth, stack, **masks)["axes"]
    if (scene / "foreground_mask.png").exists():
        scores.update(background_recovery(stack, scene))
    return scores


def adapter_overrides(adapter: Path | None, *, base: bool) -> list[str]:
    """The ``model`` overrides that select the arm: a PEFT directory or no adapter."""
    from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME

    if base == (adapter is not None):
        raise ValueError("pass exactly one of --adapter or --base")
    if base:
        return ["model.use_lora=false"]
    path = adapter / LORA_WEIGHTS_NAME if (adapter / LORA_WEIGHTS_NAME).is_dir() else adapter
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"{path} is not a PEFT adapter directory")
    return ["model.use_lora=true", f"model.lora.path={path}"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="experiment config name")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="PEFT directory or training checkpoint dir")
    parser.add_argument("--base", action="store_true", help="the checkpoint without any adapter")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4401, 4402])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--no-truth", action="store_true", help="real photos: truth-free metrics only"
    )
    parser.add_argument("overrides", nargs="*", help="preset overlays and dotlist overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    overrides = [*adapter_overrides(args.adapter, base=args.base), *args.overrides]

    import torch

    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry
    from vrl.trainers.data.layered_scenes import LAYERED_SCENE_CACHE, load_layered_scene

    scenes = sorted((args.dataset / args.split).glob(f"*/{LAYERED_SCENE_CACHE}"))[
        : args.max_scenes
    ]
    if not scenes:
        raise FileNotFoundError(f"no cached scenes under {args.dataset / args.split}")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(f"experiment/{args.experiment}", overrides=overrides)
    root = parse_config(cfg)
    entry = get_model_family_entry(str(root.model.family))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build = entry.resolve_model_build(
        root, device, precision=PrecisionPolicy.from_section(root.precision)
    )
    model = entry.build_rollout(build, expected_identity=None).model.eval()
    revision = None if root.model.revision is None else str(root.model.revision)

    rows = []
    started = time.perf_counter()
    for cache_path in scenes:
        scene = cache_path.parent
        cache = load_layered_scene(cache_path, family=entry.family, model_revision=revision)
        layer_count = int(cache["layers"])
        for seed in args.seeds:
            out = args.output / scene.name / f"seed{seed}"
            if (out / "layer_0.png").exists():
                layers = [
                    Image.open(out / f"layer_{i}.png").convert("RGBA") for i in range(layer_count)
                ]
            else:
                out.mkdir(parents=True, exist_ok=True)
                layers = decompose(model, cache, steps=args.steps, seed=seed)
                for i, layer in enumerate(layers):
                    layer.save(out / f"layer_{i}.png")
            stack = np.stack(
                [np.asarray(layer, dtype=np.float32) / 255.0 for layer in layers]
            )  # [L, H, W, 4]
            if args.no_truth:
                source = np.asarray(
                    Image.open(scene / "source.png").convert("RGB"), dtype=np.float32
                )
                scores = truth_free_metrics(stack, source / 255.0)
            else:
                scores = _score(stack, scene)
            rows.append({"scene": scene.name, "seed": seed, "layers": layer_count, **scores})
            logger.info(
                "%s", json.dumps({**rows[-1], "elapsed": round(time.perf_counter() - started, 1)})
            )
            (args.output / "scores.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    keys = (
        TRUTH_FREE_KEYS
        if args.no_truth
        else SCORE_KEYS
        + (("background_recovery",) if rows and "background_recovery" in rows[0] else ())
    )
    means = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    summary = {
        "arm": "base" if args.base else str(args.adapter),
        "experiment": args.experiment,
        "overrides": overrides,
        "split": args.split,
        "scenes": len(scenes),
        "seeds": args.seeds,
        "steps": args.steps,
        "sampler": "Heun; fp32 states; CFG1",
        "means": means,
        "rows": rows,
        "scope": "Native inference on cached scenes; hand-set known-layer metrics "
        "(layered_metrics); not real-photo evidence.",
    }
    (args.output / "scores.json").write_text(json.dumps(summary, indent=1) + "\n")
    logger.info(
        "%s",
        json.dumps({"arm": summary["arm"], "means": {k: round(v, 4) for k, v in means.items()}}),
    )


if __name__ == "__main__":
    main()
