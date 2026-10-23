"""Decompose cached layered scenes with one adapter and score every known-layer axis.

Usage::

    python -m vrl.scripts.eval.layered_decompose_eval \
        --config experiment/qwen_image_layered/offline_sft_layered_scenes \
        data.manifest=/data/layering/dataset_v3 data.split=heldout \
        --adapter outputs/layered_sft/weighted/checkpoint-2000 --output outputs/layered_eval/weighted

``--config`` takes a bundled config name or an absolute YAML path (a run's
``resolved_config.yaml`` works), with positional vrl-train overrides on top.
Inputs are the ``cache.pt`` samples of ``data.split`` under each
``data.manifest`` root, from ``vrl.scripts.denoise.encode_layered_scenes``
(rendered scenes, RORD samples or real photos). The model is the config's
family build; the arm is ``--adapter`` (a PEFT directory, or a training
checkpoint directory whose ``lora_weights/`` is taken), or without it the
original Qwen-Image-Layered with no adapter. The layer budget is each sample's
cached layer count, so every arm generates the same number of layers, with the
layered release's sampler: Heun over ``sampling.num_steps`` flow-matching
sigmas, CFG 1, fp32 states.

Writes per-sample layer PNGs (reused on rerun, so an interrupted arm resumes and a
new scorer can rescore stored layers) and ``scores.json``:

- scenes with truth (``layer_*.png`` beside the cache): the known-layer axes
  below (inferable and unknowable hidden regions scored separately), plus
  ``background_recovery`` on RORD samples that carry a ``foreground_mask.png``;
- scenes without truth (real photos): layer count, recomposition coverage and
  fidelity, and the holes left after hiding each object layer.

Known-layer scoring compares two ``[N, H, W, 4]`` RGBA stacks in ``[0, 1]``,
back to front: the truth (a rendered scene whose hidden content is known;
layer 0 is the background) and the candidate. Truth layers are matched to
candidate layers injectively by alpha IoU times color agreement, then measured
on these axes, each 1 for the truth itself:

* ``layered_object_match``: product of the per-object match qualities, so a
  merged or missing object zeroes it;
* ``layered_visible_retention``: one minus the error on visible truth pixels;
* ``layered_hidden_completion``: one minus the error on truth pixels hidden
  behind objects in front;
* ``layered_edit_holes``: one minus the composite error after hiding or moving
  each object, truth versus candidate;
* ``layered_nonredundancy``: share of candidate alpha inside matched layers;
* ``layered_reconstruction``: one minus the recomposed-image error;
* ``layered_composite_coverage``: share of the truth's opaque pixels the
  candidate composite covers opaquely;
* ``layered_inferable_completion``: the hidden-pixel error restricted to an
  ``inferable`` mask (deducible by continuation or symmetry), against truth;
* ``layered_unknowable_plausibility``: on hidden pixels inside an
  ``unknowable`` mask no truth is consulted: the matched layer must cover them
  with colors near its own visible colors.

These are hand-set evaluation metrics (the matching rule, the 0.5 alpha cuts
and the color-gap slope were never checked against human judgment). They rank
arms on one dataset; they are not a training reward.

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

logger = logging.getLogger(__name__)

SCORE_KEYS = (
    "layered_object_match",
    "layered_visible_retention",
    "layered_hidden_completion",
    "layered_edit_holes",
    "layered_nonredundancy",
    "layered_reconstruction",
    "layered_composite_coverage",
    "layered_inferable_completion",
    "layered_unknowable_plausibility",
)


def premultiplied(rgba):
    return np.concatenate((rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]), axis=-1)


def composite(layers):
    result = np.zeros_like(layers[0])
    for layer in layers:
        result = premultiplied(layer) + result * (1 - layer[..., 3:])
    return result


def match_layers(similarities):
    """Injective truth-to-candidate assignment maximizing total similarity.

    A dynamic program over subsets of the (at most eight) truth layers; candidate
    count is linear. A truth layer may stay unmatched and surplus candidates may
    stay unused.
    """
    count, candidates = similarities.shape
    if not 1 <= count <= 8:
        raise ValueError("Known-layer scenes need one to eight truth layers")
    states = {0: (0.0, [None] * count)}
    for candidate in range(candidates):
        updated = dict(states)
        for mask, (score, assignment) in states.items():
            for reference in range(count):
                if mask & (1 << reference):
                    continue
                next_mask = mask | (1 << reference)
                next_score = score + similarities[reference, candidate]
                if next_mask not in updated or next_score > updated[next_mask][0]:
                    next_assignment = assignment.copy()
                    next_assignment[reference] = candidate
                    updated[next_mask] = (next_score, next_assignment)
        states = updated
    return max(states.values(), key=lambda item: item[0])[1]


def score_layers(reference, candidate, *, inferable=None, unknowable=None):
    """Match ``candidate`` to the truth ``reference`` and score every axis.

    ``inferable`` / ``unknowable`` are optional ``[H, W]`` masks of hidden
    pixels; without one its axis is 1. Returns ``assignment`` (candidate index
    per truth layer, or None), per-object ``objects`` rows and the ``axes``.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.ndim != 4
        or candidate.ndim != 4
        or reference.shape[1:] != candidate.shape[1:]
        or reference.shape[-1] != 4
        or not len(candidate)
    ):
        raise ValueError("Expected nonempty layer stacks with identical H,W,4 dimensions")
    for stack in (reference, candidate):
        if not np.isfinite(stack).all() or stack.min() < 0 or stack.max() > 1:
            raise ValueError("RGBA samples must be finite and normalized to [0,1]")
    if any(layer[..., 3].sum() == 0 for layer in reference):
        raise ValueError("Truth layers must contain visible or amodal object support")
    ref_features = premultiplied(reference)
    pred_features = premultiplied(candidate)
    similarities = np.zeros((len(reference), len(candidate)))
    for i, ref in enumerate(reference):
        for j, pred in enumerate(candidate):
            union = np.maximum(ref[..., 3], pred[..., 3])
            iou = np.minimum(ref[..., 3], pred[..., 3]).sum() / union.sum()
            error = np.abs(ref_features[i] - pred_features[j]).mean(axis=-1)
            similarities[i, j] = iou * (1 - (error * union).sum() / union.sum())
    assignment = match_layers(similarities)
    blank = np.zeros_like(reference[0])
    rows = []
    visible_errors, hidden_errors, edit_errors = [], [], []
    inferable_error = inferable_mass = plausibility = unknowable_mass = 0.0
    offset = max(1, reference.shape[2] // 8)
    for i, predicted_index in enumerate(assignment):
        predicted = blank if predicted_index is None else candidate[predicted_index]
        transmission = np.ones(reference.shape[1:3])
        for above in reference[i + 1 :]:
            transmission *= 1 - above[..., 3]
        hidden = reference[i, ..., 3] * (1 - transmission)
        visible = reference[i, ..., 3] * transmission
        residual = np.abs(ref_features[i] - premultiplied(predicted)).mean(axis=-1)
        row = {
            "reference_layer": i,
            "candidate_layer": predicted_index,
            "layer_quality": 0.0
            if predicted_index is None
            else float(similarities[i, predicted_index]),
            "visible_rgba_mae": None,
            "hidden_rgba_mae": None,
        }
        if visible.sum() > 0:
            row["visible_rgba_mae"] = float((residual * visible).sum() / visible.sum())
            visible_errors.append(row["visible_rgba_mae"])
        if hidden.sum() > 0:
            row["hidden_rgba_mae"] = float((residual * hidden).sum() / hidden.sum())
            hidden_errors.append(row["hidden_rgba_mae"])
        # The background (layer 0) is never hidden, moved or region-scored.
        if i > 0:
            for operation in ("hide", "move"):
                truth_layers = reference.copy()
                predicted_layers = candidate.copy()
                truth_layers[i] = blank
                if operation == "move":
                    truth_layers[i, :, offset:] = reference[i, :, :-offset]
                if predicted_index is not None:
                    predicted_layers[predicted_index] = blank
                    if operation == "move":
                        predicted_layers[predicted_index, :, offset:] = predicted[:, :-offset]
                region = np.maximum(reference[i, ..., 3], predicted[..., 3])
                if operation == "move":
                    shifted_support = np.maximum(
                        reference[i, :, :-offset, 3], predicted[:, :-offset, 3]
                    )
                    region[:, offset:] = np.maximum(region[:, offset:], shifted_support)
                difference = np.abs(composite(truth_layers) - composite(predicted_layers)).mean(
                    axis=-1
                )
                row[f"{operation}_rgba_mae"] = float((difference * region).sum() / region.sum())
                edit_errors.append(row[f"{operation}_rgba_mae"])
            if inferable is not None:
                weight = hidden * inferable
                inferable_error += float((residual * weight).sum())
                inferable_mass += float(weight.sum())
            if unknowable is not None:
                weight = hidden * unknowable
                mass = float(weight.sum())
                if mass > 0:
                    alpha = float((predicted[..., 3] * weight).sum() / mass)
                    visible_px = (visible > 0.5) & (predicted[..., 3] > 0.5)
                    region_px = (weight > 0.5) & (predicted[..., 3] > 0.5)
                    if visible_px.any() and region_px.any():
                        gap = float(
                            np.abs(
                                predicted[region_px][:, :3].mean(axis=0)
                                - predicted[visible_px][:, :3].mean(axis=0)
                            ).mean()
                        )
                        continuity = max(0.0, 1.0 - 2.0 * gap)
                    else:
                        continuity = 0.0 if region_px.any() else 1.0
                    plausibility += alpha * continuity * mass
                    unknowable_mass += mass
        rows.append(row)
    used = {index for index in assignment if index is not None}
    extra_mass = sum(candidate[j, ..., 3].sum() for j in range(len(candidate)) if j not in used)
    total_mass = candidate[..., 3].sum()
    truth_composite = composite(reference)
    predicted_composite = composite(candidate)
    truth_opaque = truth_composite[..., 3] >= 0.99
    covered = predicted_composite[..., 3] >= 0.99
    axes = {
        "layered_object_match": float(np.prod([row["layer_quality"] for row in rows])),
        "layered_visible_retention": 1 - float(np.mean(visible_errors)) if visible_errors else 1.0,
        "layered_hidden_completion": 1 - float(np.mean(hidden_errors)) if hidden_errors else 1.0,
        "layered_edit_holes": 1 - float(np.mean(edit_errors)) if edit_errors else 1.0,
        "layered_nonredundancy": 1 - float(extra_mass / total_mass) if total_mass else 1.0,
        "layered_reconstruction": 1 - float(np.abs(truth_composite - predicted_composite).mean()),
        "layered_composite_coverage": float((covered & truth_opaque).sum() / truth_opaque.sum())
        if truth_opaque.any()
        else 1.0,
        "layered_inferable_completion": 1.0 - inferable_error / inferable_mass
        if inferable_mass
        else 1.0,
        "layered_unknowable_plausibility": plausibility / unknowable_mass
        if unknowable_mass
        else 1.0,
    }
    return {"assignment": assignment, "objects": rows, "axes": axes}


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


def adapter_overrides(adapter: Path | None) -> list[str]:
    """The ``model`` overrides that select the arm: a PEFT directory, or None for no adapter."""
    from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME

    if adapter is None:
        # Drop the whole section: the recipe's lora settings require use_lora.
        return ["model.use_lora=false", "model.lora=null"]
    path = adapter / LORA_WEIGHTS_NAME if (adapter / LORA_WEIGHTS_NAME).is_dir() else adapter
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"{path} is not a PEFT adapter directory")
    return ["model.use_lora=true", f"model.lora.path={path}"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="bundled config name or absolute YAML")
    parser.add_argument("overrides", nargs="*", help="preset overlays and dotlist overrides")
    parser.add_argument(
        "--adapter", type=Path, help="PEFT directory or training checkpoint dir; omit for base"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[4401, 4402])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    overrides = [*adapter_overrides(args.adapter), *args.overrides]

    import torch

    from vrl.config.data import manifest_sources
    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry
    from vrl.trainers.data.layered_scenes import load_layered_scene, scene_cache_paths

    root = parse_config(load_config(args.config, overrides=overrides))
    split = str(root.data.split)
    scenes = scene_cache_paths(manifest_sources(root.data.manifest), split)
    steps = int(root.sampling.num_steps)
    args.output.mkdir(parents=True, exist_ok=True)
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
                layers = decompose(model, cache, steps=steps, seed=seed)
                for i, layer in enumerate(layers):
                    layer.save(out / f"layer_{i}.png")
            stack = np.stack(
                [np.asarray(layer, dtype=np.float32) / 255.0 for layer in layers]
            )  # [L, H, W, 4]
            if any(scene.glob("layer_*.png")):
                scores = _score(stack, scene)
            else:
                source = np.asarray(
                    Image.open(scene / "source.png").convert("RGB"), dtype=np.float32
                )
                scores = truth_free_metrics(stack, source / 255.0)
            rows.append({"scene": scene.name, "seed": seed, "layers": layer_count, **scores})
            logger.info(
                "%s", json.dumps({**rows[-1], "elapsed": round(time.perf_counter() - started, 1)})
            )
            (args.output / "scores.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    # Scenes with and without truth carry different keys; each mean covers the
    # rows that have its key.
    means = {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in (*SCORE_KEYS, "background_recovery", *TRUTH_FREE_KEYS)
        if any(key in row for row in rows)
    }
    summary = {
        "arm": "base" if args.adapter is None else str(args.adapter),
        "config": args.config,
        "overrides": overrides,
        "split": split,
        "scenes": len(scenes),
        "seeds": args.seeds,
        "steps": steps,
        "sampler": "Heun; fp32 states; CFG1",
        "means": means,
        "rows": rows,
        "scope": "Native inference on cached scenes; hand-set known-layer metrics; "
        "not real-photo evidence.",
    }
    (args.output / "scores.json").write_text(json.dumps(summary, indent=1) + "\n")
    logger.info(
        "%s",
        json.dumps({"arm": summary["arm"], "means": {k: round(v, 4) for k, v in means.items()}}),
    )


if __name__ == "__main__":
    main()
