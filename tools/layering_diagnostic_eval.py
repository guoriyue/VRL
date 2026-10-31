"""Paired layer-budget / sampling-step evaluation, with hidden-region diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)
from vrl.scripts.eval.layered_decompose_eval import _score, decompose
from vrl.trainers.data.layered_scenes import load_layered_scene


def diagnostics(stack: LayerStack, reference: LayeredReference) -> dict:
    shares = [float((m & stack.regions[0]).sum() / m.sum()) for m in reference.objects]
    matches, _ = stack.match(reference.objects)
    result = {
        "object_background_share": float(np.mean(shares)),
        "objects_majority_in_background": float(np.mean(np.asarray(shares) > 0.5)),
        "layers_with_majority_of_multiple_objects": sum(
            sum(float((m & stack.regions[k]).sum() / m.sum()) > 0.5 for m in reference.objects)
            >= 2
            for k in stack.object_layers()
        ),
        "visible_object_matches": {str(i): int(k) for i, k in matches.items()},
        "active_foreground_layers": len(stack.object_layers()),
        "empty_or_occluded_foreground_layers": len(stack.rgba) - 1 - len(stack.object_layers()),
    }
    truth = reference.truth
    if truth is None:
        return result
    truth_matches, _ = stack.match(list(truth.regions[1:]))
    objects = []
    sums = {name: 0.0 for name in ("rgb_mae", "alpha_mae", "alpha_recall")}
    count = 0
    for i in range(1, len(truth.rgba)):
        mask = (truth.rgba[i, ..., 3] >= 0.5) & ~truth.regions[i] & reference.inferable
        n = int(mask.sum())
        if not n:
            continue
        k = truth_matches.get(i - 1)
        predicted = stack.rgba[k] if k is not None else np.zeros_like(truth.rgba[i])
        target = truth.rgba[i]
        ours_rgb = predicted[..., :3] * predicted[..., 3:]
        truth_rgb = target[..., :3] * target[..., 3:]
        metrics = {
            "rgb_mae": float(np.abs(ours_rgb - truth_rgb)[mask].mean()),
            "alpha_mae": float(np.abs(predicted[..., 3] - target[..., 3])[mask].mean()),
            "alpha_recall": float((predicted[..., 3][mask] >= 0.5).mean()),
        }
        gradient_errors, predicted_gradients, truth_gradients = [], [], []
        for axis in (0, 1):
            valid = mask[1:, :] & mask[:-1, :] if axis == 0 else mask[:, 1:] & mask[:, :-1]
            if valid.any():
                ours = np.diff(ours_rgb, axis=axis)[valid]
                theirs = np.diff(truth_rgb, axis=axis)[valid]
                gradient_errors.append(np.abs(ours - theirs).reshape(-1))
                predicted_gradients.append(np.abs(ours).reshape(-1))
                truth_gradients.append(np.abs(theirs).reshape(-1))
        extra = {}
        if gradient_errors:
            extra = {
                "gradient_mae": float(np.concatenate(gradient_errors).mean()),
                "predicted_gradient_magnitude": float(np.concatenate(predicted_gradients).mean()),
                "truth_gradient_magnitude": float(np.concatenate(truth_gradients).mean()),
            }
        objects.append(
            {"truth_layer": i, "matched_layer": k, "hidden_pixels": n, **metrics, **extra}
        )
        for name in sums:
            sums[name] += n * metrics[name]
        count += n
    result["hidden_objects"] = objects
    result["hidden_inferable_pixels"] = count
    result["hidden_objects_without_match"] = sum(x["matched_layer"] is None for x in objects)
    result.update(
        {f"hidden_{key}": value / count if count else None for key, value in sums.items()}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    plan = json.loads(args.plan.read_text())
    adapter = Path(plan["adapters"][args.arm])
    root = parse_config(
        load_config(
            "experiment/qwen_image_layered/online_grpo_layered_grouping",
            overrides=[f"model.lora.path={adapter}", "precision.float32_precision=tf32"],
        )
    )
    entry = get_model_family_entry("qwen_image_layered")
    build = entry.resolve_model_build(
        root, torch.device("cuda", rank), precision=PrecisionPolicy.from_section(root.precision)
    )
    model = entry.build_rollout(build, expected_identity=None).model.eval()
    scorer = LayeredGroupingRewardModel({})
    jobs = [
        {**row, "extra_layers": extra, "sampling_steps": steps}
        for row in plan["scenes"]
        for extra in plan["extra_layers"]
        for steps in plan["sampling_steps"]
    ]
    assignments = [[] for _ in range(world)]
    costs = [0.0] * world

    def cost(job: dict) -> float:
        return (job["layers"] + job["extra_layers"]) * job["sampling_steps"]

    for job in sorted(jobs, key=cost, reverse=True):
        owner = min(range(world), key=costs.__getitem__)
        assignments[owner].append(job)
        costs[owner] += cost(job)
    for job in assignments[rank]:
        total_layers = job["layers"] + job["extra_layers"]
        out = (
            args.output
            / args.arm
            / job["id"]
            / f"layers{total_layers}_steps{job['sampling_steps']}"
        )
        out.mkdir(parents=True, exist_ok=True)
        marker = out / "result.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            assert previous["arm"] == args.arm and previous["seed"] == plan["seed"]
            assert previous["requested_layers"] == total_layers
            assert previous["sampling_steps"] == job["sampling_steps"]
            assert all((out / f"layer_{i}.png").exists() for i in range(total_layers))
            continue
        started = time.monotonic()
        scene = Path(job["scene_dir"])
        cache = load_layered_scene(
            scene / "cache.pt", family=entry.family, model_revision=str(root.model.revision)
        )
        assert cache["layers"] == job["layers"]
        # Only decompose reads this replacement. Truth targets stay in the
        # original cache and scoring always uses the original truth directory.
        cache = {**cache, "layers": total_layers}
        torch.cuda.reset_peak_memory_stats()
        images = decompose(model, cache, steps=job["sampling_steps"], seed=plan["seed"])
        assert len(images) == total_layers
        for i, image in enumerate(images):
            assert image.size == (640, 640) and image.mode == "RGBA"
            image.save(out / f"layer_{i}.png")
        rgba = np.stack([np.asarray(image, dtype=np.float32) / 255 for image in images])
        stack = LayerStack.from_rgba(rgba)
        ref = LayeredReference.from_metadata(
            {**job["metadata"], "reference_images": [str(scene / "source.png")]}, 640, 640
        )
        scores = scorer.score(stack, ref)
        if ref.truth is not None:
            scores.update(_score(rgba, scene))
        record = {
            "arm": args.arm,
            "id": job["id"],
            "seed": plan["seed"],
            "original_layers": job["layers"],
            "extra_layers": job["extra_layers"],
            "requested_layers": total_layers,
            "actual_layers": len(images),
            "sampling_steps": job["sampling_steps"],
            "rank": rank,
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "conditioning": "Original frozen cache; only layers changed in memory",
            "adapter_sha256": plan["adapter_sha256"][args.arm],
            **scores,
            **diagnostics(stack, ref),
        }
        if job["id"] == "luma_cat_r00":
            with np.load(job["metadata"]["instances"]) as data:
                cat = data["masks"][job["diagnostic_object_indices"]["cat"]].astype(bool)
            record["cat_best_layer_coverage"] = max(
                float((cat & region).sum() / cat.sum()) for region in stack.regions[1:]
            )
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(marker)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in [
                        "arm",
                        "id",
                        "requested_layers",
                        "sampling_steps",
                        "seconds",
                        "layered_grouping",
                        "peak_allocated_bytes",
                    ]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
