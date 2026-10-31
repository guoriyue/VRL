"""Evaluate one layering arm across four GPUs; keep generated layers for rescoring.

Run under torchrun. Each rank builds one streamed BF16 rollout model and owns a
disjoint subset of the evaluation rows. This script does not initialize a
process group: evaluation has no parameter or gradient collectives.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)
from vrl.scripts.denoise.encode_layered_scenes import encode_scene
from vrl.scripts.eval.layered_decompose_eval import _score, decompose
from vrl.trainers.data.layered_scenes import load_layered_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integrity-weight", type=float, default=0.0)
    parser.add_argument("--grouping-mode", choices=["pq", "partition"], default="pq")
    parser.add_argument("--reward-kwargs", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    root = parse_config(
        load_config(
            "experiment/qwen_image_layered/online_grpo_layered_grouping",
            overrides=[f"model.lora.path={args.adapter}", "precision.float32_precision=tf32"],
        )
    )
    entry = get_model_family_entry("qwen_image_layered")
    build = entry.resolve_model_build(
        root,
        torch.device("cuda", rank),
        precision=PrecisionPolicy.from_section(root.precision),
    )
    model = entry.build_rollout(build, expected_identity=None).model.eval()
    kwargs = {"integrity_weight": args.integrity_weight, "grouping_mode": args.grouping_mode}
    if args.reward_kwargs is not None:
        kwargs.update(json.loads(args.reward_kwargs.read_text()))
    scorer = LayeredGroupingRewardModel(kwargs)
    results = []
    # Greedy longest-first assignment keeps high-layer scenes from leaving one
    # rank running alone at the tail of an evaluation.
    assignments = [[] for _ in range(world)]
    costs = [0.0] * world
    for row in sorted(rows, key=lambda item: item["layers"], reverse=True):
        owner = min(range(world), key=lambda index: costs[index])
        assignments[owner].append(row)
        costs[owner] += (row["layers"] + 2) ** 2
    for row in assignments[rank]:
        started = time.monotonic()
        scene = Path(row["scene_dir"])
        cache_path = scene / "cache.pt"
        if not cache_path.exists():
            cache = encode_scene(
                model,
                scene,
                prompt="a clean, well composed image",
                height=640,
                width=640,
                layers=row["layers"],
            )
            cache.update(family=entry.family, model_revision=str(root.model.revision))
            torch.save(cache, cache_path)
        cache = load_layered_scene(
            cache_path,
            family=entry.family,
            model_revision=str(root.model.revision),
        )
        out = args.output / row["id"] / "seed4401"
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"layer_{i}.png" for i in range(row["layers"])]
        if not all(path.exists() for path in paths):
            layers = decompose(model, cache, steps=10, seed=4401)
            for layer, path in zip(layers, paths, strict=True):
                layer.save(path)
        rgba = np.stack(
            [np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255 for p in paths]
        )
        metadata = {**row["metadata"], "reference_images": [str(scene / "source.png")]}
        stack = LayerStack.from_rgba(rgba)
        reference = LayeredReference.from_metadata(metadata, 640, 640)
        scores = scorer.score(stack, reference)
        if any(scene.glob("layer_*.png")):
            scores.update(_score(rgba, scene))
        result = {
            "id": row["id"],
            "seed": 4401,
            "rank": rank,
            "seconds": time.monotonic() - started,
            **scores,
        }
        results.append(result)
        (out / "main_scores.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / f"rank_{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
