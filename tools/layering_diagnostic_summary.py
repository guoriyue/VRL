"""Validate the complete diagnostic factorial and report paired effects."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text())
    records = []
    image_count = 0
    for arm, scene, extra, steps in itertools.product(
        plan["adapters"], plan["scenes"], plan["extra_layers"], plan["sampling_steps"]
    ):
        layers = scene["layers"] + extra
        path = args.root / "eval" / arm / scene["id"] / f"layers{layers}_steps{steps}"
        result = json.loads((path / "result.json").read_text())
        assert result["arm"] == arm and result["id"] == scene["id"]
        assert result["actual_layers"] == result["requested_layers"] == layers
        assert result["sampling_steps"] == steps and result["seed"] == plan["seed"]
        assert result["adapter_sha256"] == plan["adapter_sha256"][arm]
        assert all(math.isfinite(x) for x in result.values() if isinstance(x, (int, float)))
        assert len(list(path.glob("layer_*.png"))) == layers
        for i in range(layers):
            with Image.open(path / f"layer_{i}.png") as image:
                assert image.size == (640, 640) and image.mode == "RGBA"
                image.verify()
            image_count += 1
        stack = LayerStack.from_directory(path, 640, 640)
        reference = LayeredReference.from_metadata(
            {
                **scene["metadata"],
                "reference_images": [str(Path(scene["scene_dir"]) / "source.png")],
            },
            640,
            640,
        )
        labeled = np.logical_or.reduce(reference.objects)
        matches, _ = stack.match(reference.objects)
        ious = []
        for index, mask in enumerate(reference.objects):
            layer = matches.get(index)
            region = stack.regions[layer] & labeled if layer is not None else np.zeros_like(mask)
            intersection = float((mask & region).sum())
            ious.append(intersection / float((mask | region).sum()))
        result["annotated_object_count"] = len(ious)
        result["objects_with_iou_at_least_0_5"] = sum(value >= 0.5 for value in ious)
        result["object_match_iou"] = ious
        records.append(result)
    assert len(records) == plan["expected_results"]
    metrics = [
        "layered_grouping",
        "grouping",
        "completion",
        "fidelity",
        "object_background_share",
        "layers_with_majority_of_multiple_objects",
        "active_foreground_layers",
        "objects_with_iou_at_least_0_5",
        "empty_or_occluded_foreground_layers",
        "hidden_rgb_mae",
        "hidden_alpha_mae",
        "hidden_alpha_recall",
        "hidden_objects_without_match",
        "layered_object_match",
    ]
    groups = []
    for arm, extra, steps, kind in itertools.product(
        plan["adapters"], plan["extra_layers"], plan["sampling_steps"], ["photo", "rendered"]
    ):
        selected = [
            x
            for x in records
            if x["arm"] == arm
            and x["extra_layers"] == extra
            and x["sampling_steps"] == steps
            and x["id"].startswith("rendered") == (kind == "rendered")
        ]
        groups.append(
            {
                "arm": arm,
                "extra_layers": extra,
                "sampling_steps": steps,
                "kind": kind,
                "count": len(selected),
                "means": {
                    key: float(np.mean([x[key] for x in selected]))
                    for key in metrics
                    if all(x.get(key) is not None for x in selected)
                },
            }
        )
    report = {
        "complete": True,
        "result_count": len(records),
        "verified_images": image_count,
        "plan": plan,
        "groups": groups,
        "rows": records,
        "interpretation": "Descriptive paired case study. Do not equate sharpness alone with valid object completion; inspect RGB and alpha with truth. Layer indices may permute across outputs.",
    }
    (args.root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "results": len(records), "verified_images": image_count}))


if __name__ == "__main__":
    main()
