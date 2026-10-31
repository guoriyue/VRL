"""Independent visible and object-removal diagnostics for the natural-object pilot."""

import argparse
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack
from vrl.rewards.models.layered_removal import (
    single_object_removal_quality,
    target_removal_quality,
)


def measure(directory, row):
    stack = LayerStack.from_directory(directory, 640, 640)
    reference = LayeredReference.from_metadata(
        {**row["metadata"], "reference_images": [str(Path(row["scene_dir"]) / "source.png")]},
        640,
        640,
    )
    matches, _ = stack.match(reference.objects)
    labeled = np.logical_or.reduce(reference.objects)
    coverage, purity, background = [], [], []
    for i, mask in enumerate(reference.objects):
        k = matches.get(i)
        overlap = 0 if k is None else int((mask & stack.regions[k]).sum())
        area = 0 if k is None else int((labeled & stack.regions[k]).sum())
        coverage.append(overlap / int(mask.sum()))
        purity.append(overlap / area if area else 0.0)
        background.append(float((mask & stack.regions[0]).sum() / mask.sum()))
    result = {
        "id": row["id"],
        "objects": len(reference.objects),
        "complete_pure_objects": sum(
            c >= 0.9 and p >= 0.9 for c, p in zip(coverage, purity, strict=True)
        ),
        "background_share": float(np.mean(background)),
        "merge_layers": int(
            sum(
                sum((mask & region).sum() / mask.sum() > 0.5 for mask in reference.objects) >= 2
                for region in stack.regions[1:]
            )
        ),
    }
    if reference.complete_layer_truth:
        # Preserve the previous diagnostic formula across arms, independently of
        # the new reference-normalized training reward.
        full, clean = reference.integrity(stack)
        result.update(
            full_object_quality_diagnostic=full,
            clean_background=clean,
            target_removal_quality=target_removal_quality(stack, reference),
            single_object_removal_quality=float(
                np.mean([x["quality"] for x in single_object_removal_quality(stack, reference)])
            ),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("arms", nargs="+", type=Path)
    args = p.parse_args()
    rows = list(map(json.loads, args.rows.read_text().splitlines()))
    arms = {}
    for arm in args.arms:
        records = [measure(arm / r["id"] / "seed4401", r) for r in rows]
        groups = {}
        for group in ["natural", "coco", "rendered", "luma"]:
            selected = [r for r in records if r["id"].startswith(group)]
            if not selected:
                continue
            keys = set.intersection(*(set(r) for r in selected)) - {"id"}
            groups[group] = {
                "scenes": len(selected),
                "total_complete_pure_objects": sum(r["complete_pure_objects"] for r in selected),
                "total_objects": sum(r["objects"] for r in selected),
                "means": {k: float(np.mean([r[k] for r in selected])) for k in sorted(keys)},
            }
        arms[arm.name] = {"rows": records, "groups": groups}
    baseline = next(iter(arms))
    gates = {}
    for name, arm in arms.items():
        if name == baseline:
            continue
        a, b = arm["groups"], arms[baseline]["groups"]
        gates[name] = {
            "natural_full_object_gain_at_least_001": a["natural"]["means"][
                "full_object_quality_diagnostic"
            ]
            >= b["natural"]["means"]["full_object_quality_diagnostic"] + 0.01,
            "natural_single_removal_not_worse": a["natural"]["means"][
                "single_object_removal_quality"
            ]
            >= b["natural"]["means"]["single_object_removal_quality"] - 0.001,
            "natural_complete_objects_not_fewer": a["natural"]["total_complete_pure_objects"]
            >= b["natural"]["total_complete_pure_objects"],
            "photo_complete_objects_gain_at_least_2": a["coco"]["total_complete_pure_objects"]
            >= b["coco"]["total_complete_pure_objects"] + 2,
            "photo_background_not_worse_by_001": a["coco"]["means"]["background_share"]
            <= b["coco"]["means"]["background_share"] + 0.01,
            "photo_merges_not_more": a["coco"]["means"]["merge_layers"]
            <= b["coco"]["means"]["merge_layers"] + 1e-9,
            "rendered_full_object_not_worse_by_001": a["rendered"]["means"][
                "full_object_quality_diagnostic"
            ]
            >= b["rendered"]["means"]["full_object_quality_diagnostic"] - 0.01,
        }
    report = {
        "baseline": baseline,
        "arms": arms,
        "promotion_gates": gates,
        "scope": "Controlled photo-derived test images are source-disjoint from training, but only three source images. Fifteen real COCO photos remain a separate generalization gate. Luma is auxiliary training data. A passing single-seed screen still requires visual edit review and a second seed.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"groups": {k: v["groups"] for k, v in arms.items()}, "promotion_gates": gates},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
