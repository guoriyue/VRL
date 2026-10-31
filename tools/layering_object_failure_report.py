"""Report visible fragmentation and full-truth opacity excess independently of reward.

Opacity excess is a spatial diagnostic, not a semantic duplicate detector.
Unannotated objects and invisible copies cannot be judged from visible masks alone.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack
from vrl.rewards.models.layered_removal import single_object_removal_quality


def measure_failures(stack, reference):
    records = []
    for index, mask in enumerate(reference.objects):
        coverage = np.array([(mask & region).sum() / mask.sum() for region in stack.regions])
        records.append(
            {
                "object": index,
                "area": int(mask.sum()),
                "foreground_coverage_by_layer": coverage[1:].tolist(),
                "largest_foreground_piece": float(coverage[1:].max()),
                "background_share": float(coverage[0]),
                "split_at_10_percent": bool((coverage[1:] >= 0.1).sum() >= 2),
                "majority_in_background": bool(coverage[0] > 0.5),
            }
        )
    result = {
        "objects": records,
        "split_objects": sum(r["split_at_10_percent"] for r in records),
        "majority_background_objects": sum(r["majority_in_background"] for r in records),
    }
    if reference.complete_layer_truth:
        expected = reference.truth.rgba[1:, ..., 3].sum(axis=0)
        predicted = stack.rgba[1:, ..., 3].sum(axis=0)
        result["excess_foreground_opacity_per_truth_area"] = float(
            np.maximum(predicted - expected, 0).sum() / max(float(expected.sum()), 1.0)
        )
        result["single_object_removals"] = single_object_removal_quality(stack, reference)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for row in map(json.loads, args.rows.read_text().splitlines()):
        reference = LayeredReference.from_metadata(
            {**row["metadata"], "reference_images": [str(Path(row["scene_dir"]) / "source.png")]},
            640,
            640,
        )
        stack = LayerStack.from_directory(args.arm / row["id"] / "seed4401", 640, 640)
        records.append({"id": row["id"], **measure_failures(stack, reference)})
    groups = {}
    for group in ("coco", "natural", "rendered", "luma"):
        selected = [r for r in records if r["id"].startswith(group)]
        if not selected:
            continue
        groups[group] = {
            "objects": sum(len(r["objects"]) for r in selected),
            "split_objects": sum(r["split_objects"] for r in selected),
            "majority_background_objects": sum(r["majority_background_objects"] for r in selected),
        }
        excess = [
            r["excess_foreground_opacity_per_truth_area"]
            for r in selected
            if "excess_foreground_opacity_per_truth_area" in r
        ]
        if excess:
            groups[group]["mean_excess_foreground_opacity_per_truth_area"] = float(np.mean(excess))
    report = {
        "arm": str(args.arm),
        "groups": groups,
        "records": records,
        "limitations": "A split requires two foreground layers each holding >=10% of visible object pixels. Opacity excess includes misplaced foreground as well as copies and only applies to exhaustive full truth. Background ghosts require removal review. No semantic duplication claim for COCO.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
