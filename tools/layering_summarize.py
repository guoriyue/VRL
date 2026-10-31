"""Summarize complete arms on identical rows; add explicit background/merge diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("arms", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    summaries = {}
    for arm in args.arms:
        scores = []
        for row in rows:
            out = arm / row["id"] / "seed4401"
            result = json.loads((out / "main_scores.json").read_text())
            paths = [out / f"layer_{i}.png" for i in range(row["layers"])]
            rgba = np.stack(
                [np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255 for p in paths]
            )
            layers = LayerStack.from_rgba(rgba)
            ref = LayeredReference.from_metadata(
                {
                    **row["metadata"],
                    "reference_images": [str(Path(row["scene_dir"]) / "source.png")],
                },
                640,
                640,
            )
            shares = [float((mask & layers.regions[0]).sum() / mask.sum()) for mask in ref.objects]
            merged = sum(
                sum(
                    float((mask & layers.regions[k]).sum() / mask.sum()) > 0.5
                    for mask in ref.objects
                )
                >= 2
                for k in layers.object_layers()
            )
            result.update(
                object_background_share=float(np.mean(shares)) if shares else 0.0,
                objects_majority_in_background=float(np.mean(np.asarray(shares) > 0.5))
                if shares
                else 0.0,
                layers_with_majority_of_multiple_objects=float(merged),
            )
            if row["id"] == "luma_cat_r00":
                # The uploaded NPZ uses generic names and its mask order differs
                # from metadata.objects. Use the explicitly audited raw index.
                index = row["diagnostic_object_indices"]["cat"]
                with np.load(row["metadata"]["instances"]) as data:
                    mask = data["masks"][index].astype(np.uint8) * 255
                cat = (
                    np.asarray(Image.fromarray(mask).resize((640, 640), Image.Resampling.NEAREST))
                    > 127
                )
                result["cat_best_layer_coverage"] = max(
                    float((cat & region).sum() / cat.sum()) for region in layers.regions[1:]
                )
            scores.append(result)
        sets = {}
        for group in ("coco", "luma", "rendered"):
            selected = [s for s in scores if s["id"].startswith(group)]
            if not selected:
                continue
            keys = set.intersection(*(set(s) for s in selected)) - {
                "id",
                "seed",
                "rank",
                "seconds",
            }
            sets[group] = {
                "count": len(selected),
                "means": {key: float(np.mean([s[key] for s in selected])) for key in sorted(keys)},
            }
        summaries[arm.name] = {"sets": sets, "rows": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "reward": "main c090e97db: grouping + fidelity + 0.8 * completion",
        "diagnostics": {
            "object_background_share": "Mean fraction of each annotated object's pixels assigned to the background layer; lower is better.",
            "objects_majority_in_background": "Fraction of annotated objects with more than half their visible pixels in the background; lower is better.",
            "layers_with_majority_of_multiple_objects": "Number of foreground layers containing more than half the visible pixels of two or more annotated objects; lower is better.",
            "cat_best_layer_coverage": "Largest fraction of the annotated cat visible in any single foreground layer.",
            "scope": "Diagnostics are defined for this run and are not the historical matched/missing/merged implementation.",
        },
        "arms": summaries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({arm: summary["sets"] for arm, summary in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
