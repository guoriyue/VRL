"""Compare layering checkpoints on independent visible and edit diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("arms", nargs="+", type=Path)
    args = parser.parse_args()
    scenes = [json.loads(s) for s in args.rows.read_text().splitlines()]
    arms = {}
    for arm in args.arms:
        rows = []
        for scene in scenes:
            root = arm / scene["id"] / "seed4401"
            scores = json.loads((root / "main_scores.json").read_text())
            ref = LayeredReference.from_metadata(
                {
                    **scene["metadata"],
                    "reference_images": [str(Path(scene["scene_dir"]) / "source.png")],
                },
                640,
                640,
            )
            stack = LayerStack.from_directory(root, 640, 640)
            labeled = np.logical_or.reduce(ref.objects)
            matches, _ = stack.match(ref.objects)
            objects = []
            for i, mask in enumerate(ref.objects):
                k = matches.get(i)
                inter = 0 if k is None else int((mask & stack.regions[k]).sum())
                layer_area = 0 if k is None else int((labeled & stack.regions[k]).sum())
                objects.append(
                    {
                        "coverage": inter / int(mask.sum()),
                        "purity": inter / layer_area if layer_area else 0.0,
                        "background": float((mask & stack.regions[0]).sum() / mask.sum()),
                    }
                )
            scores.update(
                complete_pure_objects=sum(
                    o["coverage"] >= 0.9 and o["purity"] >= 0.9 for o in objects
                ),
                mean_object_coverage=float(np.mean([o["coverage"] for o in objects])),
                object_background_share=float(np.mean([o["background"] for o in objects])),
                merge_layers=int(
                    sum(
                        sum((mask & region).sum() / mask.sum() > 0.5 for mask in ref.objects) >= 2
                        for region in stack.regions[1:]
                    )
                ),
            )
            rows.append(scores)
        groups = {}
        for group in ["coco", "rendered", "luma"]:
            selected = [r for r in rows if r["id"].startswith(group)]
            keys = set.intersection(*(set(r) for r in selected)) - {
                "id",
                "seed",
                "rank",
                "seconds",
            }
            groups[group] = {
                "count": len(selected),
                "means": {k: float(np.mean([r[k] for r in selected])) for k in sorted(keys)},
            }
        arms[arm.name] = {"rows": rows, "groups": groups}
    baseline = next(iter(arms))
    gates = {}
    for name, arm in arms.items():
        if name == baseline:
            continue
        a, b = arm["groups"], arms[baseline]["groups"]
        c, bc = a["coco"]["means"], b["coco"]["means"]
        r, br = a["rendered"]["means"], b["rendered"]["means"]
        gates[name] = {
            "more_complete_pure_photo_objects": c["complete_pure_objects"]
            >= bc["complete_pure_objects"] + 0.1,
            "photo_background_not_worse_by_more_than_one_point": c["object_background_share"]
            <= bc["object_background_share"] + 0.01,
            "no_more_photo_merge_layers": c["merge_layers"] <= bc["merge_layers"] + 1e-9,
            "rendered_edit_quality_not_worse": r["layered_edit_holes"]
            >= br["layered_edit_holes"] - 0.001,
            "rendered_nonredundancy_not_worse": r["layered_nonredundancy"]
            >= br["layered_nonredundancy"] - 0.001,
        }
    report = {
        "baseline": baseline,
        "arms": arms,
        "promotion_gates": gates,
        "scope": "Single seed, 15 held-out photos and 8 rendered scenes; Luma is auxiliary training data. These gates are an initial screen, not proof of general quality.",
        "next_gate": "Visual inspection of complete layers and object-removal edits, then a second evaluation seed before promotion.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"promotion_gates": gates, "groups": {a: v["groups"] for a, v in arms.items()}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
