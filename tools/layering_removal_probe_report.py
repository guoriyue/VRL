"""Report target-removal interventions on separately reviewed partial references."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack
from vrl.rewards.models.layered_removal import (
    single_object_removal_quality,
    target_removal_quality,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("arms", nargs="+", type=Path)
    args = parser.parse_args()
    rows = list(map(json.loads, args.rows.read_text().splitlines()))
    report = {}
    for arm in args.arms:
        cases = []
        for row in rows:
            assert row["metadata"].get("target_removal_reference") is True
            reference = LayeredReference.from_metadata(row["metadata"], 640, 640)
            directory = arm / row["id"] / "seed4401"
            stack = LayerStack.from_directory(directory, 640, 640)
            quality = target_removal_quality(stack, reference)
            individual = single_object_removal_quality(stack, reference)
            for edit in individual:
                rgba = stack.rgba.copy()
                if edit["candidate_layer"] is not None:
                    rgba[edit["candidate_layer"], ..., 3] = 0
                pixels = LayerStack.from_rgba(rgba).composite()
                Image.fromarray(np.rint(pixels * 255).clip(0, 255).astype(np.uint8)).save(
                    directory / f"after_removing_object_{edit['truth_layer']}.png"
                )
            matches, _ = stack.match(reference.objects)
            retained = stack.rgba.copy()
            for k in matches.values():
                retained[k, ..., 3] = 0
            composite = LayerStack.from_rgba(retained).composite()
            Image.fromarray(np.rint(composite * 255).clip(0, 255).astype(np.uint8)).save(
                directory / "after_target_removal.png"
            )
            base = json.loads((directory / "main_scores.json").read_text())
            cases.append(
                {
                    "id": row["id"],
                    "target_removal_quality": quality,
                    "individual_object_edits": individual,
                    "mean_single_object_removal_quality": float(
                        np.mean([e["quality"] for e in individual])
                    ),
                    "visible_grouping": base["grouping"],
                    "fidelity": base["fidelity"],
                    "matched_target_layers": {str(k): v for k, v in matches.items()},
                    "annotated_targets": len(reference.objects),
                }
            )
        report[arm.name] = {
            "cases": cases,
            "mean_removal_quality": float(np.mean([r["target_removal_quality"] for r in cases])),
            "mean_single_object_removal_quality": float(
                np.mean([r["mean_single_object_removal_quality"] for r in cases])
            ),
        }
    args.output.write_text(
        json.dumps(
            {
                "arms": report,
                "scope": "Four assistant-reviewed external benchmark scenes with generated/inpainted references. Pixel agreement under target-removal interventions is diagnostic, not proof of semantic or physical correctness. These scenes are never training data.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({a: r["mean_removal_quality"] for a, r in report.items()}))


if __name__ == "__main__":
    main()
