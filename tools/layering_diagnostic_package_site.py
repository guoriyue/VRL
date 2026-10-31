"""Copy verified diagnostic results and unchanged PNGs into the report Site."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "summary.json").read_text())
    assert report["complete"] and report["result_count"] == 72
    plan = report.pop("plan")
    titles = {
        "luma_cat_r00": (
            "Luma · Cat",
            "Luma generated reference image used during training. Original budget: 9 layers. Cat mask index was audited separately; no amodal ground truth is available.",
        ),
        "coco_L6_r02": (
            "COCO · People and table",
            "Real COCO photo selected for background residual in the previous evaluation. Original budget: 6 layers. No amodal ground truth is available.",
        ),
        "rendered_scene_150000": (
            "Rendered · Scene 150000",
            "Held-out rendered scene. Original budget: 7 layers. 8,691 inferable hidden object-pixels with ground truth.",
        ),
        "rendered_scene_150004": (
            "Rendered · Scene 150004",
            "Held-out rendered scene. Original budget: 5 layers. 27,328 inferable hidden object-pixels with ground truth.",
        ),
    }
    report["scenes"] = [
        {
            "id": row["id"],
            "layers": row["layers"],
            "title": titles[row["id"]][0],
            "description": titles[row["id"]][1],
            "has_truth": "truth_dir" in row["metadata"],
        }
        for row in plan["scenes"]
    ]
    report["settings"] = {
        key: plan[key]
        for key in [
            "seed",
            "extra_layers",
            "sampling_steps",
            "resolution",
            "adapter_sha256",
            "caveats",
            "additional_diagnostic",
        ]
    }
    destination = args.site / "diagnostics"
    destination.mkdir(exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(args.root / "analysis.json", destination / "analysis.json")
    copied = 0
    for row in report["rows"]:
        setting = f"layers{row['requested_layers']}_steps{row['sampling_steps']}"
        target = destination / "assets" / row["id"] / row["arm"] / setting
        target.mkdir(parents=True, exist_ok=True)
        source = args.root / "eval" / row["arm"] / row["id"] / setting
        for i in range(row["actual_layers"]):
            name = f"layer_{i}.png"
            shutil.copyfile(source / name, target / name)
            copied += 1
    for row in plan["scenes"]:
        if "truth_dir" not in row["metadata"]:
            continue
        source = Path(row["metadata"]["truth_dir"])
        target = destination / "assets" / row["id"] / "truth"
        target.mkdir(parents=True, exist_ok=True)
        for name in [*(f"layer_{i}.png" for i in range(row["layers"])), "inferable_mask.png"]:
            shutil.copyfile(source / name, target / name)
    print(json.dumps({"copied_original_output_pngs": copied, "results": len(report["rows"])}))


if __name__ == "__main__":
    main()
