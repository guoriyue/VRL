"""Audit saved production candidates without modifying the running trainer."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from tools.layering_natural_report import measure
from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def pixel_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run = args.run
    rows = list(map(json.loads, (run / "data/train.jsonl").read_text().splitlines()))
    lookup = {pixel_hash(row["reference_image"]): row for row in rows}
    scorer = LayeredGroupingRewardModel(json.loads((run / "reward_kwargs.json").read_text()))
    records, groups = [], defaultdict(list)
    for score_path in sorted((run / "rollouts").glob("*/scores.json")):
        directory = score_path.parent
        row = lookup[pixel_hash(directory / "source.png")]
        scene = Path(row["metadata"]["truth_dir"])
        data = {**row, "id": scene.name, "scene_dir": str(scene)}
        metrics = measure(directory, data)
        reference = LayeredReference.from_metadata(row["metadata"], 640, 640)
        new_scores = scorer.score(LayerStack.from_directory(directory, 640, 640), reference)
        saved = json.loads(score_path.read_text())
        delta = abs(saved["layered_grouping"] - new_scores["layered_grouping"])
        if delta > 0.01:
            raise ValueError(f"Unexpected saved-media score difference: {directory} {delta}")
        group = directory.name.split("_qwen_image_layered-", 1)[-1].rsplit(":sample:", 1)[0]
        record = {
            "directory": str(directory),
            "group": group,
            "source_id": row["metadata"]["source_id"],
            "training_scene": scene.name,
            "reward": saved["layered_grouping"],
            "quantization_reward_difference": delta,
            **metrics,
        }
        records.append(record)
        groups[group].append(record)
    summary = []
    for key, samples in groups.items():
        if len(samples) != 4:
            continue
        low, high = (
            min(samples, key=lambda r: r["reward"]),
            max(samples, key=lambda r: r["reward"]),
        )
        summary.append(
            {
                "group": key,
                "source_id": high["source_id"],
                "training_scene": high["training_scene"],
                "highest": high,
                "lowest": low,
                "reward_spread": high["reward"] - low["reward"],
                "complete_objects_change": high["complete_pure_objects"]
                - low["complete_pure_objects"],
                "full_object_diagnostic_change": high["full_object_quality_diagnostic"]
                - low["full_object_quality_diagnostic"],
            }
        )
    report = {
        "samples": len(records),
        "complete_groups": len(summary),
        "unique_scenes": len({r["training_scene"] for r in records}),
        "groups": summary,
        "records": records,
        "scope": "Training samples and old independent diagnostics; not held-out improvement or independent human preference labels.",
    }
    (run / "training_rollout_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["groups", "records"]}, indent=2)
    )


if __name__ == "__main__":
    main()
