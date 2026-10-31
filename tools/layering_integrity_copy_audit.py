"""Try invisible duplicate layers against real generated and reference stacks."""

import argparse
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorer = LayeredGroupingRewardModel({"integrity_weight": 1})
    records = []
    for row in map(json.loads, args.rows.read_text().splitlines()):
        if not row["metadata"].get("complete_layer_truth"):
            continue
        reference = LayeredReference.from_metadata(
            {**row["metadata"], "reference_images": [str(Path(row["scene_dir"]) / "source.png")]},
            640,
            640,
        )
        generated = LayerStack.from_directory(args.generated / row["id"] / "seed4401", 640, 640)
        for name, original in [("truth", reference.truth), ("generated", generated)]:
            base = scorer.score(original, reference)["layered_grouping"]
            for k in range(1, len(original.rgba)):
                duplicate = original.rgba[k].copy()
                duplicate[duplicate[..., 3] < 1] = 0
                if not duplicate[..., 3].any():
                    continue
                changed = LayerStack.from_rgba(
                    np.concatenate(
                        [
                            original.rgba[:k],
                            duplicate[None],
                            original.rgba[k:],
                        ]
                    )
                )
                delta = scorer.score(changed, reference)["layered_grouping"] - base
                composite_delta = float(np.abs(original.composite() - changed.composite()).max())
                records.append(
                    {
                        "scene": row["id"],
                        "stack": name,
                        "copied_layer": k,
                        "reward_delta": delta,
                        "composite_max_delta": composite_delta,
                    }
                )
    assert records
    report = {
        "cases": records,
        "all_composites_identical": all(r["composite_max_delta"] < 1e-6 for r in records),
        "no_copy_reward_gain": all(r["reward_delta"] <= 1e-8 for r in records),
        "strictly_lower": sum(r["reward_delta"] < -1e-8 for r in records),
        "worst_reward_delta": max(r["reward_delta"] for r in records),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "cases"}, "cases", len(records))
    assert report["all_composites_identical"] and report["no_copy_reward_gain"]


if __name__ == "__main__":
    main()
