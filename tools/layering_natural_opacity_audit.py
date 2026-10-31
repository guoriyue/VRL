"""Check hidden copies across opacity thresholds without changing the composite."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.run / "data/dataset_audit.json").read_text())
    scorer = LayeredGroupingRewardModel(json.loads((args.run / "reward_kwargs.json").read_text()))
    records = []
    for row in audit["scenes"]:
        scene = Path(row["scene_dir"])
        ref = LayeredReference.from_metadata(
            {
                "reference_images": [str(scene / "source.png")],
                "truth_dir": str(scene),
                "complete_layer_truth": True,
            },
            640,
            640,
        )
        truth = ref.truth
        correct = scorer.score(truth, ref)["layered_grouping"]
        scores, errors = [], []
        for alpha in [0.1, 0.49, 0.51, 1.0]:
            duplicate = truth.rgba[1].copy()
            duplicate[duplicate[..., 3] != 1] = 0
            duplicate[..., 3] *= alpha
            stack = LayerStack.from_rgba(
                np.concatenate([truth.rgba[:1], duplicate[None], truth.rgba[1:]])
            )
            scores.append(scorer.score(stack, ref)["layered_grouping"])
            errors.append(float(np.max(np.abs(stack.composite() - truth.composite()))))
        passed = (
            max(errors) < 1e-6 and correct > scores[0] and all(a > b for a, b in pairwise(scores))
        )
        records.append(
            {
                "id": row["id"],
                "correct": correct,
                "opacities": [0.1, 0.49, 0.51, 1.0],
                "scores": scores,
                "maximum_composite_error": max(errors),
                "passed": passed,
            }
        )
    report = {
        "passed": all(r["passed"] for r in records),
        "hidden_copy_variants": 4 * len(records),
        "records": records,
    }
    (args.run / "opacity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "records"})
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
