"""Audit actual photo-derived truth against controlled decomposition failures."""

import argparse
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def counterexamples(truth):
    rgba = truth.rgba
    base, back, front = rgba
    # Two non-overlapping pieces of one object, still reproducing exactly its image.
    _, xx = np.where(back[..., 3] >= 0.5)
    cut = int(np.median(xx))
    major, minor = back.copy(), back.copy()
    major[:, cut:, :] = 0
    minor[:, :cut, :] = 0
    split = np.stack([base, major, minor, front])
    dropped_bg = base.copy()
    dropped_bg[..., :3] = minor[..., :3] * minor[..., 3:] + base[..., :3] * (1 - minor[..., 3:])
    merged_alpha = front[..., 3:] + back[..., 3:] * (1 - front[..., 3:])
    premul = front[..., :3] * front[..., 3:] + back[..., :3] * back[..., 3:] * (1 - front[..., 3:])
    merged = np.concatenate(
        [
            np.divide(premul, merged_alpha, out=np.zeros_like(premul), where=merged_alpha > 0),
            merged_alpha,
        ],
        axis=-1,
    )
    duplicate = back.copy()
    duplicate[back[..., 3] != 1] = 0
    dirty_bg = base.copy()
    mask = back[..., 3] == 1
    dirty_bg[mask, :3] = back[mask, :3]
    missing = base.copy()
    missing[..., :3] = back[..., :3] * back[..., 3:] + base[..., :3] * (1 - back[..., 3:])
    empty = np.zeros_like(back)
    cases = {
        "correct": rgba,
        "legal_empty_slots": np.stack([base, empty, back, front, empty]),
        "split": split,
        "fragment_in_background": np.stack([dropped_bg, major, front]),
        "merged": np.stack([base, merged]),
        "missing_object": np.stack([missing, empty, front]),
        "hidden_copy": np.stack([base, duplicate, back, front]),
        "background_copy": np.stack([dirty_bg, back, front]),
    }
    hidden = (back[..., 3] >= 0.5) & (front[..., 3] == 1)
    if hidden.any():
        corrupted = rgba.copy()
        corrupted[1, hidden, :3] = [1.0, 0.0, 1.0]
        cases["wrong_hidden_content"] = corrupted
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.data / "dataset_audit.json").read_text())
    scorer = LayeredGroupingRewardModel(
        {
            "integrity_weight": 1,
            "grouping_mode": "partition",
            "integrity_mode": "reference",
            "completion_weight": 0,
        }
    )
    records, failures = [], []
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
        scores, errors = {}, {}
        for name, rgba in counterexamples(ref.truth).items():
            stack = LayerStack.from_rgba(rgba)
            scores[name] = scorer.score(stack, ref)
            errors[name] = float(np.max(np.abs(stack.composite() - ref.truth.composite())))
        totals = {name: value["layered_grouping"] for name, value in scores.items()}
        checks = {
            "same_composite": max(errors.values()) < 1e-5,
            "empty_neutral": abs(totals["correct"] - totals["legal_empty_slots"]) < 1e-6,
            "split_over_dropped_fragment": totals["split"] > totals["fragment_in_background"],
            "all_failures_below_correct": all(
                totals[k] < totals["correct"] - 1e-6
                for k in scores
                if k not in ["correct", "legal_empty_slots"]
            ),
        }
        if not all(checks.values()):
            failures.append({"id": row["id"], "checks": checks})
        records.append(
            {"id": row["id"], "checks": checks, "scores": scores, "composite_errors": errors}
        )
    report = {
        "passed": not failures,
        "scenes": len(records),
        "scored_configurations": sum(len(r["scores"]) for r in records),
        "failures": failures,
        "records": records,
        "scope": "Controlled counterexamples, not proof of human preference agreement or learned improvement.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
