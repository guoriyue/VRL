"""Compute descriptive, within-scene paired changes from the completed grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

METRICS = (
    "layered_grouping",
    "grouping",
    "completion",
    "fidelity",
    "object_background_share",
    "layers_with_majority_of_multiple_objects",
    "objects_with_iou_at_least_0_5",
    "active_foreground_layers",
    "hidden_rgb_mae",
    "hidden_alpha_mae",
    "hidden_alpha_recall",
    "cat_best_layer_coverage",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "summary.json").read_text())
    assert report["complete"] and report["result_count"] == 72
    rows = report["rows"]
    lookup = {(r["arm"], r["id"], r["extra_layers"], r["sampling_steps"]): r for r in rows}
    pairs = []
    for a in rows:
        comparisons = []
        if a["sampling_steps"] == 10:
            comparisons.append(("steps_10_to_30", (a["arm"], a["id"], a["extra_layers"], 30)))
        if a["extra_layers"] == 0:
            for extra in (2, 4):
                comparisons.append(
                    (f"layers_plus_{extra}", (a["arm"], a["id"], extra, a["sampling_steps"]))
                )
        for src, dst in (("v3_seg3", "v3c_28"), ("v3_seg3", "v3e_44"), ("v3c_28", "v3e_44")):
            if a["arm"] == src:
                comparisons.append(
                    (f"{src}_to_{dst}", (dst, a["id"], a["extra_layers"], a["sampling_steps"]))
                )
        for name, key in comparisons:
            b = lookup[key]
            pairs.append(
                {
                    "comparison": name,
                    "id": a["id"],
                    "kind": "rendered" if a["id"].startswith("rendered") else "reference_image",
                    "arm_a": a["arm"],
                    "arm_b": b["arm"],
                    "extra_layers_a": a["extra_layers"],
                    "extra_layers_b": b["extra_layers"],
                    "steps_a": a["sampling_steps"],
                    "steps_b": b["sampling_steps"],
                    "delta_b_minus_a": {
                        m: b[m] - a[m]
                        for m in METRICS
                        if a.get(m) is not None and b.get(m) is not None
                    },
                }
            )
    groups = []
    for name in sorted({p["comparison"] for p in pairs}):
        for kind in ("rendered", "reference_image"):
            for arm in sorted({p["arm_a"] for p in pairs if p["comparison"] == name}):
                selected = [
                    p
                    for p in pairs
                    if p["comparison"] == name and p["kind"] == kind and p["arm_a"] == arm
                ]
                metrics = {}
                for metric in METRICS:
                    values = [
                        p["delta_b_minus_a"][metric]
                        for p in selected
                        if metric in p["delta_b_minus_a"]
                    ]
                    if values:
                        metrics[metric] = {
                            "n": len(values),
                            "mean": mean(values),
                            "median": median(values),
                            "increases": sum(v > 1e-8 for v in values),
                            "decreases": sum(v < -1e-8 for v in values),
                        }
                groups.append({"comparison": name, "kind": kind, "arm_a": arm, "metrics": metrics})
    output = {
        "note": "Descriptive differences only. Settings on a shared scene are not independent samples; no significance or population claim.",
        "pairs": pairs,
        "groups": groups,
    }
    (args.root / "paired_effects.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"pairs": len(pairs), "groups": len(groups)}))


if __name__ == "__main__":
    main()
