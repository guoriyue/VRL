"""Summarize completed layered evaluations without treating fidelity as editability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_records(records):
    """Reject mismatched sampling conditions before calculating paired deltas."""
    if len(records) < 2:
        return {}
    first = next(iter(records.values()))
    for arm, record in records.items():
        for key in (
            "case",
            "split",
            "seed",
            "sampler",
            "steps",
            "layers",
            "condition_sha256",
            "upstream_sha256",
        ):
            if record[key] != first[key]:
                raise ValueError(f"Unmatched evaluation condition {key}: {arm}")
    if "base" not in records or "tuned" not in records:
        return {}
    base = records["base"]["diagnostics"]
    tuned = records["tuned"]["diagnostics"]
    return {
        "reconstruction_mae_change": tuned["rgb_mae_over_black"] - base["rgb_mae_over_black"],
        "composite_under_opaque_pixels_change": tuned["composite_under_opaque_pixels"]
        - base["composite_under_opaque_pixels"],
        "semantic_improvement": "not established by these measurements",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    for case in plan["cases"]:
        records = {}
        views = {}
        for arm in ("base", "stable", "tuned"):
            directory = root / "evaluation" / case["name"] / arm / f"seed{args.seed}"
            path = directory / "record.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text())
            assert record["case"] == case["name"] and record["arm"] == arm
            assert record["split"] == case["split"]
            for index in range(record["layers"]):
                assert (directory / f"layer_{index}.png").is_file()
            assert (directory / "review/review.png").is_file()
            records[arm] = record
            views[arm] = str((directory / "review/review.png").relative_to(root))
        rows.append(
            {
                "case": case["name"],
                "split": case["split"],
                "arms": records,
                "missing_arms": sorted({"base", "stable", "tuned"} - records.keys()),
                "paired_deltas": compare_records(records),
                "views": views,
            }
        )
    report = {
        "seed": args.seed,
        "rows": rows,
        "all_arms_complete": all(not row["missing_arms"] for row in rows),
        "scope": "Paired reconstruction diagnostics only. Inspect complete layers and hide/move views for object separation, fragmentation and plausible hidden content. No quality improvement or statistical significance inferred.",
    }
    (root / "paired_report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "complete_outputs": sum(len(row["arms"]) for row in rows),
                "expected_outputs": len(rows) * 3,
                "matched_base_tuned_cases": [row["case"] for row in rows if row["paired_deltas"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
