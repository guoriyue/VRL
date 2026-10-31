"""Compare fixed photo sets using visible-instance diagnostics, without reward gates.

These measurements cannot establish hidden-object or hidden-copy correctness.
"""

import argparse
import json
from pathlib import Path

from tools.layering_natural_report import measure
from tools.layering_object_failure_report import measure_failures
from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack


def summarize(records):
    keys = (
        "objects",
        "complete_pure_objects",
        "merge_layers",
        "split_objects",
        "majority_background_objects",
    )
    return {
        "scenes": len(records),
        **{key: sum(record[key] for record in records) for key in keys},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4401)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    flagged = set()
    if args.review:
        flagged = set(json.loads(args.review.read_text())["flagged_scenes"])
        if not flagged <= {row["id"] for row in rows}:
            raise ValueError("Review includes scenes outside the evaluation manifest")
    arms = {}
    for name, directory in (("baseline", args.baseline), ("candidate", args.candidate)):
        records = []
        for row in rows:
            output = directory / row["id"] / f"seed{args.seed}"
            record = measure(output, row)
            reference = LayeredReference.from_metadata(
                {
                    **row["metadata"],
                    "reference_images": [str(Path(row["scene_dir"]) / "source.png")],
                },
                640,
                640,
            )
            failures = measure_failures(LayerStack.from_directory(output, 640, 640), reference)
            for key in ("split_objects", "majority_background_objects"):
                record[key] = failures[key]
            records.append(record)
        arms[name] = {
            "directory": str(directory),
            "records": records,
            "all": summarize(records),
            "unflagged": summarize([record for record in records if record["id"] not in flagged]),
        }
    deltas = {
        subset: {
            key: arms["candidate"][subset][key] - value
            for key, value in arms["baseline"][subset].items()
        }
        for subset in ("all", "unflagged")
    }
    report = {
        "arms": arms,
        "candidate_minus_baseline": deltas,
        "seed": args.seed,
        "limitations": (
            "Visible COCO instances only; completeness and purity use >=90% thresholds. "
            "Purity is relative to annotated objects and does not certify unannotated pixels. "
            "Split means two foreground layers each contain >=10% of one visible object. "
            "No claim about hidden duplication, occluded texture, or statistical significance. "
            "Unflagged subset is declared in the source-only visual review before generation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
