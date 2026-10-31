"""Compare the repaired reward with an immutable audit and prepare corrected budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.layering_reward_audit import synthetic
from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.audit / "reward_audit.json").read_text())
    data = json.loads((args.audit / "data_audit.json").read_text())
    original_lines = args.manifest.read_text().splitlines()
    corrected = []
    changes = []
    for index, line in enumerate(original_lines):
        row = json.loads(line)
        meta = row["metadata"]
        stats = data["mask_stats"].get(meta.get("instances"))
        if stats is None:
            count = data["truth_stats"][meta["truth_dir"]]["visible_objects"]
        else:
            count = stats["objects"]
        before = row["request_overrides"]["layers"]
        after = max(before, count + 1)
        if after != before:
            row["request_overrides"]["layers"] = after
            changes.append({"row": index, "before": before, "after": after})
            line = json.dumps(row, ensure_ascii=False)
        corrected.append(line)
    fixed_manifest = args.manifest.with_name(args.manifest.stem + "_rewardfix.jsonl")
    fixed_manifest.write_text("\n".join(corrected) + "\n")

    plan = json.loads((args.diagnostics / "summary.json").read_text())["plan"]
    refs = {
        s["id"]: LayeredReference.from_metadata(
            {**s["metadata"], "reference_images": [str(Path(s["scene_dir"]) / "source.png")]},
            640,
            640,
        )
        for s in plan["scenes"]
    }
    model = LayeredGroupingRewardModel({})
    rows = []
    for baseline in old["real_outputs"]["rows"]:
        scene, arm = baseline["id"], baseline["arm"]
        budget, steps = baseline["requested_layers"], baseline["sampling_steps"]
        directory = args.diagnostics / "eval" / arm / scene / f"layers{budget}_steps{steps}"
        stack = LayerStack.from_directory(directory, 640, 640)
        ref = refs[scene]
        score = model.score(stack, ref)
        matches, _ = stack.match(ref.objects)
        labeled = np.logical_or.reduce(ref.objects)
        per_object = []
        for i, mask in enumerate(ref.objects):
            k = matches.get(i)
            inter = 0 if k is None else int((mask & stack.regions[k]).sum())
            area = 0 if k is None else int((stack.regions[k] & labeled).sum())
            per_object.append(
                {
                    "object": i,
                    "layer": k,
                    "coverage": inter / int(mask.sum()),
                    "purity_on_annotated_pixels": inter / area if area else 0.0,
                }
            )
        rows.append(
            {
                "scene": scene,
                "arm": arm,
                "layers": budget,
                "steps": steps,
                "old_reward": baseline["layered_grouping"],
                "new_reward": score["layered_grouping"],
                "old_grouping": baseline["grouping"],
                "new_grouping": score["grouping"],
                "old_completion": baseline["completion"],
                "new_completion": score["completion"],
                "fidelity_change": score["fidelity"] - baseline["fidelity"],
                "complete_pure_objects": sum(
                    p["coverage"] >= 0.9 and p["purity_on_annotated_pixels"] >= 0.9
                    for p in per_object
                ),
                "objects": len(per_object),
                "per_object": per_object,
            }
        )
    with (args.output / "comparison.csv").open("w") as handle:
        fields = [k for k in rows[0] if k != "per_object"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    controls = synthetic()
    checks = controls["acceptance_checks"]
    required = [k for k in checks if k != "separating_unannotated_object_is_rewarded"]
    assert all(checks[k] for k in required), checks
    assert all(abs(r["fidelity_change"]) < 1e-7 for r in rows)
    report = {
        "controlled_checks": checks,
        "controlled_scores": controls["scores"],
        "real_outputs": rows,
        "changed_manifest_rows": changes,
        "manifest": str(fixed_manifest),
        "original_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "corrected_manifest_sha256": hashlib.sha256(fixed_manifest.read_bytes()).hexdigest(),
        "reward_sha256": hashlib.sha256(
            Path("vrl/rewards/models/layered_grouping.py").read_bytes()
        ).hexdigest(),
        "limitations": [
            "Unannotated objects still have no grouping supervision.",
            "Strict coverage/purity counts are diagnostics, not independent human preferences.",
            "Removing fragments into background remains a split-versus-missing tradeoff.",
            "Global grouping assignment also changes completion's object-layer matching.",
            "Offline checks do not validate short-run RL gains or completion semantics.",
        ],
        "training_started": False,
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": checks, "outputs": len(rows), "budget_fixes": len(changes)}))


if __name__ == "__main__":
    main()
