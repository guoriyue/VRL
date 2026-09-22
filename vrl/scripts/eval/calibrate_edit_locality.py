"""Calibrate bounded EditReward influence using real localized-edit failures.

Fit on development seeds 0/2; audit seeds 1/3 and the previous RL pilot.
The prompt-feedback positives calibrate the verifier only, never the RL prompt.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from PIL import Image

from vrl.rewards.models.color_locality import color_locality, combine_locality
from vrl.utils.artifacts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/qwen_image_21_reward_study")
    )
    parser.add_argument("--previous", type=Path, default=Path("outputs/qwen_image_21_edit_rl"))
    parser.add_argument(
        "--config-out", type=Path, default=Path("manifests/edit_locality/locality_reward.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/qwen_image_21_locality_rl/calibration.json")
    )
    args = parser.parse_args()
    rules = json.loads((args.baseline / "region_checks.json").read_text())
    training = [
        json.loads(line)
        for line in Path("manifests/edit_locality/train_localized.jsonl").read_text().splitlines()
    ]
    config = {
        k: copy.deepcopy(v)
        for k, v in rules.items()
        if k not in {"tasks", "combination", "interpretation"}
    }
    config["tasks"] = {}
    for task in training:
        rule = copy.deepcopy(rules["tasks"][task["task_id"]])
        rule.update(
            prompt=task["prompt"],
            source_sha256=sha256_file(args.baseline / task["reference_image"]),
        )
        rule.pop("review_crop", None)
        rule.pop("crop_instruction", None)
        config["tasks"][task["task_id"]] = rule
    labels = json.loads((args.baseline / "visual_labels_development.json").read_text())["labels"]
    previous_labels = json.loads((args.previous / "visual_review.json").read_text())["pairs"]
    rows = []
    datasets = [
        ("baseline", args.baseline),
        ("base20", args.previous / "eval/base"),
        ("rl20", args.previous / "eval/rl20"),
    ]
    for dataset, root in datasets:
        report = json.loads((root / "report.json").read_text())
        quality = {
            row["name"]: row["score"]
            for row in map(json.loads, (root / "editreward_seed42.jsonl").read_text().splitlines())
        }
        for name, case in report["cases"].items():
            task_id = case.get("evaluation_task_id", case["task_id"])
            if task_id != "armchair_seat_blue":
                continue
            source_path = root / f"{Path(case['source']).stem}_input.png"
            with Image.open(source_path) as source, Image.open(root / case["output"]) as edited:
                evidence = color_locality(source, edited, config["tasks"][task_id], config)
            success = (
                labels[name]["joint_success"]
                if dataset == "baseline"
                else previous_labels[name][
                    "base_joint_success" if dataset == "base20" else "rl20_joint_success"
                ]
            )
            if not isinstance(success, bool):
                raise ValueError(f"Missing visual calibration label: {dataset}/{name}")
            split = "fit" if dataset == "baseline" and case["seed"] in {0, 2} else "audit"
            rows.append(
                {
                    "dataset": dataset,
                    "name": name,
                    "split": split,
                    "success": success,
                    "quality": quality[name],
                    "evidence": evidence,
                    "image_sha256": sha256_file(root / case["output"]),
                }
            )
    fit = [r for r in rows if r["split"] == "fit"]
    gaps = [
        good["evidence"]["locality"] - bad["evidence"]["locality"]
        for good in fit
        if good["success"]
        for bad in fit
        if not bad["success"]
    ]
    gap = min(gaps)
    if gap <= 0:
        raise ValueError("Color witnesses cannot separate calibration successes and failures")
    # tanh quality spans [-1, 1]. Even the worst possible quality reversal
    # can consume at most half the smallest observed fit locality margin.
    config["quality_weight"] = gap / 4
    config["calibration"] = {
        "fit_min_locality_gap": gap,
        "rule": "quality_weight = fit_min_locality_gap / 4; bounded quality range [-1,1]",
        "source_rules_sha256": sha256_file(args.baseline / "region_checks.json"),
    }
    for row in rows:
        row["scores"] = combine_locality(
            row["evidence"], row["quality"], quality_weight=config["quality_weight"]
        )
    checks = {}
    for split in ("fit", "audit"):
        group = [r for r in rows if r["split"] == split]
        margins = [
            good["scores"]["editreward_locality"] - bad["scores"]["editreward_locality"]
            for good in group
            if good["success"]
            for bad in group
            if not bad["success"]
        ]
        checks[split] = {
            "pairs": len(margins),
            "correct_pairs": sum(m > 0 for m in margins),
            "minimum_margin": min(margins),
        }
        if not all(m > 0 for m in margins):
            raise ValueError(f"Locality reward failed {split} ranking audit")
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": str(args.config_out),
        "quality_weight": config["quality_weight"],
        "checks": checks,
        "caveat": "Calibration/audit on previously inspected development photos, not independent downstream success evidence. Positive retries change prompts only to supply verifier calibration images; RL uses original prompts.",
        "candidates": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
