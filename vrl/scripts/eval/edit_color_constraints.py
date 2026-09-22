"""Add fixed source-region color evidence to EditScore; never reads review labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def membership(pixels: np.ndarray, box: list[float], *, hue: float, config: dict) -> float:
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = [
        round(v * n) for v, n in zip(box, [width, height, width, height], strict=True)
    ]
    patch = pixels[y0:y1, x0:x1]
    if not patch.size:
        raise ValueError(f"Empty witness patch: {box}")
    hsv = np.asarray(
        Image.fromarray((patch * 255).round().astype(np.uint8)).convert("HSV"), dtype=np.float32
    )
    hsv[..., 0] *= 360.0 / 255.0
    hsv[..., 1:] /= 255.0
    difference = np.abs(hsv[..., 0] - hue)
    difference = np.minimum(difference, 360 - difference)
    hue_score = np.exp(-0.5 * (difference / config["hue_sigma_degrees"]) ** 2)
    saturation = np.clip(
        (hsv[..., 1] - config["saturation_start"])
        / (config["saturation_full"] - config["saturation_start"]),
        0,
        1,
    )
    return float((hue_score * saturation).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/qwen_image_21_reward_study")
    parser.add_argument("--rules", default="region_checks.json")
    parser.add_argument("--output", default="grounded_scores.json")
    args = parser.parse_args()
    out = Path(args.out)
    config = json.loads((out / args.rules).read_text())
    report = json.loads((out / "report.json").read_text())
    score_file = out / "editscore_seed42.jsonl"
    originals = (
        {row["name"]: row for row in map(json.loads, score_file.read_text().splitlines())}
        if score_file.exists()
        else {}
    )
    rows = []
    for name, case in report["cases"].items():
        rule = config["tasks"].get(case.get("evaluation_task_id", case["task_id"]))
        row = {
            "name": name,
            "task_id": case["task_id"],
            "split": case["split"],
            "has_color_verifier": rule is not None,
        }
        base = originals.get(name)
        if base is not None:
            if not 0 <= base["score"] <= 10:
                raise ValueError(f"Invalid upstream score: {base}")
            row["original_editscore"] = base["score"] / 10
        if rule is not None:
            source = rgb_pixels(out / f"{Path(case['source']).stem}_input.png")
            edited = rgb_pixels(out / case["output"])
            if source.shape != edited.shape:
                raise ValueError(f"Source/output geometry differs: {name}")
            hue = config["color_hue_degrees"][rule["color"]]
            target = float(
                np.mean([membership(edited, b, hue=hue, config=config) for b in rule["target"]])
            )
            protected = {}
            for label, box in rule["protected"].items():
                before = membership(source, box, hue=hue, config=config)
                after = membership(edited, box, hue=hue, config=config)
                protected[label] = {
                    "source_target_color": before,
                    "output_target_color": after,
                    "added_target_color": max(0, after - before),
                }
            preservation = 1 - max(v["added_target_color"] for v in protected.values())
            row.update(
                target_color=target,
                protected_color_preservation=preservation,
                protected_patches=protected,
                color_constraint_score=min(target, preservation),
            )
        if base is not None:
            row["grounded_editscore"] = min(
                row["original_editscore"], row.get("color_constraint_score", 1.0)
            )
        rows.append(row)
    (out / args.output).write_text(
        json.dumps({"rule_version": config["version"], "candidates": rows}, indent=2) + "\n"
    )
    print(f"Scored {len(rows)} candidates; {len(originals)} have EditScore baselines.")


if __name__ == "__main__":
    main()
