"""Pair fixed-seed base/checkpoint edits, summarize rewards, and render a gallery."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--images-only", action="store_true", help="Render before viewing scores")
    parser.add_argument("--review", type=Path, help="Optional separately recorded visual review")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("manifests/edit_locality/train_localized.jsonl"),
    )
    args = parser.parse_args()
    training_tasks = {
        row["task_id"] for row in map(json.loads, args.train_manifest.read_text().splitlines())
    }
    roots = {"base": args.base, "trained": args.trained}
    reports = {
        label: json.loads((root / "report.json").read_text()) for label, root in roots.items()
    }
    base_cases, trained_cases = reports["base"]["cases"], reports["trained"]["cases"]
    if base_cases.keys() != trained_cases.keys() or not base_cases:
        raise ValueError("Both evaluation arms must contain the same non-empty candidate set")
    review = json.loads(args.review.read_text()) if args.review else {}
    if review and review["pairs"].keys() != base_cases.keys():
        raise ValueError("Visual review must describe the same candidate set")
    for key in (
        "seeds",
        "steps",
        "resolution",
        "config",
        "model_identity",
        "task_manifest_sha256",
        "source_hashes",
    ):
        if reports["base"]["settings"][key] != reports["trained"]["settings"][key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    scores = {}
    for label, root in roots.items():
        scores[label] = {}
        for reward in ("editreward", "editscore"):
            path = root / f"{reward}_seed42.jsonl"
            if path.exists() and not args.images_only:
                scores[label][reward] = {
                    row["name"]: row["score"]
                    for row in map(json.loads, path.read_text().splitlines())
                }
                if not base_cases.keys() <= scores[label][reward].keys():
                    raise ValueError(f"Incomplete reward file: {path}")
        color_path = root / "grounded_scores.json"
        if color_path.exists() and not args.images_only:
            colors = json.loads(color_path.read_text())["candidates"]
            for metric in (
                "target_color",
                "protected_color_preservation",
                "color_constraint_score",
            ):
                scores[label][metric] = {
                    row["name"]: row[metric] for row in colors if metric in row
                }
    args.out.mkdir(parents=True, exist_ok=True)
    rows, sections, previews = [], [], []
    for name, before in base_cases.items():
        after = trained_cases[name]
        for key in (
            "prompt",
            "source_sha256",
            "seed",
            "size",
            "task_id",
            "split",
            "initial_latents_sha256",
            "initial_latents_shape",
            "initial_latents_dtype",
        ):
            if before[key] != after[key]:
                raise ValueError(f"Unpaired input: {name}/{key}")
        images = [Image.open(before["source"]).convert("RGB")]
        images.extend(
            Image.open(roots[label] / reports[label]["cases"][name]["output"]).convert("RGB")
            for label in roots
        )
        arrays = [np.asarray(im, dtype=np.float32) for im in images[1:]]
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(f"Output shapes differ: {name}")
        row = {
            "name": name,
            "task_id": before["task_id"],
            "split": before["split"],
            "evaluation_group": (
                "training_instruction"
                if before["task_id"] in training_tasks
                else "heldout_source"
                if before["split"] == "heldout"
                else "new_instruction_training_source"
            ),
            "seed": before["seed"],
            "base_vs_trained_mean_absolute_pixel_change_255": float(
                np.abs(arrays[1] - arrays[0]).mean()
            ),
            "rewards": {},
        }
        for reward in scores["base"].keys() & scores["trained"].keys():
            if any(name not in scores[label][reward] for label in roots):
                continue
            a, b = (scores[label][reward][name] for label in roots)
            row["rewards"][reward] = {"base": a, "trained": b, "delta": b - a}
        rows.append(row)
        visual_note = review.get("pairs", {}).get(name)
        if visual_note is not None:
            row["visual_review"] = visual_note
        width = 320
        height = max(round(im.height * min(width / im.width, 480 / im.height)) for im in images)
        preview = Image.new("RGB", (width * 3, height + 52), "white")
        draw = ImageDraw.Draw(preview)
        draw.text((8, 5), name, fill="black")
        for index, (label, im) in enumerate(zip(("Source", "Base", "RL"), images, strict=True)):
            im.thumbnail((width, height))
            preview.paste(im, (index * width + (width - im.width) // 2, 52))
            draw.text((index * width + 8, 29), label, fill="black")
        preview.save(args.out / f"{name}.jpg", quality=92)
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        sections.append(
            f"<article><h2>{html.escape(name)} ({before['split']})</h2>"
            f"<p>{html.escape(before['prompt'])}</p>"
            f'<img src="data:image/jpeg;base64,{encoded}" alt="Source, base, RL comparison">'
            + (
                f"<p><b>Visual review:</b> {html.escape(visual_note['note'])}</p>"
                if visual_note
                else ""
            )
            + f"<pre>{html.escape(json.dumps(row['rewards'], indent=2))}</pre></article>"
        )
        if before["seed"] == min(reports["base"]["settings"]["seeds"]):
            previews.append(preview)
    summary = {}
    for split in sorted({row["evaluation_group"] for row in rows}):
        group = [row for row in rows if row["evaluation_group"] == split]
        summary[split] = {"pairs": len(group), "rewards": {}}
        for reward in scores["base"].keys() & scores["trained"].keys():
            values = [row["rewards"][reward] for row in group if reward in row["rewards"]]
            if not values:
                continue
            summary[split]["rewards"][reward] = {
                key: statistics.mean(value[key] for value in values)
                for key in ("base", "trained", "delta")
            }
            summary[split]["rewards"][reward]["positive_delta_pairs"] = sum(
                value["delta"] > 0 for value in values
            )
            summary[split]["rewards"][reward]["pairs"] = len(values)
    result = {
        "arms": {label: str(root.resolve()) for label, root in roots.items()},
        "caveat": "Reward differences are not independent visual or human preference wins.",
        "review_source": review.get("review_source"),
        "summary": summary,
        "pairs": rows,
    }
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Localized editing RL comparison</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px}"
        "img{width:100%}article{border-top:1px solid #aaa;margin-top:30px}pre{white-space:pre-wrap}"
        "</style><h1>Fixed-instruction, same-seed RL comparison</h1>"
        "<p>Columns: source / base / trained LoRA. Higher optimization reward alone "
        "does not establish better edits. Inspect requested and protected regions.</p>"
        + (f"<p><b>{html.escape(review['summary'])}</b></p>" if review.get("summary") else "")
        + (f"<p>{html.escape(review['review_source'])}</p>" if review.get("review_source") else "")
        + f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
        + "".join(sections)
    )
    if previews:
        sheet = Image.new("RGB", (previews[0].width, sum(im.height for im in previews)), "white")
        offset = 0
        for preview in previews:
            sheet.paste(preview, (0, offset))
            offset += preview.height
        sheet.save(args.out / "overview.jpg", quality=92)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
