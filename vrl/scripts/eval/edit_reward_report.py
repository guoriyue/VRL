"""Summarize rewards against score-blind review and render a standalone gallery."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image


def selection_metrics(cases: list[dict], scores: dict, labels: dict) -> dict | None:
    names = [case["name"] for case in cases if case["name"] in scores]
    if len(names) != len(cases) or not names:
        return None
    known = {name: labels.get(name, {}).get("joint_success") for name in names}
    # Callers order original candidates before retries and use low-seed tie breaking.
    best = max(names, key=lambda name: scores[name])
    pairs = [
        (good, bad)
        for good in names
        if known[good] is True
        for bad in names
        if known[bad] is False
    ]
    wins = sum(
        1 if scores[good] > scores[bad] else 0.5 if scores[good] == scores[bad] else 0
        for good, bad in pairs
    )
    return {
        "name": best,
        "success": known[best],
        "score": scores[best],
        "tied_winners": [name for name in names if scores[name] == scores[best]],
        "correct_pair_credit": wins,
        "comparable_pairs": len(pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/qwen_image_21_reward_study")
    args = parser.parse_args()
    out = Path(args.out)
    report = json.loads((out / "report.json").read_text())
    tasks = json.loads((out / "tasks.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    labels = {}
    for path in sorted(out.glob("visual_labels_*.json")):
        labels.update(json.loads(path.read_text())["labels"])
    rewards = {}
    for reward in ["editscore", "editreward", "editscore_task_crop"]:
        path = out / f"{reward}_seed42.jsonl"
        rewards[reward] = (
            {row["name"]: row["score"] for row in map(json.loads, path.read_text().splitlines())}
            if path.exists()
            else {}
        )
    grounded_report = json.loads((out / "grounded_scores.json").read_text())
    grounded = grounded_report["candidates"]
    rewards["grounded_editscore"] = {
        row["name"]: row["grounded_editscore"] for row in grounded if "grounded_editscore" in row
    }
    rows = []
    for task in tasks:
        cases = [case for case in report["cases"].values() if case["task_id"] == task["name"]]
        if not cases:
            continue
        cases.sort(key=lambda case: case["seed"])
        known = {case["name"]: labels.get(case["name"], {}).get("joint_success") for case in cases}
        row = {
            "task_id": task["name"],
            "split": task["split"],
            "retry": "evaluation_task_id" in task,
            "candidates": len(cases),
            "reviewed_successes": sum(value is True for value in known.values()),
            "unknown_labels": sum(value is None for value in known.values()),
            "requested_edit_successes": sum(
                labels.get(case["name"], {}).get("edit_success") is True for case in cases
            ),
            "protected_region_passes": sum(
                labels.get(case["name"], {}).get("protected_parts_pass") is True for case in cases
            ),
            "selection": {},
        }
        for reward, scores in rewards.items():
            selection = selection_metrics(cases, scores, labels)
            if selection is not None:
                row["selection"][reward] = selection
        row["seed0_success"] = known[cases[0]["name"]]
        rows.append(row)
    summary = {
        "candidates": len(report["cases"]),
        "color_rule_version": grounded_report["rule_version"],
        "evaluation_caveat": protocol.get("annotation_correction"),
        "label_source": "Assistant visual review, not independent human preference labels.",
        "tie_policy": "Lowest seed; all tied winners reported. Pairwise ties receive half credit.",
        "tasks": rows,
    }
    for split in ["development", "heldout"]:
        subset = [row for row in rows if row["split"] == split and not row["retry"]]
        aggregate = {}
        for reward in rewards:
            scored = [row["selection"][reward] for row in subset if reward in row["selection"]]
            eligible = [row for row in scored if row["success"] is not None]
            aggregate[reward] = {
                "top1_successes": sum(row["success"] for row in eligible),
                "top1_known_tasks": len(eligible),
                "pair_credit": sum(row["correct_pair_credit"] for row in scored),
                "comparable_pairs": sum(row["comparable_pairs"] for row in scored),
            }
        summary[split] = aggregate
    summary["retry_pools"] = []
    for task in tasks:
        parent = task.get("evaluation_task_id")
        if not parent:
            continue
        cases = [
            case for case in report["cases"].values() if case["task_id"] in [parent, task["name"]]
        ]
        cases.sort(key=lambda case: (case["task_id"] != parent, case["seed"]))
        selection = {}
        for reward, scores in rewards.items():
            metrics = selection_metrics(cases, scores, labels)
            if metrics is not None:
                selection[reward] = metrics
        summary["retry_pools"].append(
            {
                "original_task": parent,
                "retry_task": task["name"],
                "candidates": len(cases),
                "note": "Development-only pool with extra generation and feedback; not a heldout policy-training result.",
                "selection": selection,
            }
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def picture(path: Path) -> str:
        rgba = Image.open(path).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        im.thumbnail((750, 1000))
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Qwen editing reward baseline</title><style>body{font:16px system-ui;background:#f6f7fa;color:#182233;margin:28px}h1{font-size:28px}.grid{display:grid;grid-template-columns:repeat(5,minmax(190px,1fr));gap:12px;overflow:auto}.card{background:white;padding:12px;border-radius:10px}.card img{width:100%;cursor:zoom-in}section{margin:40px 0}p{line-height:1.5}pre{white-space:pre-wrap;font-size:13px}.bad{color:#a42525}.good{color:#126232}summary{cursor:pointer}dialog{max-width:94vw;max-height:94vh}dialog img{max-height:88vh;max-width:90vw}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd3df}</style><h1>Real editing candidates: EditScore / EditReward baseline</h1><p>Images are natural outputs, without synthetic corruption or no-op requests.</p><p>Review labels are assistant visual judgments, not independent human annotations. Color checks use source-defined sparse patches; they are evidence of specific violations, not complete correctness certificates. Retry outputs have extra generation cost and are evaluated against the original request.</p>'
    ]
    page.append(
        f"<p>{html.escape(report['model'])} · resolution budget {report['settings']['resolution']} · {report['settings']['steps']} steps · no model training.</p>"
        "<p>EditScore: 0-10. EditReward: raw unbounded score, compare within a request. Grounded EditScore: 0-1.</p>"
    )
    if summary["evaluation_caveat"]:
        page.append("<p>" + html.escape(summary["evaluation_caveat"]) + "</p>")
    preview = out / "feedback_improvement.jpg"
    if preview.exists():
        preview_data = base64.b64encode(preview.read_bytes()).decode()
        page.append(
            "<h2>Feedback repair: same photograph and seed</h2>"
            f'<img style="width:100%;max-width:1800px" src="data:image/jpeg;base64,{preview_data}">'
        )
    page.append(
        "<details><summary>Measured summary and tie handling</summary><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre></details>"
    )
    for task in tasks:
        cases = [case for case in report["cases"].values() if case["task_id"] == task["name"]]
        if not cases:
            continue
        page.append(
            f"<section><h2>{html.escape(task['name'])} · {task['split']}</h2><p>{html.escape(task.get('evaluation_prompt', task['prompt']))}</p>"
        )
        if task.get("evaluation_prompt"):
            page.append(
                "<details><summary>Feedback retry generation prompt</summary><p>"
                + html.escape(task["prompt"])
                + "</p></details>"
            )
        source = out / f"{Path(task['source']).stem}_input.png"
        page.append(
            f'<div class="grid"><div class="card"><b>Original</b><img src="{picture(source)}"></div>'
        )
        for case in sorted(cases, key=lambda item: item["seed"]):
            name = case["name"]
            label = labels.get(name, {})
            success = label.get("joint_success")
            status = (
                "Pass"
                if success is True
                else "Fail"
                if success is False
                else "Uncertain / pending"
            )
            color = "good" if success is True else "bad" if success is False else ""
            scores = "<br>".join(
                f"{key}: {values[name]:.3f}" for key, values in rewards.items() if name in values
            )
            page.append(
                f'<div class="card"><b>Seed {case["seed"]}</b><img src="{picture(out / case["output"])}"><p class="{color}">{status}</p><p>{html.escape(label.get("evidence", "Not yet reviewed."))}</p><p>{scores}</p></div>'
            )
        page.append("</div></section>")
    page.append(
        '<dialog id="zoom"><img></dialog><script>document.querySelectorAll(".card img").forEach(im=>im.onclick=()=>{zoom.querySelector("img").src=im.src;zoom.showModal()});zoom.onclick=()=>zoom.close()</script></html>'
    )
    (out / "index.html").write_text("\n".join(page))
    print(
        json.dumps(
            {key: summary[key] for key in ["candidates", "development", "heldout"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
