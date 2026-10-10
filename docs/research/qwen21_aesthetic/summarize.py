"""Collect qualification evidence and render reviewable result artifacts."""

import html
import json
import statistics
from pathlib import Path

import numpy as np

root = Path("outputs/qwen21_aesthetic_qualification")
docs = Path("docs/research/qwen21_aesthetic")
read = lambda p: json.loads(p.read_text())
verdict = read(root / "blind_verdict.json")
keys = [json.loads(l) for l in (root / "selection_key.jsonl").read_text().splitlines()]
labels = [json.loads(l) for l in (root / "unblinded_labels.jsonl").read_text().splitlines()]
prompts = {
    r["id"]: r
    for r in map(json.loads, (docs / "qualification_prompts.jsonl").read_text().splitlines())
}
selected = []
random_scores = []
for row in keys:
    arms = {v["arm"]: v for v in row["arms"].values()}
    selected.append(arms["selected"]["aesthetic"])
    random_scores.append(arms["random"]["aesthetic"])
x = np.array(selected) - random_scores
rng = np.random.default_rng(2026092504)
reward_ci = np.quantile(
    x[rng.integers(0, len(x), (10000, len(x)))].mean(1), [0.025, 0.975]
).tolist()
category = []
for name in dict.fromkeys(p["category"] for p in prompts.values()):
    rows = [r for r in labels if prompts[r["id"]]["category"] == name]
    category.append(
        {
            "category": name,
            "n": len(rows),
            "wins": sum(r["utility"] == 1 for r in rows),
            "ties": sum(r["utility"] == 0.5 for r in rows),
            "losses": sum(r["utility"] == 0 for r in rows),
            "utility": statistics.mean(r["utility"] for r in rows),
            "adherence_delta": statistics.mean(r["adherence_delta"] for r in rows),
        }
    )
# Sensitivity: q00-s0 was viewed in a generation smoke check before blinding.
sens = np.array([r["utility"] for r in labels if r["id"] != "q00"])
sensitivity = {
    "n": len(sens),
    "utility": float(sens.mean()),
    "ci95": np.quantile(
        sens[rng.integers(0, len(sens), (10000, len(sens)))].mean(1), [0.025, 0.975]
    ).tolist(),
}
summary = {
    "decision": "DO_NOT_TRAIN_UNDER_TESTED_RECIPE",
    "interpretation": "Preference gate inconclusive, not proof of no possible benefit. No RL improvement established.",
    "user_gate": verdict,
    "aesthetic_scores": {
        "selected_mean": statistics.mean(selected),
        "random_mean": statistics.mean(random_scores),
        "paired_delta": float(x.mean()),
        "delta_ci95": reward_ci,
    },
    "categories": category,
    "exclude_smoke_prompt_sensitivity": sensitivity,
    "continuous_spread": {
        k: v for k, v in read(root / "continuous_spread.json").items() if k != "per_prompt"
    },
    "repeatability": {
        k: v for k, v in read(root / "repeat.json")["axes"]["aesthetic"].items() if k != "samples"
    },
    "stress": read(root / "stress.json")["transforms"],
    "environment": read(root / "environment.json"),
    "conditional_stages": {
        "rl_pilot": "not run: predeclared paired preference gate failed",
        "checkpoint_verdict_120": "not run: no trained checkpoints; 120 test prompts frozen and unused",
        "post_rl_diversity": "not measured: no RL performed",
    },
    "limitations": [
        "One AI visual judge, not human preference data.",
        "64 purposively diverse prompts, eight per category; category comparisons exploratory.",
        "512px/10-step SDE noise 0.7, not normal deployment sampling.",
        "Native-sampler diagnostic is four prompts, not a new qualification or efficacy test.",
        "Prompt overlap excluded exactly; no pretraining decontamination claim.",
        "q00-s0 was viewed for smoke testing before randomization; sensitivity excluding q00 is reported.",
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(docs / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
# Preserve independently authored labels with the small tracked report.
(docs / "blind_labels.jsonl").write_bytes((root / "blind/labels.jsonl").read_bytes())
# Unblinded comparisons with full provenance, after labels were locked.
page = [
    '<!doctype html><meta charset="utf-8"><title>Aesthetic reward qualification results</title><style>body{font:17px system-ui;max-width:1100px;margin:auto;padding:20px}img{width:100%}section{margin:36px 0}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}</style>',
    "<h1>Qwen-Image-2.1 aesthetic reward: qualification did not pass</h1>",
    "<p>64 prompts × 8 seeds. Independent AI visual review, not human ratings. No RL was launched.</p>",
    f"<p>Selected images: {verdict['wins']} wins, {verdict['losses']} losses, {verdict['ties']} identical ties. Preference utility {100 * verdict['utility']['mean']:.1f}% (95% CI {100 * verdict['utility']['ci95'][0]:.1f}–{100 * verdict['utility']['ci95'][1]:.1f}%). The interval includes chance.</p>",
    '<p>These are noisy training-sampler outputs (512px, 10 steps, noise 0.7). See <a href="sampler_diagnostic.html">native-sampler diagnostic</a> before interpreting the visible artifacts as base-model quality.</p>',
]
label_map = {r["id"]: r for r in labels}
for key in keys:
    pid = key["id"]
    l = label_map[pid]
    title = " | ".join(
        f"{side}: {v['arm']}, score {v['aesthetic']:.3f}" for side, v in key["arms"].items()
    )
    page.append(
        f'<section><h2>{pid}: {html.escape(title)}</h2><p>{html.escape(prompts[pid]["prompt"])}</p><img src="blind/{pid}_pair.jpg"><p>Blind appeal: {l["appeal"]}; adherence A={l["adherence_A"]}, B={l["adherence_B"]}. {html.escape(l["note"])}</p></section>'
    )
(root / "index.html").write_text("\n".join(page))
page = [
    '<!doctype html><meta charset="utf-8"><title>Sampler diagnostic</title><style>body{font:17px system-ui;padding:20px}img{width:32%;max-width:512px}</style><h1>Base-model sampler diagnostic</h1><p>Four fixed qualification prompts, seed s0. Columns: 512/10 SDE noise 0.7; 512/10 native; 1024/40 native. Same seed labels, not identical latent arrays across resolutions. These post-gate examples diagnose sampling quality only; they do not constitute a second preference experiment.</p>'
]
for pid in ["q00", "q16", "q40", "q56"]:
    page.append(f"<h2>{pid}</h2><p>{html.escape(prompts[pid]['prompt'])}</p>")
    for directory in ["generation", "diagnostic_512_10", "diagnostic_1024_40"]:
        image = root / directory / pid / "s0.png"
        if not image.exists():
            raise FileNotFoundError(image)
        page.append(f'<img src="{directory}/{pid}/s0.png" alt="{directory} {pid}">')
(root / "sampler_diagnostic.html").write_text("\n".join(page))
print(
    json.dumps(
        {"decision": summary["decision"], "categories": category, "sensitivity": sensitivity},
        indent=2,
    )
)
