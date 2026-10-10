"""Export blinded review without exposing score/arm mapping to reviewers."""

import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from vrl.rewards.evaluation import Evaluation

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("outputs/qwen21_aesthetic_qualification"))
a = p.parse_args()
root = a.root
review = root / "blind"
review.mkdir(exist_ok=True)
e = Evaluation.load(root / "scores")
if len(e.records) != 512 or any(r["status"] != "success" for r in e.records.values()):
    raise ValueError("All 512 successful scores required")
by_prompt = {}
for sample_id, r in sorted(e.records.items()):
    by_prompt.setdefault(r["input"]["prompt_id"], []).append((sample_id, r))
selection = random.Random(2026092501)
sides = random.Random(2026092502)
public = []
private = []
for pid, candidates in sorted(by_prompt.items()):
    candidates = sorted(candidates)
    baseline = selection.choice(candidates)
    best = max(candidates, key=lambda x: x[1]["result"]["scores"]["aesthetic"])
    arms = [("selected", best), ("random", baseline)]
    sides.shuffle(arms)
    case = {"id": pid, "prompt": candidates[0][1]["input"]["prompt"], "images": {}}
    mapping = {"id": pid, "identical": best[0] == baseline[0], "arms": {}}
    canvas = Image.new("RGB", (1024, 548), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, (sid, r)) in enumerate(arms):
        side = "AB"[i]
        image = Image.open(r["input"]["path"]).convert("RGB")
        image.save(review / f"{pid}_{side}.png")
        canvas.paste(image.resize((512, 512)), (512 * i, 36))
        draw.text((512 * i + 12, 12), f"{pid} | {side}", fill="black")
        case["images"][side] = f"{pid}_{side}.png"
        mapping["arms"][side] = {
            "arm": label,
            "sample_id": sid,
            "aesthetic": r["result"]["scores"]["aesthetic"],
        }
    canvas.save(review / f"{pid}_pair.jpg", quality=95)
    public.append(case)
    private.append(mapping)
(review / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in public))
(root / "selection_key.jsonl").write_text("".join(json.dumps(r) + "\n" for r in private))
page = [
    '<!doctype html><meta charset="utf-8"><title>Blind Qwen aesthetic comparisons</title><style>body{font:18px system-ui;max-width:1100px;margin:auto}img{width:100%}section{margin:40px 0}</style>',
    "<h1>Blind aesthetic comparisons</h1><p>Judge A/B/tie for visual appeal, then rate each image’s prompt adherence separately (0 wrong, 1 major misses, 2 minor misses, 3 fully meets). Scores and selection identities are withheld. These are training-sampler images (512 px, 10 steps), not deployment-quality claims.</p>",
]
for r in public:
    page.append(
        f'<section><h2>{r["id"]}</h2><p>{html.escape(r["prompt"])}</p><img src="{r["id"]}_pair.jpg"></section>'
    )
(review / "index.html").write_text("\n".join(page))
print("Exported 64 blinded comparisons; private selection key withheld.")
