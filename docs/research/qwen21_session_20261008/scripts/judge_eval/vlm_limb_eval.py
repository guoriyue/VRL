"""Held-out limb evaluation with Qwen3-VL-32B: P(limb defect) and P(exactly N people) per image.

usage: vlm_limb_eval.py out.json eval_s0.jsonl,eval_s1.jsonl,eval_s2.jsonl label=/dir/{s}/images/<label> ...
The {s} placeholder is the shard index; files are prompt%04d_sample%02d.png within each shard dir.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
WORDS = {2: "two", 3: "three", 4: "four", 6: "six", 8: "eight"}
Q_LIMB = (
    "Look at this image. Does any person in it have extra, missing, duplicated or merged limbs, hands, "
    "fingers, legs, heads or eyes, a torso fused with another person's body, or a body part attached in an "
    "impossible place? Answer with a single word: yes or no."
)
Q_COUNT = "Look at this image. Are there exactly {n} ({w}) people in it? Answer with a single word: yes or no."


def main():
    shards = [[json.loads(line) for line in open(p)] for p in sys.argv[2].split(",")]
    proc = AutoProcessor.from_pretrained(MODEL, revision="4bf2c2f3", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision="4bf2c2f3", local_files_only=True, dtype="auto", device_map="cuda:0"
    ).eval()
    tok = proc.tokenizer
    yes = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")}
    no = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")}

    def ask(im, q):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im}, {"type": "text", "text": q}],
            }
        ]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits[0, -1].float(), -1)
        py, pn = float(sum(p[i] for i in yes)), float(sum(p[i] for i in no))
        return py / max(py + pn, 1e-9)

    out = {}
    for spec in sys.argv[3:]:
        label, pattern = spec.split("=", 1)
        rows = []
        for s, shard in enumerate(shards):
            d = Path(pattern.format(s=s))
            for i, r in enumerate(shard):
                n = r["metadata"]["expected_people"]
                for f in sorted(d.glob(f"prompt{i:04d}_sample*.png")):
                    im = Image.open(f).convert("RGB")
                    im.thumbnail((896, 896))
                    rows.append(
                        {
                            "file": str(f),
                            "id": r["metadata"]["id"],
                            "style": r["metadata"]["style"],
                            "defect": ask(im, Q_LIMB),
                            "count_ok": ask(im, Q_COUNT.format(n=n, w=WORDS[n])),
                        }
                    )
        k = max(len(rows), 1)
        summary = {
            "images": len(rows),
            "defect_rate": sum(r["defect"] > 0.5 for r in rows) / k,
            "defect_mean": sum(r["defect"] for r in rows) / k,
            "count_ok_rate": sum(r["count_ok"] > 0.5 for r in rows) / k,
        }
        out[label] = {"summary": summary, "rows": rows}
        print(label, json.dumps(summary), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


main()
