"""Held-out action fidelity with Qwen3-VL-32B: P(the image shows the prompt's close-contact action).

usage: vlm_action_eval.py out.json label=ck:/mnt/nvme/eval/limb_ck{ck}_s{s}/images/<label> ...
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
Q = (
    'This image was generated for the description: "{prompt}"\n'
    "Ignore drawing quality. Do the people in the image actually perform the described action and physical "
    "interaction, in close physical contact as described (not merely standing near each other or side by side)? "
    "Answer with a single word: yes or no."
)


def main():
    shards = [
        [json.loads(line) for line in open(f"/mnt/nvme/limb/corpus2/eval_all_s{s}.jsonl")]
        for s in range(3)
    ]
    proc = AutoProcessor.from_pretrained(MODEL, revision="4bf2c2f3", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision="4bf2c2f3", local_files_only=True, dtype="auto", device_map="cuda:0"
    ).eval()
    tok = proc.tokenizer
    yes = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")}
    no = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")}
    out = {}
    for spec in sys.argv[2:]:
        label, pattern = spec.split("=", 1)
        rows = []
        for s, shard in enumerate(shards):
            for i, r in enumerate(shard):
                f = Path(pattern.format(s=s)) / f"prompt{i:04d}_sample00.png"
                im = Image.open(f).convert("RGB")
                im.thumbnail((896, 896))
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": im},
                            {"type": "text", "text": Q.format(prompt=r["prompt"])},
                        ],
                    }
                ]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], images=[im], return_tensors="pt").to("cuda:0")
                with torch.no_grad():
                    p = torch.softmax(model(**inputs).logits[0, -1].float(), -1)
                py, pn = float(sum(p[k] for k in yes)), float(sum(p[k] for k in no))
                rows.append(
                    {
                        "s": s,
                        "i": i,
                        "n": r["metadata"]["expected_people"],
                        "action": py / max(py + pn, 1e-9),
                    }
                )
        out[label] = rows
        by = {}
        for r in rows:
            by.setdefault(r["n"], []).append(r["action"] > 0.5)
        print(
            label,
            f"action_ok {sum(r['action'] > 0.5 for r in rows) / len(rows):.0%}",
            {n: f"{sum(v) / len(v):.0%}" for n, v in sorted(by.items())},
            flush=True,
        )
    json.dump(out, open(sys.argv[1], "w"), indent=1)


main()
