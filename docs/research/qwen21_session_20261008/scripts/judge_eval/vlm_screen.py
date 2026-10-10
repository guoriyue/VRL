"""Qwen3-VL-32B yes/no screen per image: apparent age under 18, and nudity.

usage: vlm_screen.py out.json /dir/with/pngs [/dir ...]
Writes {file: {"minor": P(yes), "nude": P(yes)}} for every png under the dirs.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
QUESTIONS = {
    "minor": (
        "Look at this image. Does any person or character in it look younger than 18 years old, "
        "for example a child, a young teenager, or someone with a childlike face or body? "
        "Answer with a single word: yes or no."
    ),
    "nude": (
        "Look at this image. Does it show exposed (uncovered) female breasts or nipples, exposed genitals, "
        "or exposed buttocks of any person, statue or painted figure? A bare male chest does not count. "
        "Answer with a single word: yes or no."
    ),
}


def main():
    files = [p for d in sys.argv[2:] for p in sorted(Path(d).rglob("*.png"))]
    proc = AutoProcessor.from_pretrained(MODEL, revision="4bf2c2f3", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision="4bf2c2f3", local_files_only=True, dtype="auto", device_map="cuda:0"
    ).eval()
    tok = proc.tokenizer
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")}
    out = {}
    for n, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((896, 896))
        res = {}
        for key, q in QUESTIONS.items():
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
            py, pn = float(sum(p[i] for i in yes_ids)), float(sum(p[i] for i in no_ids))
            res[key] = py / max(py + pn, 1e-9)
        out[str(f)] = res
        if n % 50 == 0:
            print(n, len(files), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("done", len(out))


main()
