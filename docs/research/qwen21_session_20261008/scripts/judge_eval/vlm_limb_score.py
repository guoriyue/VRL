"""Qwen3-VL-32B limb-defect check per image: P(yes) that a figure has extra/missing/misplaced limbs.

usage: vlm_limb_score.py out.json [--variant full896|flip768] /dir/with/pngs [/dir ...]
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
Q = (
    "Look at this image. Does any person, animal or creature in it have extra, missing, duplicated or merged "
    "limbs, hands, fingers, legs, heads or eyes, or a body part attached in an impossible place? "
    "Answer with a single word: yes or no."
)


def main():
    args = sys.argv[2:]
    variant = "full896"
    if args and args[0] == "--variant":
        variant, args = args[1], args[2:]
    files = [p for d in args for p in sorted(Path(d).glob("*.png"))]
    proc = AutoProcessor.from_pretrained(MODEL, revision="4bf2c2f3", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision="4bf2c2f3", local_files_only=True, dtype="auto", device_map="cuda:0"
    ).eval()
    tok = proc.tokenizer
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")}
    out = {}
    for f in files:
        im = Image.open(f).convert("RGB")
        if variant == "flip768":
            from PIL import ImageOps

            im = ImageOps.mirror(im)
            im.thumbnail((768, 768))
        else:
            im.thumbnail((896, 896))
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im}, {"type": "text", "text": Q}],
            }
        ]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits[0, -1].float(), -1)
        py, pn = float(sum(p[i] for i in yes_ids)), float(sum(p[i] for i in no_ids))
        out[str(f)] = py / max(py + pn, 1e-9)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("done", len(out))


main()
