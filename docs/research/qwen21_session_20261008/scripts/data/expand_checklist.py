"""List the concrete elements each training prompt requires (metadata.judge_checklist).

usage: expand_checklist.py <in.jsonl> <out.jsonl>   (local Qwen3-VL-32B-FP8, text only, one GPU)
"""

import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
SYSTEM = (
    "You turn a text-to-image prompt into a checklist of the CONCRETE THINGS the picture must visibly contain. "
    "Output a JSON array of at most 4 short noun phrases, in this priority: the main subject; every other named "
    "object (vehicle, prop, animal, building, text to be written); every action or relation (riding, holding, "
    "fighting, standing on). Read idioms by their meaning ('jaguar print hoodie' -> 'a hoodie with a "
    "leopard-spot pattern'; 'riding a suzuki' -> 'a Suzuki motorcycle being ridden'; 'wet clay of Capitol Hill' "
    "-> 'the U.S. Capitol building made of clay'). NEVER list style, mood, quality, medium, lighting, color "
    "palette, artist names or adjectives like cute, chibi, kawaii, detailed, cinematic, 3D: those are not "
    "things. If the prompt names fewer than 4 things, list fewer. Never add anything the prompt does not ask "
    "for. Output the JSON array only."
)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    proc = AutoProcessor.from_pretrained(MODEL, revision="4bf2c2f3", local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        revision="4bf2c2f3",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    with open(sys.argv[2], "w") as out:
        for i, r in enumerate(rows):
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"]},
            ]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            intent = proc.batch_decode(
                ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0].strip()
            import re

            m = re.search(r"\[.*\]", intent, re.S)
            try:
                items = [str(x).strip() for x in json.loads(m.group(0))] if m else []
            except Exception:
                items = []
            r.setdefault("metadata", {})["judge_checklist"] = [x for x in items if x][:4]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            print(i, r["prompt"][:60], "->", r["metadata"]["judge_checklist"], flush=True)


main()
