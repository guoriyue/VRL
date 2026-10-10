"""CPU-only, non-training preference cross-evaluation of fixed native samples."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/nvme/hf"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from huggingface_hub import HfApi
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from vrl.scripts.eval.score_report import write_curve_report

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
root = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=root)
parser.add_argument("--steps", nargs="+", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reuse", type=Path)
parser.add_argument("--prompts-per-shard", type=int, default=16)
args = parser.parse_args()
if args.prompts_per_shard < 1:
    raise ValueError("--prompts-per-shard must be positive")
prompt_count = 4 * args.prompts_per_shard
root = args.root
if len(set(args.steps)) != len(args.steps):
    raise ValueError("Duplicate evaluation steps")
out = args.output
out.mkdir(exist_ok=True)
api = HfApi()
model_name = "yuvalkirstain/PickScore_v1"
processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
provenance_path = out / "provenance.json"
reuse_rows = {}
if args.reuse:
    provenance = json.loads((args.reuse / "provenance.json").read_text())
    if provenance["model"] != model_name or provenance["processor"] != processor_name:
        raise ValueError("Reuse scorer identity mismatch")
    for line in (args.reuse / "raw_scores.jsonl").read_text().splitlines():
        row = json.loads(line)
        reuse_rows[(row["image_sha256"], row["prompt"])] = row["r_pickscore"]
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Existing output scorer identity mismatch")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
if provenance_path.exists():
    provenance = json.loads(provenance_path.read_text())
else:
    provenance = {
        "model": model_name,
        "model_revision": api.model_info(model_name).sha,
        "processor": processor_name,
        "processor_revision": api.model_info(processor_name).sha,
        "device": "cpu",
        "dtype": "float32",
        "threads": 4,
        "score": "exp(logit_scale) * normalized_text_dot_image / 26",
        "caveat": "Independent of optimized aesthetic reward, not human preference evidence. PickScore and prompt source share Pick-a-Pic lineage.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
processor = model = None
rows = []
for step in args.steps:
    for gpu in range(4):
        for line in (
            (root / f"eval_step{step}/gpu{gpu}/report/scores.jsonl").read_text().splitlines()
        ):
            row = json.loads(line)
            if step != args.steps[0] and row["checkpoint_label"] == "base":
                continue
            row["prompt_index"] += gpu * args.prompts_per_shard
            rows.append(row)
assert len(rows) == prompt_count * (len(args.steps) + 1)
for label in ["base", *(f"lora{step}" for step in args.steps)]:
    indices = [r["prompt_index"] for r in rows if r["checkpoint_label"] == label]
    if sorted(indices) != list(range(prompt_count)):
        raise ValueError(f"Incomplete or duplicate prompt grid for {label}")
result_path = out / "raw_scores.jsonl"
if result_path.exists():
    raise RuntimeError("Existing scores: inspect before any restart; never silently overwrite.")


def features(value):
    return value if isinstance(value, torch.Tensor) else value.pooler_output


with result_path.open("x") as handle, torch.inference_mode():
    for i, row in enumerate(rows):
        path = Path(row["image_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
        key = (row["image_sha256"], row["prompt"])
        if key in reuse_rows:
            score = reuse_rows[key]
        else:
            if model is None:
                print("Loading pinned CPU scorer.", flush=True)
                processor = CLIPProcessor.from_pretrained(
                    processor_name, revision=provenance["processor_revision"]
                )
                model = (
                    CLIPModel.from_pretrained(model_name, revision=provenance["model_revision"])
                    .eval()
                    .to("cpu")
                )
            with Image.open(path) as original:
                image_inputs = processor(images=original.convert("RGB"), return_tensors="pt")
            text_inputs = processor(
                text=[row["prompt"]],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            ie = features(model.get_image_features(**image_inputs))
            te = features(model.get_text_features(**text_inputs))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            score = float((model.logit_scale.exp() * (te @ ie.T)[0, 0] / 26).item())
        assert torch.isfinite(torch.tensor(score))
        result = {key: value for key, value in row.items() if not key.startswith("r_")}
        result["r_pickscore"] = score
        handle.write(json.dumps(result) + "\n")
        handle.flush()
        print(f"Scored {i + 1}/{len(rows)}", flush=True)
scored = [json.loads(line) for line in result_path.read_text().splitlines()]
write_curve_report(scored, out / "vs_base", bootstrap_resamples=10000, seed=260925)
write_curve_report(
    [r for r in scored if r["checkpoint_label"] != "base"],
    out / f"vs_lora{args.steps[0]}",
    base_label=f"lora{args.steps[0]}",
    bootstrap_resamples=10000,
    seed=260925,
)
print(f"Complete: {len(rows)} samples / {prompt_count} paired prompts.", flush=True)
