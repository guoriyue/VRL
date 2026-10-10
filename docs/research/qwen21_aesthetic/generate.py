"""Resumable sharded qualification generation through VRL's native executor."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.registry import get_model_family_entry
from vrl.run import resolve_model
from vrl.utils.config import import_from_path
from vrl.utils.media import write_png

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--shards", type=int, default=4)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--samples", type=int, default=8)
p.add_argument("--resolution", type=int, default=512)
p.add_argument("--steps", type=int, default=10)
p.add_argument("--noise", type=float, default=0.7)
p.add_argument("--denoise-mode", choices=["sde", "native"], default="sde")
a = p.parse_args()
torch.set_num_threads(4)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(s) for s in Path(a.manifest).read_text().splitlines()]
rows = rows[a.shard :: a.shards]
if a.limit:
    rows = rows[: a.limit]
root = parse_config(
    OmegaConf.create(
        {
            "model": {
                "family": "qwen_image_21",
                "path": "Qwen/Qwen-Image-2.1",
                "revision": "b3179ad355be050328e483a9dfdd9e60cd62adfa",
                "use_lora": False,
                "memory": {"cpu_resident": [], "vae_decode": {"tiling": True, "slicing": True}},
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "bf16", "outer_autocast": False},
            },
        }
    )
)
entry = get_model_family_entry("qwen_image_21")
resolved = resolve_model(
    entry,
    root,
    torch.device("cuda:0"),
    precision=PrecisionPolicy.from_section(root.precision),
    for_rollout=True,
)
model = resolved.materialize(context="aesthetic qualification").model.eval()
executor = import_from_path(entry.executor_cls)(model, gatherer=entry.new_gatherer())
settings = {
    "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
    "model_identity": resolved.identity,
    "steps": a.steps,
    "resolution": a.resolution,
    "noise": a.noise,
    "samples": a.samples,
    "batch_size": 1,
    "seed_formula": "20260925 + 100*int(id[1:]) + sample_index",
}
if a.denoise_mode != "sde":
    settings["denoise_mode"] = a.denoise_mode
settings_file = out / f"settings_{a.shard}.json"
if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
    raise ValueError("Generation settings changed; use a new output directory")
settings_file.write_text(json.dumps(settings, indent=2) + "\n")
with torch.inference_mode():
    for row in rows:
        folder = out / row["id"]
        folder.mkdir(exist_ok=True)
        for sample in range(a.samples):
            path = folder / f"s{sample}.png"
            record_path = folder / f"s{sample}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("Existing image digest mismatch")
                continue
            seed = 20260925 + 100 * int(row["id"][1:]) + sample
            started = time.monotonic()
            request = GenerationRequest(
                request_id=f"{row['id']}-s{sample}",
                family="qwen_image_21",
                task="t2i",
                inputs=[GenerationInput(prompt=row["prompt"])],
                samples_per_prompt=1,
                sampling={
                    "height": a.resolution,
                    "width": a.resolution,
                    "num_steps": a.steps,
                    "guidance_scale": 1.0,
                    "seed": seed,
                    "reference_resolution": a.resolution,
                    "output_mode": "rgb",
                },
                denoise=DenoiseRequestOptions(
                    denoise_mode=a.denoise_mode, noise_level=a.noise, sde_type="flow_grpo"
                ),
            )
            batch = executor.forward_batch(
                request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
            )
            result = executor.merge_generation_batches(request, request.sample_rows(), [batch])
            write_png(result.output[0], path)
            record = {
                **row,
                "sample_id": request.request_id,
                "prompt_id": row["id"],
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seconds": time.monotonic() - started,
            }
            record_path.write_text(json.dumps(record) + "\n")
            print(
                json.dumps({"sample_id": record["sample_id"], "seconds": record["seconds"]}),
                flush=True,
            )
