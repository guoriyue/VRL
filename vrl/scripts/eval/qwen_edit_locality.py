"""Run an untrained official Qwen pipeline on independent localized edits."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/qwen_image_21_reward_study")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    tasks = json.loads((out / "tasks.json").read_text())
    cases = [
        {**task, "task_id": task["name"], "name": f"{task['name']}_s{seed}", "seed": seed}
        for task in tasks
        for seed in [0, 1, 2, 3]
    ]
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.88)
    device = torch.device("cuda:0")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1",
        revision="b3179ad355be050328e483a9dfdd9e60cd62adfa",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # Only placement changes: official preprocessing, scheduler, and decode remain intact.
    pipe.transformer.to(device)
    pipe.vae.disable_tiling()
    original_vae_encode = pipe._encode_vae_image
    original_vae_decode = pipe.vae.decode

    reference_cache = {}

    def vae_encode_on_cpu(image, generator):
        image_cpu = image.cpu()
        key = hashlib.sha256(
            image_cpu.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        if key not in reference_cache:
            reference_cache[key] = original_vae_encode(image_cpu, generator).to(device)
        return reference_cache[key]

    def vae_decode_on_cpu(latents, **kwargs):
        return original_vae_decode(latents.cpu(), **kwargs)

    pipe._encode_vae_image = vae_encode_on_cpu
    pipe.vae.decode = vae_decode_on_cpu
    original_encode = pipe.encode_prompt

    prompt_cache = {}

    def encode_on_cpu(*a, **kw):
        key = (
            kw.get("prompt"),
            tuple(hashlib.sha256(im.tobytes()).hexdigest() for im in kw.get("image") or []),
        )
        if key not in prompt_cache:
            kw["device"] = torch.device("cpu")
            prompt_cache[key] = tuple(
                None if t is None else t.to(device) for t in original_encode(*a, **kw)
            )
        return prompt_cache[key]

    pipe.encode_prompt = encode_on_cpu
    report_path = out / "report.json"
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {
            "pipeline": type(pipe).__name__,
            "model": "Qwen/Qwen-Image-2.1",
            "revision": "b3179ad355be050328e483a9dfdd9e60cd62adfa",
            "settings": vars(args),
            "dtype": "bfloat16",
            "true_cfg_scale": 1.0,
            "use_kv_cache": True,
            "trained": False,
            "vae_tiling": False,
            "vae_device": "cpu",
            "cases": {},
        }
    )
    report["settings"].pop("seed", None)
    report["settings"]["seeds"] = [0, 1, 2, 3]
    report["reference_encoding_cache"] = (
        "Exact input bytes; deterministic VAE mode and prompt encoding."
    )
    for case in cases:
        name = case["name"]
        if name in report["cases"] and (out / "candidates" / f"{name}.png").exists():
            continue
        print(f"START {name}", flush=True)
        started = time.monotonic()
        with Image.open(out / case["source"]) as im:
            source = im.convert("RGBA")
        result = pipe(
            image=[source],
            prompt=case["prompt"],
            output_resolution=args.resolution,
            num_inference_steps=args.steps,
            true_cfg_scale=1.0,
            use_kv_cache=True,
            generator=torch.Generator(device=device).manual_seed(case["seed"]),
        ).images[0]
        output_path = out / "candidates" / f"{name}.png"
        output_tmp = output_path.with_suffix(".tmp")
        result.save(output_tmp, format="PNG")
        output_tmp.replace(output_path)
        resized = pipe.image_processor.resize(source, width=result.width, height=result.height)
        source_path = out / f"{Path(case['source']).stem}_input.png"
        source_tmp = source_path.with_suffix(".tmp")
        resized.save(source_tmp, format="PNG")
        source_tmp.replace(source_path)
        metrics = {
            **case,
            "output": f"candidates/{name}.png",
            "size": list(result.size),
            "seconds": time.monotonic() - started,
            "alpha_min": int(np.asarray(result)[..., 3].min()),
        }
        report["cases"][name] = metrics
        report_tmp = report_path.with_suffix(".tmp")
        report_tmp.write_text(json.dumps(report, indent=2) + "\n")
        report_tmp.replace(report_path)
        print(f"DONE {name}: {metrics}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
