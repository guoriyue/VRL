"""Generate fixed-seed reference-image edits from base or a VRL checkpoint.

This evaluator keeps references and the original instructions in every request.
It saves a resumable candidate manifest for independent reward/visual evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.registry import get_model_family_entry
from vrl.run import resolve_model
from vrl.trainers.checkpointing import TrainingCheckpoint, restore_model_checkpoint
from vrl.utils.artifacts import sha256_file
from vrl.utils.config import import_from_path
from vrl.utils.media import write_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="manifests/edit_locality/tasks.json")
    parser.add_argument("--source-root", default="outputs/qwen_image_21_reward_study")
    parser.add_argument("--checkpoint")
    parser.add_argument("--label", default="base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--task-ids", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = parse_config(load_config(args.config))
    if root.model.family != "qwen_image_21":
        raise ValueError("This reference geometry protocol currently targets qwen_image_21")
    entry = get_model_family_entry(root.model.family)
    resolved = resolve_model(
        entry,
        root,
        torch.device("cuda:0"),
        precision=PrecisionPolicy.from_section(root.precision),
        for_rollout=True,
    )
    source_root = Path(args.source_root).resolve()
    tasks = [
        task
        for task in json.loads(Path(args.tasks).read_text())
        if "evaluation_prompt" not in task
    ]
    if args.task_ids:
        tasks = [task for task in tasks if task["name"] in args.task_ids]
    if not tasks:
        raise ValueError("No evaluation tasks selected")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None and checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.pt"
    settings = {
        **vars(args),
        "model_identity": resolved.identity,
        "task_manifest_sha256": sha256_file(Path(args.tasks)),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "source_hashes": {
            task["source"]: sha256_file(source_root / task["source"]) for task in tasks
        },
    }
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"settings": settings, "cases": {}}
    )
    if report["settings"] != settings:
        raise ValueError("Resume settings differ; choose a new output directory")
    bundle = resolved.materialize(context="reference-image checkpoint evaluation")
    model = bundle.model.eval()
    if args.checkpoint:
        checkpoint = TrainingCheckpoint.load(args.checkpoint)
        restore_model_checkpoint(
            checkpoint,
            bundle=bundle,
            family=entry.family,
            expected_model_identity=resolved.identity,
            strict=True,
        )
        del checkpoint
    executor = import_from_path(entry.executor_cls)(model, gatherer=entry.new_gatherer())
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    with (
        torch.inference_mode(),
        model.disable_adapter() if not args.checkpoint else contextlib.nullcontext(),
    ):
        for task in tasks:
            source = source_root / task["source"]
            with Image.open(source) as im:
                width, height, _ = calculate_dimensions(args.resolution**2, im.width / im.height)
                rgba = im.convert("RGBA")
                reference = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
            reference.resize((width, height), Image.Resampling.LANCZOS).save(
                out / f"{source.stem}_input.png"
            )
            for seed in args.seeds:
                name = f"{task['name']}_s{seed}"
                if name in report["cases"] and (out / report["cases"][name]["output"]).is_file():
                    continue
                request = GenerationRequest(
                    request_id=f"{args.label}-{name}",
                    family=entry.family,
                    task=entry.task,
                    inputs=[GenerationInput(prompt=task["prompt"], reference_image=str(source))],
                    samples_per_prompt=1,
                    sampling={
                        "height": height,
                        "width": width,
                        "num_steps": args.steps,
                        "guidance_scale": 1.0,
                        "seed": seed,
                        "reference_resolution": args.resolution,
                        "output_mode": "rgb",
                    },
                    denoise=DenoiseRequestOptions(denoise_mode="native"),
                )
                print(f"START {args.label} {name}", flush=True)
                batch = executor.forward_batch(
                    request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
                )
                initial_latents = batch.observations[:, 0].detach().cpu().contiguous()
                initial_latents_sha256 = hashlib.sha256(
                    initial_latents.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                generated = executor.merge_generation_batches(
                    request, request.sample_rows(), [batch]
                )
                path = out / f"{name}.png"
                write_png(generated.output[0], path)
                report["cases"][name] = {
                    **task,
                    "name": name,
                    "task_id": task["name"],
                    "seed": seed,
                    "initial_latents_sha256": initial_latents_sha256,
                    "initial_latents_shape": list(initial_latents.shape),
                    "initial_latents_dtype": str(initial_latents.dtype),
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "output": path.name,
                    "size": [width, height],
                    "checkpoint_label": args.label,
                }
                tmp = report_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(report, indent=2) + "\n")
                tmp.replace(report_path)
                print(f"DONE {args.label} {name}", flush=True)
                del batch, generated, initial_latents
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
