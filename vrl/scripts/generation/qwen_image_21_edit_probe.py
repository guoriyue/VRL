"""Exercise Qwen-Image-2.1 editing through VRL's batch executor and replay.

Accepts existing source/reference images or a prompt manifest; writes RGB/RGBA PNGs
and a JSON report. Optional official-pipeline comparison shares VRL's initial latent.
Block offload allows this probe to coexist with a resident training process.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.registry import get_model_family_entry
from vrl.utils.artifacts import sha256_file
from vrl.utils.config import import_from_path
from vrl.utils.json_files import canonical_json_sha256
from vrl.utils.media import write_png


def main() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--reference")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Reference-conditioned prompt JSONL instead of the three built-in cases",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit manifest rows; zero means all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--path", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--revision", default="b3179ad355be050328e483a9dfdd9e60cd62adfa")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--width", type=int, help="Output width; set together with --height")
    parser.add_argument("--height", type=int, help="Output height; set together with --width")
    parser.add_argument("--reference-resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-mode", choices=("rgb", "rgba"), default="rgba")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-mode", choices=("shared", "independent"), default="shared")
    parser.add_argument("--denoise-mode", choices=("native", "sde"), default="native")
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--block-offload", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--check-lora-backward", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Restore a compatible VRL generator checkpoint before probing",
    )
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be set together")
    width = args.resolution if args.width is None else args.width
    height = args.resolution if args.height is None else args.height
    if any(value < 32 or value % 32 for value in (width, height)):
        parser.error("output dimensions must be positive multiples of 32")
    if args.manifest is None and (not args.source or not args.reference):
        parser.error("provide --manifest or both --source and --reference")
    if args.manifest is not None and (args.source or args.reference):
        parser.error("--manifest cannot be combined with --source or --reference")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.checkpoint and args.check_lora_backward:
        parser.error("checkpoint evaluation cannot also mutate a fresh LoRA adapter")
    if not math.isfinite(args.noise_level) or args.noise_level <= 0:
        parser.error("--noise-level must be finite and positive")
    if args.compare_reference and args.denoise_mode != "native":
        parser.error("official pipeline comparison requires --denoise-mode native")
    manifest_cases = None
    if args.manifest is not None:
        from vrl.trainers.data.prompts import load_prompt_dataset_index

        examples = load_prompt_dataset_index(args.manifest)
        if args.limit:
            examples = examples[: args.limit]
        if not examples:
            parser.error("manifest has no selected examples")
        manifest_cases = []
        for index, example in enumerate(examples):
            references = example.reference_images
            if not references or example.reference_video or example.request_overrides:
                parser.error(
                    "manifest probe requires image references and fixed sampling without per-row overrides"
                )
            manifest_cases.append(
                (
                    f"sample_{index:05d}",
                    example.prompt,
                    [
                        str((args.manifest.parent / Path(path).expanduser()).resolve(strict=True))
                        for path in references
                    ],
                )
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    from vrl.run import resolve_model
    from vrl.trainers.checkpointing import (
        TrainingCheckpoint,
        restore_model_checkpoint,
        validate_checkpoint_compatibility,
    )

    checkpoint = TrainingCheckpoint.load(args.checkpoint) if args.checkpoint else None
    model_config = {
        "family": "qwen_image_21",
        "path": args.path,
        "revision": args.revision,
        "use_lora": False,
        "memory": {
            "cpu_resident": ["text_encoder"],
            "vae_decode": {"tiling": True, "slicing": True},
        },
    }
    if checkpoint is not None:
        if not checkpoint.model_identity:
            raise ValueError("checkpoint evaluation requires a recorded model identity")
        model_config.update(checkpoint.model_identity["build"])
    root = parse_config(
        OmegaConf.create(
            {
                "model": model_config,
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
        device,
        precision=PrecisionPolicy.from_section(root.precision),
        for_rollout=True,
    )
    build = resolved.build
    if args.block_offload:
        build = replace(build, rollout=replace(build.rollout, pipeline_offload_mode="block"))
        resolved = replace(resolved, build=build)
    validate_checkpoint_compatibility(
        checkpoint, family=entry.family, expected_model_identity=resolved.identity
    )
    bundle = resolved.materialize(context="Qwen reference/RGBA probe")
    restore_model_checkpoint(
        checkpoint, bundle=bundle, family=entry.family, expected_model_identity=resolved.identity
    )
    model = bundle.model.eval()
    pipe = model.pipeline
    replay_cls = import_from_path(entry.family_build.replay_cls)
    replay = replay_cls(transformer=pipe.transformer, scheduler=pipe.scheduler, device=device)
    replay.precision = model.precision
    executor = import_from_path(entry.executor_cls)(model, gatherer=entry.new_gatherer())
    if args.compare_reference:
        # Both loops start from exactly the same bf16-representable values;
        # sharing only a seed would compare different dtype-specific draws.
        def draw_comparison_latents(*, request, encoded, config, prepare_kwargs):
            state = model.prepare_sampling(request, encoded, **(prepare_kwargs or {}))
            return state.latents.to(torch.bfloat16).float()

        executor.draw_initial_latents = draw_comparison_latents
    cases = [
        (
            "localized_edit",
            'Change the text on the wooden sign to say "Closed Today". Keep everything else unchanged.',
            [args.source],
        ),
        (
            "multiple_references",
            "Place the maple leaf from image 2 onto the wooden sign of the bookshop in image 1, keeping everything else unchanged.",
            [args.source, args.reference],
        ),
        (
            "subject_extraction",
            "This is an RGBA image with transparency. Extract only the wooden bookshop sign from the input image, preserving its lettering and appearance. The image has alpha channel and the background is transparent.",
            [args.source],
        ),
    ]
    if manifest_cases is not None:
        cases = manifest_cases
    report = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "manifest_sha256": sha256_file(args.manifest) if args.manifest else None,
        "model_identity": resolved.identity,
        "checkpoint_sha256": sha256_file(checkpoint.checkpoint_path) if checkpoint else None,
        "cases": {},
    }
    original_encode = pipe.encode_prompt
    original_prepare = pipe.prepare_latents

    def encode_on_cpu(*a, **kw):
        kw["device"] = torch.device("cpu")
        return tuple(None if t is None else t.to(device) for t in original_encode(*a, **kw))

    def prepare_in_vae_dtype(
        images, batch_size, channels, height, width, dtype, device, generator, latents=None
    ):
        # Match VRL's fp32 VAE boundary without first rounding source pixels
        # to text dtype. The supplied initial latent is already shared exactly.
        target, references = original_prepare(
            images, batch_size, channels, height, width, pipe.vae.dtype, device, generator, latents
        )
        return target.to(dtype), None if references is None else references.to(dtype)

    with torch.inference_mode():
        for name, prompt, paths in cases:
            print(f"[{name}] starting", flush=True)
            started = time.monotonic()
            reference_records = [
                {"path": path, "sha256": sha256_file(Path(path))} for path in paths
            ]
            case_seed = (
                args.seed
                if args.seed_mode == "shared"
                else int(
                    canonical_json_sha256(
                        [args.seed, prompt, [item["sha256"] for item in reference_records]],
                        allow_nan=False,
                    )[:16],
                    16,
                )
                % (2**63)
            )
            request = GenerationRequest(
                request_id=name,
                family="qwen_image_21",
                task="t2i",
                inputs=[GenerationInput(prompt=prompt, reference_images=paths)],
                samples_per_prompt=1,
                sampling={
                    "height": height,
                    "width": width,
                    "num_steps": args.steps,
                    "guidance_scale": 1.0,
                    "seed": case_seed,
                    "reference_resolution": args.reference_resolution,
                    "output_mode": args.output_mode,
                },
                denoise=DenoiseRequestOptions(
                    denoise_mode=args.denoise_mode,
                    noise_level=args.noise_level,
                    sde_type="flow_grpo",
                ),
            )
            batch = executor.forward_batch(
                request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
            )
            # Exercise driver gather as well as worker encode/prepare/denoise/decode.
            generated = executor.merge_generation_batches(request, request.sample_rows(), [batch])
            image = generated.output[0]
            write_png(image, out / f"{name}.png")
            output_sha256 = sha256_file(out / f"{name}.png")
            restored = replay.restore_eval_state(
                {**batch.replay_tensors, "timesteps": batch.timesteps},
                batch.context,
                batch.observations[:, 0],
                0,
            )
            pred = replay.forward_step(restored, 0)["noise_pred"]
            if args.denoise_mode == "sde":
                from vrl.math.denoise.flow_matching import sde_step_with_logprob

                step = sde_step_with_logprob(
                    pipe.scheduler,
                    pred.float(),
                    restored.timesteps[:1],
                    restored.latents.float(),
                    prev_sample=batch.actions[:, 0].float(),
                    noise_level=args.noise_level,
                    step_index=0,
                )
                replay_error = (step.log_prob - batch.log_probs[:, 0]).abs().max().item()
                replay_metric = "replay_logprob_max_error"
            else:
                # Match the scheduler's actual dtype rules, including its output cast.
                replay_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    pipe.scheduler.config
                )
                replay_scheduler.timesteps = pipe.scheduler.timesteps.clone()
                replay_scheduler.sigmas = pipe.scheduler.sigmas.clone()
                expected_action = replay_scheduler.step(
                    pred, restored.timesteps[0], batch.observations[:, 0], return_dict=False
                )[0]
                replay_error = (expected_action - batch.actions[:, 0]).abs().max().item()
                replay_metric = "replay_action_max_error"
            if not math.isfinite(replay_error) or replay_error > 1e-4:
                raise AssertionError(f"{name}: {replay_metric} mismatch {replay_error}")
            if any(
                sha256_file(Path(item["path"])) != item["sha256"] for item in reference_records
            ):
                raise ValueError(f"{name}: reference changed during generation")
            expected_shape = (4 if args.output_mode == "rgba" else 3, height, width)
            if tuple(image.shape) != expected_shape:
                raise AssertionError(f"{name}: decoded shape differs from {expected_shape}")
            metrics = {
                "prompt": prompt,
                "references": reference_records,
                "seed": case_seed,
                "path": str(out / f"{name}.png"),
                "sha256": output_sha256,
                "shape": list(image.shape),
                "output_mode": args.output_mode,
                replay_metric: replay_error,
                "reference_latent_tokens": batch.replay_tensors["reference_latents"].shape[1],
            }
            if args.output_mode == "rgba":
                alpha = image[3].float()
                metrics.update(
                    transparent_fraction=(alpha < 128).float().mean().item(),
                    partial_alpha_fraction=((alpha > 0) & (alpha < 255)).float().mean().item(),
                    mid_alpha_fraction=((alpha >= 8) & (alpha <= 247)).float().mean().item(),
                )
            if args.compare_reference:
                print(
                    f"[{name}] VRL saved; replay max error={replay_error}; running official comparison",
                    flush=True,
                )
                images = []
                for path in paths:
                    with Image.open(path) as im:
                        images.append(im.convert("RGBA"))
                pipe.encode_prompt = encode_on_cpu
                pipe.prepare_latents = prepare_in_vae_dtype
                try:
                    reference = pipe(
                        prompt=prompt,
                        image=images,
                        width=width,
                        height=height,
                        output_resolution=args.reference_resolution,
                        num_inference_steps=args.steps,
                        latents=batch.observations[:, 0].clone(),
                        use_kv_cache=False,
                        output_type="pt",
                    ).images[0]
                finally:
                    pipe.encode_prompt = original_encode
                    pipe.prepare_latents = original_prepare
                if args.output_mode == "rgb" and reference.shape[0] == 4:
                    reference = reference[:3] * reference[3:4] + (1.0 - reference[3:4])
                write_png(reference, out / f"{name}_official.png")
                metrics["official_sha256"] = sha256_file(out / f"{name}_official.png")
                difference = (image.float() - (reference.float() * 255).round()).abs()
                metrics[f"official_{args.output_mode}_mae_255"] = difference.mean().item()
                metrics["official_fraction_over_8"] = (difference > 8).float().mean().item()
            metrics["seconds"] = time.monotonic() - started
            report["cases"][name] = metrics
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"[{name}] {json.dumps(metrics)}", flush=True)

    if args.check_lora_backward:
        from vrl.math.denoise.flow_matching import sde_step_with_logprob

        print("[lora] checking SDE replay and one optimizer step", flush=True)
        lora_build = replace(
            build,
            model_config={
                **(build.model_config or {}),
                "use_lora": True,
                "lora": {
                    "rank": 4,
                    "alpha": 4,
                    "target_modules": ["to_q", "to_v"],
                    "parameter_dtype": "float32",
                },
            },
        )
        model.apply_lora(lora_build)
        replay = replay_cls(transformer=pipe.transformer, scheduler=pipe.scheduler, device=device)
        replay.precision = model.precision
        _, lora_prompt, lora_references = cases[min(1, len(cases) - 1)]
        request = GenerationRequest(
            request_id="lora_edit",
            family="qwen_image_21",
            task="t2i",
            inputs=[GenerationInput(prompt=lora_prompt, reference_images=lora_references)],
            samples_per_prompt=1,
            sampling={
                "height": 256,
                "width": 256,
                "num_steps": 2,
                "guidance_scale": 1.0,
                "seed": args.seed,
                "reference_resolution": 256,
                "output_mode": args.output_mode,
            },
            denoise=DenoiseRequestOptions(noise_level=0.7),
        )
        with torch.no_grad():
            batch = executor.forward_batch(
                request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
            )
        restored = replay.restore_eval_state(
            {**batch.replay_tensors, "timesteps": batch.timesteps},
            batch.context,
            batch.observations[:, 0],
            0,
        )
        pred = replay.forward_step(restored, 0)["noise_pred"]
        step = sde_step_with_logprob(
            pipe.scheduler,
            pred.float(),
            restored.timesteps[:1],
            restored.latents.float(),
            prev_sample=batch.actions[:, 0].float(),
            noise_level=0.7,
            step_index=0,
        )
        parity = (step.log_prob - batch.log_probs[:, 0]).abs().max().item()
        if parity > 1e-4:
            raise AssertionError(f"LoRA SDE replay log-prob mismatch: {parity}")
        (-step.log_prob.mean()).backward()
        parameters = [p for p in model.transformer.parameters() if p.requires_grad]
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("Missing or non-finite LoRA gradients")
        gradient_norm = sum(g.float().square().sum().item() for g in gradients) ** 0.5
        before = [p.detach().clone() for p in parameters]
        torch.optim.SGD(parameters, lr=1e-3).step()
        delta = max(
            (p - old).abs().max().item() for p, old in zip(parameters, before, strict=True)
        )
        if gradient_norm == 0 or delta == 0:
            raise AssertionError("LoRA optimizer step did not update parameters")
        report["lora_backward"] = {
            "sde_logprob_max_error": parity,
            "gradient_norm": gradient_norm,
            "parameter_max_delta": delta,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[lora] {json.dumps(report['lora_backward'])}", flush=True)


if __name__ == "__main__":
    main()
