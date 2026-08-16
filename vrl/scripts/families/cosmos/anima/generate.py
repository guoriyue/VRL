"""Generate Anima images through the repo-native runtime."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.generation.types import VideoGenerationRequest
from vrl.models.families.registry import get_model_family_entry
from vrl.scripts.eval._device import resolve_eval_device, resolve_eval_dtype
from vrl.trainers.data import load_prompt_manifest
from vrl.utils.media import to_pil_image

logger = logging.getLogger(__name__)

_DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, artist name"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Anima images with the repo-native diffusers/transformers runtime. "
            "This does not start or import a UI server."
        ),
    )
    parser.add_argument(
        "--config",
        default="experiment/anima_preview3/online_grpo_aesthetic_nsfw_safety",
        help="Bundled config name or absolute YAML path.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to generate. Repeat this flag for multiple prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        default="",
        help="Plain text file with one prompt per line.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Prompt manifest supported by vrl.trainers.data.load_prompt_manifest.",
    )
    parser.add_argument(
        "--eval-manifest",
        action="store_true",
        help="Use cfg.data.eval_manifest, falling back to cfg.data.manifest.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum prompts to generate.")
    parser.add_argument(
        "--output-dir",
        default="outputs/anima_generate",
        help="Directory for generated PNGs and metadata.",
    )
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--guidance-scale",
        "--cfg-scale",
        dest="guidance_scale",
        type=float,
        default=None,
    )
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument(
        "--negative-prompt",
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt used when CFG is enabled.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Defaults to cuda:0 when CUDA is available, else cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"),
        default="auto",
        help="Model weight dtype. auto follows the training config.",
    )
    parser.add_argument(
        "--lora-path",
        default="",
        help=(
            "Optional trained LoRA adapter path. A checkpoint directory is accepted "
            "when it contains lora_weights/."
        ),
    )
    parser.add_argument(
        "--use-config-lora",
        action="store_true",
        help=(
            "Respect cfg.model.use_lora even when cfg.model.lora.path is empty. "
            "By default empty training LoRA configs are disabled for inference."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, prompts, paths, and sampling settings without loading the model.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.use_lora=false sampling.num_steps=20",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    cfg = load_config(args.config, overrides=args.overrides)
    _configure_lora_for_inference(
        cfg, lora_path=args.lora_path, use_config_lora=args.use_config_lora
    )
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    prompts = _load_prompts(args, cfg)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError("provide at least one prompt source")

    sampling = _resolve_sampling(args, cfg)
    out_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": args.config,
        "prompt_count": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "sampling": sampling,
        "model": {
            "path": str(cfg.model.path),
            "use_lora": bool(cfg.model.use_lora),
            "lora_path": str(cfg.model.lora.path or ""),
        },
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    import torch

    device = resolve_eval_device(args.device)
    dtype = resolve_eval_dtype(
        args.dtype,
        root,
        precision=precision,
        device=device,
        requires_trainer="Anima generation",
    )
    logger.info("Building Anima runtime on device=%s dtype=%s", device, dtype)
    if root.model is None:
        raise ValueError("Anima generation requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    bundle = entry.build_rollout(
        entry.resolve_model_build(
            root,
            device,
            precision=precision,
            parameter_dtype_override=dtype,
        ),
    )
    model = bundle.model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for prompt_index, prompt in enumerate(prompts):
            prompt_seed = int(args.seed) + prompt_index * int(args.samples_per_prompt)
            logger.info(
                "Generating prompt %s/%s seed=%s",
                prompt_index + 1,
                len(prompts),
                prompt_seed,
            )
            images = _generate_images(
                model,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                seed=prompt_seed,
                samples_per_prompt=args.samples_per_prompt,
                sampling=sampling,
                torch=torch,
            )
            for sample_index, image in enumerate(images):
                image_path = image_dir / f"anima_{prompt_index:04d}_{sample_index:02d}.png"
                image.save(image_path)
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        # The whole samples_per_prompt batch is drawn from ONE
                        # generator seeded with prompt_seed (prepare_sampling);
                        # samples have no individual seeds, so record the batch
                        # seed that actually reproduces this row.
                        "seed": prompt_seed,
                        "image_path": str(image_path),
                        "prompt": prompt,
                    }
                )

    _write_metadata(rows, out_dir)
    print(json.dumps({"total_images": len(rows), "output_dir": str(out_dir)}, indent=2))


def _load_prompts(args: argparse.Namespace, cfg: DictConfig) -> list[str]:
    prompts: list[str] = [str(prompt).strip() for prompt in args.prompt if str(prompt).strip()]
    if args.prompt_file:
        path = Path(args.prompt_file).expanduser()
        prompts.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    manifest_path = args.manifest
    if args.eval_manifest:
        manifest_path = str(OmegaConf.select(cfg, "data.eval_manifest", default=cfg.data.manifest))
    if manifest_path:
        prompts.extend(example.prompt for example in load_prompt_manifest(manifest_path))
    return prompts


def _configure_lora_for_inference(
    cfg: DictConfig,
    *,
    lora_path: str,
    use_config_lora: bool,
) -> None:
    model = cfg.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    lora = model.get("lora")
    if lora is not None and not isinstance(lora, Mapping):
        raise ValueError("model.lora must be a mapping")

    if lora_path:
        OmegaConf.update(cfg, "model.use_lora", True, force_add=True)
        OmegaConf.update(
            cfg,
            "model.lora.path",
            str(_resolve_lora_path(lora_path)),
            force_add=True,
        )
        return

    configured_path = str((lora or {}).get("path") or "").strip()
    if bool(model.get("use_lora")) and not configured_path and not use_config_lora:
        logger.warning("Disabling empty training LoRA config for base-model inference")
        OmegaConf.update(cfg, "model.use_lora", False)


def _resolve_lora_path(path: str) -> Path:
    raw = Path(path).expanduser().resolve()
    exported = raw / "lora_weights"
    return exported if exported.exists() else raw


def _resolve_sampling(args: argparse.Namespace, cfg: DictConfig) -> dict[str, Any]:
    num_steps = int(args.steps or OmegaConf.select(cfg, "sampling.num_steps", default=20))
    return {
        "width": int(args.width or OmegaConf.select(cfg, "sampling.width", default=512)),
        "height": int(args.height or OmegaConf.select(cfg, "sampling.height", default=512)),
        "num_steps": num_steps,
        "guidance_scale": float(
            OmegaConf.select(cfg, "sampling.guidance_scale", default=4.5)
            if args.guidance_scale is None
            else args.guidance_scale,
        ),
        "max_sequence_length": int(
            args.max_sequence_length
            or OmegaConf.select(cfg, "sampling.max_sequence_length", default=128),
        ),
    }


def _generate_images(
    model: Any,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    samples_per_prompt: int,
    sampling: dict[str, Any],
    torch: Any,
) -> list[Image.Image]:
    prompts = [prompt] * samples_per_prompt
    negative_prompts = [negative_prompt] * samples_per_prompt
    encoded = model.encode_prompt(
        prompts,
        negative_prompts,
        max_sequence_length=int(sampling["max_sequence_length"]),
        guidance_scale=float(sampling["guidance_scale"]),
    )
    request = VideoGenerationRequest(
        negative_prompt=negative_prompt,
        width=int(sampling["width"]),
        height=int(sampling["height"]),
        frame_count=1,
        num_steps=int(sampling["num_steps"]),
        guidance_scale=float(sampling["guidance_scale"]),
        seed=int(seed),
    )
    state = model.prepare_sampling(request, encoded)
    with torch.no_grad():
        for step_idx, timestep in enumerate(state.timesteps):
            step_output = model.forward_step(state, step_idx)
            state.latents = state.scheduler.step(
                step_output["noise_pred"].float(),
                timestep,
                state.latents.float(),
                return_dict=False,
            )[0]
    decoded = model.decode_latents(state.latents)
    return [to_pil_image(image) for image in decoded]


def _write_metadata(rows: list[dict[str, Any]], out_dir: Path) -> None:
    jsonl_path = out_dir / "metadata.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = out_dir / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
