"""Generate Anima images through the repo-native runtime."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

from omegaconf import DictConfig

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.registry import get_model_family_entry
from vrl.models.interfaces.runtime import ModelBuild
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.denoise_generation import (
    GeneratorRuntimeIdentity,
    ImageSampling,
    generate_images,
)
from vrl.scripts.families.cosmos.anima.generation_protocol import ANIMA_GENERATION_SCHEMA
from vrl.trainers.data import PromptExample, load_prompt_manifest
from vrl.utils.artifacts import sha256_file

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Anima images with the repo-native diffusers/transformers runtime. "
            "This does not start or import a UI server."
        ),
    )
    parser.add_argument(
        "--config",
        default="model/cosmos/anima_preview3",
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
        default="worst quality, low quality, score_1, score_2, score_3, artist name",
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
        help="Model weight dtype. auto follows the precision config, or fp32 on CPU.",
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

    # Keep the established inference defaults without importing training rewards,
    # data, or trainer requirements. User overrides retain final precedence.
    overrides = (
        [
            "+sampling/image=512",
            "+sampling/denoise=10_step_cfg_4_5",
            "precision.float32_precision=tf32",
            "precision.training.dtype=bf16",
            "precision.training.outer_autocast=true",
        ]
        if args.config == "model/cosmos/anima_preview3"
        else []
    )
    overrides.extend(args.overrides)
    cfg = load_config(args.config, overrides=overrides)
    lora_overrides = _lora_overrides(cfg, lora_path=args.lora_path)
    if lora_overrides:
        cfg = load_config(args.config, overrides=[*overrides, *lora_overrides])
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    prompts = _load_prompts(args, root)
    manifest_path = _resolve_manifest_path(args, root)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError("provide at least one prompt source")

    sampling = _resolve_sampling(args, root)
    device = resolve_eval_device(args.device)
    dtype = resolve_torch_dtype(
        ("fp32" if device.type == "cpu" else precision.training.dtype)
        if args.dtype == "auto"
        else args.dtype
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(out_dir)

    if root.model is None:
        raise ValueError("Anima generation requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        parameter_dtype_override=dtype,
    )
    model_identity = resolve_checkpoint_model_identity(build)
    lora_path = str((root.model.lora.path if root.model.lora is not None else None) or "")
    lora_hashes = {
        "lora_weights_sha256": _lora_artifact_sha256(
            lora_path,
            "adapter_model.safetensors",
        ),
        "lora_config_sha256": _lora_artifact_sha256(
            lora_path,
            "adapter_config.json",
        ),
    }
    lora_checkpoint = _lora_checkpoint_provenance(lora_path, model_identity)
    generator_runtime = GeneratorRuntimeIdentity.capture()

    metadata = {
        "schema": ANIMA_GENERATION_SCHEMA,
        "config": args.config,
        "config_overrides": overrides,
        "prompt_count": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "base_seed": int(args.seed),
        "sampling": sampling.to_record(),
        "negative_prompt": args.negative_prompt,
        "execution": {
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
        },
        "generator_runtime": generator_runtime.to_record(),
        "prompt_source": {
            "manifest_path": str(manifest_path) if manifest_path else "",
            "manifest_sha256": sha256_file(manifest_path) if manifest_path else "",
            "prompt_file": str(Path(args.prompt_file).expanduser().resolve())
            if args.prompt_file
            else "",
            "prompt_file_sha256": (
                sha256_file(Path(args.prompt_file).expanduser().resolve())
                if args.prompt_file
                else ""
            ),
            "inline_prompt_count": sum(bool(str(prompt).strip()) for prompt in args.prompt),
            "limit": int(args.limit),
        },
        # This registry/schema-derived value is the sole base-checkpoint
        # identity. It includes independent transformer/text-encoder/VAE
        # sources and cannot drift when the public model schema gains fields.
        "model_identity": model_identity,
        "generation_policy": _generation_policy(build, precision),
        "model": {
            "use_lora": bool(root.model.use_lora),
            # Training presets declare lora rank/alpha/target_modules without a
            # path (nothing trained yet), so this key is genuinely absent.
            "lora_path": lora_path,
            "lora_checkpoint": lora_checkpoint,
            **lora_hashes,
        },
    }
    # Exclusive creation is the ownership reservation after the emptiness
    # check: two concurrent launches cannot both relabel the same directory.
    with (out_dir / "run_config.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    import torch

    logger.info("Building Anima runtime on device=%s dtype=%s", device, dtype)
    bundle = entry.build_rollout(build)
    loaded_identity = resolve_checkpoint_model_identity(build)
    loaded_generator_runtime = GeneratorRuntimeIdentity.capture()
    loaded_lora_hashes = {
        "lora_weights_sha256": _lora_artifact_sha256(
            lora_path,
            "adapter_model.safetensors",
        ),
        "lora_config_sha256": _lora_artifact_sha256(
            lora_path,
            "adapter_config.json",
        ),
    }
    loaded_lora_checkpoint = _lora_checkpoint_provenance(lora_path, loaded_identity)
    if (
        loaded_identity != model_identity
        or loaded_generator_runtime != generator_runtime
        or loaded_lora_checkpoint != lora_checkpoint
        or any(loaded_lora_hashes[key] != metadata["model"][key] for key in loaded_lora_hashes)
    ):
        raise RuntimeError("Anima model artifacts changed while the runtime was loading")
    model = bundle.model.eval()
    image_dir = out_dir / "images"
    image_dir.mkdir()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for prompt_index, example in enumerate(prompts):
            prompt = example.prompt
            prompt_seed = int(args.seed) + prompt_index * int(args.samples_per_prompt)
            logger.info(
                "Generating prompt %s/%s seed=%s",
                prompt_index + 1,
                len(prompts),
                prompt_seed,
            )
            images = generate_images(
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
                        "prompt_metadata": dict(example.metadata),
                        # Keep reward-only typed fields (target_text, target
                        # artifacts, references) beside the generated image so
                        # offline evaluators score the exact training target
                        # without reparsing the prompt.
                        "reward_metadata": example.reward_metadata(),
                    }
                )

    _write_metadata(
        rows,
        out_dir,
        anchor_source=(
            "anima_lora_synthetic" if bool(root.model.use_lora) else "anima_base_synthetic"
        ),
    )
    print(json.dumps({"total_images": len(rows), "output_dir": str(out_dir)}, indent=2))


def _load_prompts(args: argparse.Namespace, root: RootConfig) -> list[PromptExample]:
    prompts = [
        PromptExample(prompt=str(prompt).strip()) for prompt in args.prompt if str(prompt).strip()
    ]
    if args.prompt_file:
        path = Path(args.prompt_file).expanduser()
        prompts.extend(
            PromptExample(prompt=line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    manifest_path = _resolve_manifest_path(args, root)
    if manifest_path:
        prompts.extend(load_prompt_manifest(manifest_path))
    return prompts


def _resolve_manifest_path(args: argparse.Namespace, root: RootConfig) -> Path | None:
    manifest_path = args.manifest
    if args.eval_manifest:
        data = root.data
        if data is None or not (data.eval_manifest or data.manifest):
            raise ValueError(
                "--eval-manifest needs data.eval_manifest (or data.manifest) in the config"
            )
        manifest_path = str(data.eval_manifest or data.manifest)
    if not manifest_path:
        return None
    return Path(manifest_path).expanduser().resolve()


def _prepare_output_dir(out_dir: Path) -> None:
    """Create one empty artifact root without relabeling an earlier run."""

    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Anima output path is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise FileExistsError(f"Anima output directory is not empty: {out_dir}")
        return
    out_dir.mkdir(parents=True)


def _lora_artifact_sha256(lora_path: str, filename: str) -> str:
    if not lora_path:
        return ""
    artifact = Path(lora_path).expanduser().resolve() / filename
    return sha256_file(artifact) if artifact.is_file() else ""


def _lora_checkpoint_provenance(
    lora_path: str,
    expected_model_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind an exported adapter to its trainer checkpoint content, when present."""

    if not lora_path:
        return None
    adapter_dir = Path(lora_path).expanduser().resolve()
    checkpoint_dir = adapter_dir.parent if adapter_dir.name == "lora_weights" else None
    if checkpoint_dir is None:
        return None
    metadata_path = checkpoint_dir / "checkpoint_meta.json"
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{metadata_path} must contain a JSON object")
    if payload.get("family") != "cosmos-predict2-anima":
        raise ValueError(f"LoRA checkpoint family is not Anima: {metadata_path}")
    if payload.get("uses_lora") is not True:
        raise ValueError(f"LoRA checkpoint metadata must declare uses_lora=true: {metadata_path}")
    if payload.get("model_identity") != expected_model_identity:
        raise ValueError(
            f"LoRA checkpoint model identity differs from generation: {metadata_path}"
        )
    integer_fields: dict[str, int] = {}
    for field_name in (
        "schema_version",
        "global_step",
        "trainer_step",
        "completed_epoch",
        "next_epoch",
    ):
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"LoRA checkpoint {field_name} must be a non-negative integer: {metadata_path}",
            )
        integer_fields[field_name] = value
    return {
        "label": checkpoint_dir.name,
        "metadata_sha256": sha256_file(metadata_path),
        **integer_fields,
    }


def _generation_policy(build: ModelBuild, precision: PrecisionPolicy) -> dict[str, Any]:
    """Project the resolved rollout behavior that can change generated pixels."""

    rollout = build.require_rollout()
    return {
        "family": str(build.family),
        "parameter_dtype": str(build.parameter_dtype).removeprefix("torch."),
        "role_precision": asdict(build.precision),
        "diffusion_math_dtype": str(precision.diffusion_math),
        "prompt_encoder_dtype": str(rollout.prompt_encoder_dtype).removeprefix("torch."),
        "generation_memory": (
            asdict(build.generation_memory) if build.generation_memory is not None else None
        ),
        "rollout": {
            "base_weight_sync": rollout.base_weight_sync,
            "pipeline_offload_mode": rollout.pipeline_offload_mode,
        },
        "torch_compile": build.torch_compile,
    }


def _lora_overrides(cfg: DictConfig, *, lora_path: str) -> list[str]:
    """Dotlist overrides selecting the inference LoRA.

    They go back through ``load_config`` instead of mutating the loaded tree,
    so the result is composed and validated like any other override.
    """

    model = cfg.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    lora = model.get("lora")
    if lora is not None and not isinstance(lora, Mapping):
        raise ValueError("model.lora must be a mapping")

    if lora_path:
        raw_path = Path(lora_path).expanduser().resolve()
        exported_path = raw_path / "lora_weights"
        resolved_path = exported_path if exported_path.exists() else raw_path
        return ["model.use_lora=true", f"model.lora.path={resolved_path}"]

    configured_path = str((lora or {}).get("path") or "").strip()
    if bool(model.get("use_lora")) and not configured_path:
        logger.warning("Disabling empty training LoRA config for base-model inference")
        return ["model.use_lora=false"]
    return []


def _resolve_sampling(args: argparse.Namespace, root: RootConfig) -> ImageSampling:
    """The Anima sampling values, from the parsed config with the CLI on top."""

    return ImageSampling.from_root(
        root,
        overrides={
            "width": args.width,
            "height": args.height,
            "num_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
        },
    )


def _write_metadata(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    anchor_source: str,
) -> None:
    """Persist the evaluation index and its SFT-compatible target projection."""

    jsonl_path = out_dir / "metadata.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    # The same generation run can directly become synthetic clean-data
    # supervision. Paths stay relative to the chosen artifact root (out_dir),
    # while run_config.json pins the model, sampling, and negative prompt that
    # produced them. Multiple base samples for one prompt are valid independent
    # anchors because each image has its own stable target identity.
    anchor_rows = [
        {
            "prompt": row["prompt"],
            "target_image": str(Path(row["image_path"]).relative_to(out_dir)),
            "metadata": {
                **row.get("prompt_metadata", {}),
                "anchor_seed": row["seed"],
                "anchor_sample_index": row["sample_index"],
                "anchor_source": anchor_source,
            },
        }
        for row in rows
    ]
    (out_dir / "anchor_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in anchor_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
