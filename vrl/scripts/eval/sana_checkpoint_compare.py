"""Generate a SANA base-versus-checkpoint inference comparison.

This tool is intentionally separate from the registered LoRA aesthetic-curve
evaluation. It answers a narrower trust question for one full-parameter
checkpoint: can the base model and the restored checkpoint both generate under
the official SANA DPM-Solver++ inference protocol in one process?

The base image is generated before checkpoint loading. The checkpoint is then
restored through the same generic ``RuntimeBundle.trainable_modules`` boundary
used by training resume, and the current image is generated with a fresh
scheduler and an independently reset generator carrying the same seed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.models.interfaces.runtime import ForwardPrecision
from vrl.scripts.eval.sana_inference import (
    SCHEDULER_PROTOCOL,
    generate_prompt_images,
    load_official_scheduler,
    validate_model_precision,
    validate_scheduler,
)
from vrl.trainers.checkpointing import (
    load_trainable_state,
    load_training_checkpoint,
)
from vrl.utils.media import to_pil_image, write_png

logger = logging.getLogger(__name__)

# Compatibility facade for the public comparison tool's focused protocol tests.
_generate_prompt_group = generate_prompt_images
_load_official_scheduler = load_official_scheduler
_validate_model_precision = validate_model_precision
_validate_scheduler = validate_scheduler

# Persisted report identity is a protocol boundary.
REPORT_SCHEMA = "vrl.sana_checkpoint_compare/v1"
REPORT_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory containing resolved_config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint-final"),
        help="Checkpoint directory or checkpoint.pt path, relative to --run-dir by default.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <run-dir>/sana_checkpoint_compare.",
    )
    parser.add_argument(
        "--prompt",
        default="a red apple on a blue ceramic plate, studio photo",
        help="Prompt used for both images.",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument(
        "--device",
        default="auto",
        help="Generation device; auto selects cuda:0 when available, otherwise cpu.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_comparison(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


def run_comparison(args: argparse.Namespace) -> dict[str, str]:
    """Generate both images and return their materialized artifact paths."""

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"training run has no resolved config: {config_path}")
    cfg = load_config(config_path)
    _validate_resolved_config(cfg)
    _validate_sampling_args(args)

    checkpoint_input = args.checkpoint.expanduser()
    if not checkpoint_input.is_absolute():
        checkpoint_input = run_dir / checkpoint_input
    expected_checkpoint_file = (
        checkpoint_input if checkpoint_input.is_file() else checkpoint_input / "checkpoint.pt"
    )
    if not expected_checkpoint_file.is_file():
        raise FileNotFoundError(f"training checkpoint file not found: {expected_checkpoint_file}")

    output_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else run_dir / "sana_checkpoint_compare"
    )
    # A failed comparison must never leave a fresh base image beside a stale
    # current image or manifest from an older checkpoint. Treat each directory
    # as an immutable one-shot evidence bundle; callers choose a new directory
    # (or explicitly remove the incomplete one) before retrying.
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing comparison directory: {output_dir}",
        )

    device = _resolve_device(args.device)
    from vrl.models.diffusion.build import (
        build_family_runtime_bundle,
        resolve_family_model_build,
    )

    build = resolve_family_model_build(cfg, device, for_rollout=True)
    bundle = build_family_runtime_bundle(build)
    model = bundle.model.eval()
    dtype_record = _validate_model_precision(model, bundle.forward_precision)

    output_dir.mkdir(parents=True)
    base_path = output_dir / "base.png"
    current_path = output_dir / "current.png"
    side_by_side_path = output_dir / "side_by_side.png"

    logger.info("Generating SANA base image before reading %s", checkpoint_input)
    base_scheduler = _load_official_scheduler(build)
    base_image = _generate_one(
        model,
        forward_precision=bundle.forward_precision,
        scheduler=base_scheduler,
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    write_png(base_image, base_path)

    checkpoint = load_training_checkpoint(checkpoint_input)
    _validate_checkpoint(checkpoint)
    checkpoint_path = checkpoint.checkpoint_path
    checkpoint_meta = dict(checkpoint.meta)
    logger.info("Loading full-parameter checkpoint through the generic trainable-state boundary")
    load_trainable_state(bundle, checkpoint.trainable_state, strict=True)
    del checkpoint
    gc.collect()
    checkpoint_record = _checkpoint_record(checkpoint_path, checkpoint_meta)

    logger.info("Generating SANA current image after strict checkpoint restore")
    current_scheduler = _load_official_scheduler(build)
    if current_scheduler is base_scheduler:
        raise RuntimeError("official scheduler loader reused an instance across comparison images")
    current_image = _generate_one(
        model,
        forward_precision=bundle.forward_precision,
        scheduler=current_scheduler,
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    write_png(current_image, current_path)
    write_png(_side_by_side(base_image, current_image), side_by_side_path)

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "resolved_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "model": {
            "family": "sana",
            "path": str(build.model_name_or_path),
            "revision": _model_revision(build),
        },
        "checkpoint": checkpoint_record,
        "sampling": {
            "prompt": args.prompt,
            "negative_prompt": "",
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": 300,
            "use_resolution_binning": True,
            "complex_human_instruction": "official_pipeline_default",
        },
        "execution": {
            "device": str(device),
            "generation_order": ["base", "current"],
            "checkpoint_loaded_between_images": True,
            "strict_trainable_state_restore": True,
            "fresh_scheduler_per_image": True,
            "same_seed_generator_reset_per_image": True,
        },
        "dtype": dtype_record,
        "scheduler_protocol": dict(SCHEDULER_PROTOCOL),
        "artifacts": {
            "base": _artifact_record(base_path, output_dir),
            "current": _artifact_record(current_path, output_dir),
            "side_by_side": _artifact_record(side_by_side_path, output_dir),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return {
        "base": str(base_path),
        "current": str(current_path),
        "side_by_side": str(side_by_side_path),
        "manifest": str(manifest_path),
    }


def _validate_resolved_config(cfg: Any) -> None:
    family = str(cfg.model.get("family", "")).strip().lower()
    if family != "sana":
        raise ValueError(
            f"SANA checkpoint comparison requires model.family='sana'; got {family!r}"
        )
    if cfg.model.get("use_lora") is not False or cfg.model.get("lora") is not None:
        raise ValueError(
            "SANA checkpoint comparison accepts full-parameter runs only: "
            "model.use_lora must be false and model.lora must be null",
        )

    policy = resolve_precision_policy(cfg)
    if policy.training.dtype != policy.rollout.dtype:
        raise ValueError(
            "SANA comparison requires matching training and rollout precision; "
            f"got training={policy.training.dtype!r}, rollout={policy.rollout.dtype!r}",
        )
    if policy.training.dtype != "fp16":
        raise ValueError(
            "SANA comparison requires aligned native-FP16 policy precision; "
            f"got {policy.training.dtype!r}",
        )
    if policy.training.quantization is not None or policy.rollout.quantization is not None:
        raise ValueError("SANA comparison does not accept quantized training or rollout")
    if policy.prompt_encoder_dtype != "bf16":
        raise ValueError(
            "SANA comparison requires the BF16 prompt encoder; "
            f"got {policy.prompt_encoder_dtype!r}",
        )


def _validate_sampling_args(args: argparse.Namespace) -> None:
    if not str(args.prompt).strip():
        raise ValueError("prompt must not be empty")
    for name in ("height", "width", "steps"):
        value = int(getattr(args, name))
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}")
    if float(args.guidance_scale) <= 1.0:
        raise ValueError(
            "guidance-scale must be > 1.0 so the comparison exercises SANA's CFG path",
        )


def _validate_checkpoint(checkpoint: Any) -> None:
    payload_family = str(checkpoint.payload.get("family", "")).strip().lower()
    if payload_family != "sana":
        raise ValueError(f"checkpoint payload family must be 'sana'; got {payload_family!r}")
    meta_family = str(checkpoint.meta.get("family", "")).strip().lower()
    if meta_family != "sana":
        raise ValueError(f"checkpoint metadata family must be 'sana'; got {meta_family!r}")
    if checkpoint.meta.get("uses_lora") is not False:
        raise ValueError(
            "checkpoint metadata must declare uses_lora=false for full-parameter comparison",
        )
    expected_bytes = checkpoint.meta.get("checkpoint_file_bytes")
    actual_bytes = checkpoint.checkpoint_path.stat().st_size
    if not isinstance(expected_bytes, int) or expected_bytes != actual_bytes:
        raise ValueError(
            "checkpoint metadata byte count disagrees with checkpoint.pt: "
            f"metadata={expected_bytes!r}, actual={actual_bytes}",
        )


def _generate_one(
    model: Any,
    *,
    forward_precision: ForwardPrecision,
    scheduler: Any,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> Any:
    images = _generate_prompt_group(
        model,
        forward_precision=forward_precision,
        scheduler=scheduler,
        prompt=prompt,
        seed=seed,
        num_images=1,
        device=device,
        sampling={
            "negative_prompt": "",
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": 300,
            "use_resolution_binning": True,
            "complex_human_instruction": "official_pipeline_default",
        },
        require_official=False,
    )
    return images[0]


def _side_by_side(base_image: Any, current_image: Any) -> Any:
    from PIL import Image

    base = to_pil_image(base_image)
    current = to_pil_image(current_image)
    canvas = Image.new("RGB", (base.width + current.width, max(base.height, current.height)))
    canvas.paste(base, (0, 0))
    canvas.paste(current, (base.width, 0))
    return canvas


def _checkpoint_record(checkpoint_path: Path, checkpoint_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "meta": checkpoint_meta,
    }


def _artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _model_revision(build: Any) -> str | None:
    model_config = build.model_config or {}
    revision = model_config.get("revision")
    if revision is None:
        return None
    text = str(revision).strip()
    return text or None


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


if __name__ == "__main__":
    main()
