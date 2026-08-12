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
from vrl.config.schema import parse_config
from vrl.models import checkpoint_identity
from vrl.models.dtypes import dtype_to_wire_name
from vrl.models.precision import float32_precision_state, model_precision
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.sana_inference import (
    SCHEDULER_PROTOCOL,
    generate_prompt_images,
    load_official_scheduler,
)
from vrl.trainers.checkpointing import (
    load_training_checkpoint,
    read_checkpoint_meta,
    restore_model_checkpoint,
    validate_checkpoint_meta_compatibility,
)
from vrl.utils.artifacts import sha256_file
from vrl.utils.media import to_pil_image, write_png

logger = logging.getLogger(__name__)

# Compatibility facade for the public comparison tool's focused protocol tests.
_generate_prompt_group = generate_prompt_images
_load_official_scheduler = load_official_scheduler

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
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
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

    device = resolve_eval_device(args.device)
    from vrl.models.families.registry import get_model_family_entry
    from vrl.run import resolve_model

    entry = get_model_family_entry("sana")
    resolved = resolve_model(
        entry,
        root,
        device,
        precision=precision,
        for_rollout=True,
    )
    build = resolved.build
    model_identity = resolved.identity
    validate_checkpoint_meta_compatibility(
        read_checkpoint_meta(expected_checkpoint_file.parent),
        family="sana",
        expected_model_identity=model_identity,
        strict=True,
    )
    bundle = entry.build_rollout(build)
    # The registered mismatch wording below predates run.materialize and is
    # pinned by this tool's tests, so the recheck stays inline.
    if checkpoint_identity.resolve_checkpoint_model_identity(build) != model_identity:
        raise RuntimeError("SANA model source changed during runtime construction")
    model = bundle.model.eval()
    dtype_record = _model_precision_snapshot(model)

    output_dir.mkdir(parents=True)
    base_path = output_dir / "base.png"
    current_path = output_dir / "current.png"
    side_by_side_path = output_dir / "side_by_side.png"

    logger.info("Generating SANA base image before reading %s", checkpoint_input)
    base_scheduler = _load_official_scheduler(build)
    base_image = _generate_one(
        model,
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
    logger.info("Loading full-parameter checkpoint through the generic checkpoint boundary")
    restore_model_checkpoint(
        checkpoint,
        bundle=bundle,
        family="sana",
        expected_model_identity=model_identity,
        strict=True,
    )
    del checkpoint
    gc.collect()
    checkpoint_record = _checkpoint_record(checkpoint_path, checkpoint_meta)

    logger.info("Generating SANA current image after strict checkpoint restore")
    current_scheduler = _load_official_scheduler(build)
    if current_scheduler is base_scheduler:
        raise RuntimeError("official scheduler loader reused an instance across comparison images")
    current_image = _generate_one(
        model,
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
            "revision": build.revision_kwargs.get("revision"),
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


def _model_precision_snapshot(model: Any) -> dict[str, Any]:
    """Record the materialized SANA precision state for report provenance."""

    pipeline = getattr(model, "pipeline", None)
    if pipeline is None:
        raise TypeError("SANA runtime model does not expose its native pipeline")
    components = {
        "transformer": getattr(model, "transformer", None),
        "prompt_encoder": getattr(pipeline, "text_encoder", None),
        "vae": getattr(pipeline, "vae", None),
    }
    for name, component in components.items():
        if component is None or not hasattr(component, "dtype"):
            raise TypeError(f"SANA {name} does not expose a dtype")
    precision = model_precision(model)
    return {
        **{name: dtype_to_wire_name(component.dtype) for name, component in components.items()},
        "outer_autocast": precision.outer_autocast,
        "effective_float32_precision": float32_precision_state(),
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


# Canonical per-file digest lives in vrl.rewards.inference; keep the private name
# as an alias so the pinned test ref (checkpoint_compare._sha256) keeps resolving.
_sha256 = sha256_file


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
