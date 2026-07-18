"""Shared native SANA inference protocol for evaluation entrypoints.

Training uses a stochastic Flow-GRPO scheduler at 512px, but model quality is
judged through SANA's public 1024px DPM-Solver++ pipeline.  Keeping scheduler,
precision, and generation validation here prevents one evaluation CLI from
silently drifting back to the training sampler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vrl.models.forward_precision import float32_precision_state
from vrl.models.interfaces.runtime import ForwardPrecision
from vrl.utils.media import to_pil_image

# These mappings are persisted protocol identities, not tunable defaults.
OFFICIAL_SAMPLING_PROTOCOL = {
    "negative_prompt": "",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 4.5,
    "max_sequence_length": 300,
    "use_resolution_binning": True,
    "complex_human_instruction": "official_pipeline_default",
}
SCHEDULER_PROTOCOL = {
    "class_name": "DPMSolverMultistepScheduler",
    "algorithm_type": "dpmsolver++",
    "solver_order": 2,
    "solver_type": "midpoint",
    "use_flow_sigmas": True,
    "flow_shift": 3.0,
    "prediction_type": "flow_prediction",
}


def validate_model_precision(
    model: Any,
    forward_precision: ForwardPrecision,
) -> dict[str, Any]:
    """Require a validated no-autocast SANA precision boundary."""

    pipeline = getattr(model, "pipeline", None)
    if pipeline is None:
        raise TypeError("SANA runtime model does not expose its native pipeline")
    actual = {
        "transformer": _dtype_name(_component_dtype(model.transformer, "transformer")),
        "prompt_encoder": _dtype_name(
            _component_dtype(getattr(pipeline, "text_encoder", None), "text_encoder"),
        ),
        "vae": _dtype_name(_component_dtype(getattr(pipeline, "vae", None), "vae")),
        "forward_precision": {
            "autocast": forward_precision.autocast,
            "float32_precision": forward_precision.float32_precision,
        },
        "effective_float32_precision": float32_precision_state(),
    }
    expected = {
        "transformer": "float16",
        "prompt_encoder": "bfloat16",
        "vae": "float32",
        "forward_precision": {
            "autocast": "off",
            "float32_precision": "ieee",
        },
        "effective_float32_precision": {
            "matmul": "ieee",
            "cudnn": "ieee",
        },
    }
    if actual != expected:
        raise ValueError(
            f"SANA precision boundary mismatch: expected={expected}, actual={actual}",
        )
    return actual


def load_official_scheduler(build: Any) -> Any:
    """Load and validate a fresh scheduler from the pinned model snapshot."""

    from diffusers import DPMSolverMultistepScheduler

    kwargs: dict[str, Any] = {"subfolder": "scheduler"}
    revision = _model_revision(build)
    if revision is not None:
        kwargs["revision"] = revision
    if Path(str(build.model_name_or_path)).expanduser().is_dir():
        kwargs["local_files_only"] = True
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        build.model_name_or_path,
        **kwargs,
    )
    validate_scheduler(scheduler)
    return scheduler


def validate_scheduler(scheduler: Any) -> dict[str, Any]:
    """Require SANA's pinned DPM-Solver++ scheduler identity."""

    config = getattr(scheduler, "config", None)
    if config is None:
        raise TypeError("official SANA scheduler has no config")
    actual = {
        "class_name": type(scheduler).__name__,
        "algorithm_type": _config_value(config, "algorithm_type"),
        "solver_order": _config_value(config, "solver_order"),
        "solver_type": _config_value(config, "solver_type"),
        "use_flow_sigmas": _config_value(config, "use_flow_sigmas"),
        "flow_shift": _config_value(config, "flow_shift"),
        "prediction_type": _config_value(config, "prediction_type"),
    }
    if actual != SCHEDULER_PROTOCOL:
        raise ValueError(
            "SANA scheduler does not match the official DPM-Solver++ protocol: "
            f"expected={SCHEDULER_PROTOCOL}, actual={actual}",
        )
    return actual


def generate_prompt_images(
    model: Any,
    *,
    forward_precision: ForwardPrecision,
    scheduler: Any,
    prompt: str,
    seed: int,
    num_images: int,
    device: torch.device,
    sampling: dict[str, Any] | None = None,
    require_official: bool = True,
) -> list[Any]:
    """Generate one prompt group through the official native pipeline."""

    if num_images < 1:
        raise ValueError(f"num_images must be >= 1; got {num_images}")
    if forward_precision != ForwardPrecision(
        autocast="off",
        float32_precision="ieee",
    ):
        raise ValueError(
            "SANA native inference requires resolved forward precision "
            "autocast='off', float32_precision='ieee'; "
            f"got {forward_precision!r}",
        )
    if _is_autocast_enabled(device):
        raise RuntimeError("SANA native inference must run without an outer autocast context")
    validate_scheduler(scheduler)
    protocol = dict(OFFICIAL_SAMPLING_PROTOCOL if sampling is None else sampling)
    if require_official and protocol != OFFICIAL_SAMPLING_PROTOCOL:
        raise ValueError(
            "SANA quality evaluation sampling changed from the official protocol: "
            f"{protocol!r} != {OFFICIAL_SAMPLING_PROTOCOL!r}",
        )
    model.pipeline.scheduler = scheduler
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        output = model.pipeline(
            prompt=prompt,
            negative_prompt=protocol["negative_prompt"],
            num_images_per_prompt=num_images,
            num_inference_steps=protocol["num_inference_steps"],
            guidance_scale=protocol["guidance_scale"],
            height=protocol["height"],
            width=protocol["width"],
            generator=generator,
            output_type="pil",
            use_resolution_binning=protocol["use_resolution_binning"],
            max_sequence_length=protocol["max_sequence_length"],
        )
    images = getattr(output, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != num_images:
        raise RuntimeError(
            f"SanaPipeline must return exactly {num_images} PIL images; "
            f"got {type(images).__name__} with length "
            f"{len(images) if isinstance(images, (list, tuple)) else 'n/a'}",
        )
    return [to_pil_image(image) for image in images]


def _component_dtype(component: Any, name: str) -> Any:
    if component is None or not hasattr(component, "dtype"):
        raise TypeError(f"SANA {name} does not expose a dtype")
    return component.dtype


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _model_revision(build: Any) -> str | None:
    model_config = build.model_config or {}
    revision = model_config.get("revision")
    if revision is None:
        return None
    text = str(revision).strip()
    return text or None


def _config_value(config: Any, key: str) -> Any:
    if hasattr(config, "get"):
        return config.get(key)
    return getattr(config, key, None)


def _is_autocast_enabled(device: torch.device) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device.type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


__all__ = [
    "OFFICIAL_SAMPLING_PROTOCOL",
    "SCHEDULER_PROTOCOL",
    "generate_prompt_images",
    "load_official_scheduler",
    "validate_model_precision",
    "validate_scheduler",
]
