"""Cosmos Predict2 Anima family runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vrl.generation.diffusion import DiffusionPipelineExecutorBase
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)

logger = logging.getLogger(__name__)

ANIMA_FAMILY = "cosmos-predict2-anima"
ANIMA_CAPABILITY = diffusion_family_capability(ANIMA_FAMILY, "t2i")


def extract_anima_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Slice the Anima-specific runtime fields out of a whole RL config."""

    lora_cfg: dict[str, Any] | None = None
    lora_path: str | None = None
    if cfg.model.use_lora:
        lora_path = cfg.model.lora.path or None
        lora_cfg = {
            "rank": int(cfg.model.lora.rank),
            "alpha": int(cfg.model.lora.alpha),
            "target_modules": list(cfg.model.lora.target_modules),
        }

    extra: dict[str, Any] = {
        "transformer_path": str(getattr(cfg.model, "transformer_path", "") or ""),
        "transformer_file": str(getattr(cfg.model, "transformer_file", "") or ""),
        "text_encoder_path": str(getattr(cfg.model, "text_encoder_path", "") or ""),
        "text_encoder_file": str(getattr(cfg.model, "text_encoder_file", "") or ""),
        "vae_path": str(getattr(cfg.model, "vae_path", "") or ""),
        "vae_file": str(getattr(cfg.model, "vae_file", "") or ""),
        "qwen_tokenizer_path": str(getattr(cfg.model, "qwen_tokenizer_path", "") or ""),
        "t5_tokenizer_path": str(getattr(cfg.model, "t5_tokenizer_path", "") or ""),
        "scheduler_shift": float(getattr(cfg.model, "scheduler_shift", 3.0)),
    }
    torch_compile_cfg = getattr(cfg.model, "torch_compile", None)
    if torch_compile_cfg is not None and torch_compile_cfg.enable:
        extra["torch_compile"] = {
            "enable": True,
            "mode": torch_compile_cfg.mode,
        }

    return RuntimeBuildSpec(
        model_name_or_path=str(cfg.model.path),
        device=device,
        dtype=weight_dtype,
        backend_preference=("diffusers",),
        task_variant="text_to_image",
        use_lora=bool(cfg.model.use_lora),
        lora_path=lora_path,
        lora_config=lora_cfg,
        scheduler_config={"num_steps": int(cfg.sampling.num_steps)},
        extra=extra,
    )


def extract_anima_replay_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Slice the trainer replay-only Anima runtime fields out of whole cfg."""

    lora_cfg: dict[str, Any] | None = None
    lora_path: str | None = None
    if cfg.model.use_lora:
        lora_path = cfg.model.lora.path or None
        lora_cfg = {
            "rank": int(cfg.model.lora.rank),
            "alpha": int(cfg.model.lora.alpha),
            "target_modules": list(cfg.model.lora.target_modules),
        }

    extra: dict[str, Any] = {
        "transformer_path": str(getattr(cfg.model, "transformer_path", "") or ""),
        "transformer_file": str(getattr(cfg.model, "transformer_file", "") or ""),
        "scheduler_shift": float(getattr(cfg.model, "scheduler_shift", 3.0)),
    }
    torch_compile_cfg = getattr(cfg.model, "torch_compile", None)
    if torch_compile_cfg is not None and torch_compile_cfg.enable:
        extra["torch_compile"] = {
            "enable": True,
            "mode": torch_compile_cfg.mode,
        }

    return RuntimeBuildSpec(
        model_name_or_path=str(cfg.model.path),
        device=device,
        dtype=weight_dtype,
        backend_preference=("diffusers",),
        task_variant="text_to_image",
        use_lora=bool(cfg.model.use_lora),
        lora_path=lora_path,
        lora_config=lora_cfg,
        scheduler_config={"num_steps": int(cfg.sampling.num_steps)},
        extra=extra,
    )


def build_anima_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    from vrl.models.diffusion.cosmos.anima.model import AnimaModel

    logger.info("Building Anima runtime bundle from %s", spec.model_name_or_path)
    _resolve_anima_paths(spec)
    model = AnimaModel.from_spec(spec)
    if spec.use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        model.set_num_steps(int(num_steps))

    extra = spec.extra or {}
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_kind="diffusers",
        backend_handle=model.backend_handle,
        runtime_caps={
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": True,
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            "transformer_path": extra.get("transformer_path"),
            "text_encoder_path": extra.get("text_encoder_path"),
            "vae_path": extra.get("vae_path"),
            **full_generation_bundle_metadata(
                replay_modules=("transformer", "scheduler"),
                generation_only_modules=("text_encoder", "llm_adapter", "vae"),
            ),
        },
    )


def build_anima_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without Anima generation-only modules."""

    from vrl.models.diffusion.cosmos.anima.model import AnimaReplayModel

    logger.info("Building Anima replay runtime bundle from %s", spec.model_name_or_path)
    model = AnimaReplayModel(
        transformer=_load_anima_transformer_component(spec),
        scheduler=_build_anima_scheduler(spec),
        device=spec.device,
        dtype=_resolve_torch_dtype(spec.dtype),
    )

    if spec.use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        model.set_num_steps(int(num_steps))

    extra = spec.extra or {}
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_kind="diffusers",
        backend_handle=None,
        runtime_caps={
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": False,
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            "transformer_path": extra.get("transformer_path"),
            **minimal_replay_bundle_metadata(
                replay_modules=("transformer", "scheduler"),
                generation_only_modules=(
                    "text_encoder",
                    "llm_adapter",
                    "vae",
                    "tokenizers",
                ),
            ),
        },
    )


def build_anima_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    spec = extract_anima_runtime_spec(cfg, device, weight_dtype)
    return build_anima_runtime_bundle(spec)


def build_anima_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    spec = extract_anima_replay_runtime_spec(cfg, device, weight_dtype)
    return build_anima_replay_runtime_bundle(spec)


class AnimaPipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Anima text-to-image rollouts."""

    family: str = ANIMA_FAMILY
    task: str = "t2i"
    family_capability = ANIMA_CAPABILITY
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512

    def __init__(self, model: Any, *, sample_batch_size: int = 1) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))


def _resolve_anima_paths(spec: RuntimeBuildSpec) -> None:
    """Resolve raw config path strings into real file paths in spec.extra."""

    extra = spec.extra or {}
    root = str(spec.model_name_or_path or "").strip()
    fields = [
        ("transformer_path", "transformer_file"),
        ("text_encoder_path", "text_encoder_file"),
        ("vae_path", "vae_file"),
    ]
    for path_field, file_field in fields:
        resolved = _resolve_artifact(
            root,
            explicit_path=extra.get(path_field, ""),
            relative_file=extra.get(file_field, ""),
            field_name=path_field,
        )
        if resolved:
            extra[path_field] = resolved


def _resolve_artifact(
    root: str,
    *,
    explicit_path: str,
    relative_file: str,
    field_name: str,
) -> str:
    if explicit_path:
        return explicit_path
    if not (root and relative_file):
        return ""
    root_path = Path(root).expanduser()
    if root_path.exists() or root.startswith(("/", "./", "../", "~")):
        return str(root_path / relative_file)
    raise ValueError(
        f"model.path={root!r} is not a local artifact root; set model.{field_name} "
        "or point model.path at a local directory",
    )


def _load_anima_transformer_component(spec: RuntimeBuildSpec) -> Any:
    from safetensors.torch import load_file

    from vrl.models.diffusion.cosmos.anima.model import _load_anima_transformer

    extra = spec.extra or {}
    path = extra.get("transformer_path") or _resolve_artifact(
        str(spec.model_name_or_path or ""),
        explicit_path="",
        relative_file=extra.get("transformer_file", ""),
        field_name="transformer_path",
    )
    if not path:
        raise ValueError("Anima replay runtime spec is missing transformer_path")
    return _load_anima_transformer(
        load_file(path, device="cpu"),
        dtype=_resolve_torch_dtype(spec.dtype),
    ).to(spec.device, dtype=_resolve_torch_dtype(spec.dtype))


def _build_anima_scheduler(spec: RuntimeBuildSpec) -> Any:
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=float((spec.extra or {}).get("scheduler_shift", 3.0)),
    )
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        scheduler.set_timesteps(int(num_steps), device=spec.device)
    return scheduler


def _resolve_torch_dtype(value: Any) -> Any:
    import torch

    if isinstance(value, torch.dtype):
        return value
    text = str(value).replace("torch.", "")
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(text, torch.bfloat16)


__all__ = [
    "ANIMA_FAMILY",
    "AnimaPipelineExecutor",
    "build_anima_replay_runtime_bundle",
    "build_anima_replay_runtime_bundle_from_cfg",
    "build_anima_runtime_bundle",
    "build_anima_runtime_bundle_from_cfg",
    "extract_anima_replay_runtime_spec",
    "extract_anima_runtime_spec",
]
