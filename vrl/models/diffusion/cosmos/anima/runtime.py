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
        "resolved_paths": _resolve_anima_paths(cfg.model),
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
        "resolved_paths": _resolve_anima_replay_paths(cfg.model),
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

    paths = (spec.extra or {}).get("resolved_paths", {})
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
            "transformer_path": paths.get("transformer"),
            "text_encoder_path": paths.get("text_encoder"),
            "vae_path": paths.get("vae"),
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

    paths = (spec.extra or {}).get("resolved_paths", {})
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
            "transformer_path": paths.get("transformer"),
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


def _resolve_anima_paths(model_cfg: Any) -> dict[str, str]:
    root = Path(str(model_cfg.path)).expanduser()
    transformer = _optional_path(model_cfg, "transformer_path")
    text_encoder = _optional_path(model_cfg, "text_encoder_path")
    vae = _optional_path(model_cfg, "vae_path")

    if transformer is None or text_encoder is None or vae is None:
        base = root.parents[1] if root.is_file() else root
        if transformer is None:
            transformer = base / "diffusion_models" / "anima-preview3-base.safetensors"
        if text_encoder is None:
            text_encoder = base / "text_encoders" / "qwen_3_06b_base.safetensors"
        if vae is None:
            vae = base / "vae" / "qwen_image_vae.safetensors"

    tokenizer_root = _optional_path(model_cfg, "tokenizer_root")

    resolved = {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "tokenizer_root": tokenizer_root,
    }
    missing = {
        name: str(path)
        for name, path in resolved.items()
        if path is None or not Path(path).exists()
    }
    if missing:
        raise FileNotFoundError(
            "Anima full-generation runtime requires explicit local files; "
            f"missing={missing}",
        )
    return {name: str(Path(path)) for name, path in resolved.items()}


def _resolve_anima_replay_paths(model_cfg: Any) -> dict[str, str]:
    root = Path(str(model_cfg.path)).expanduser()
    transformer = _optional_path(model_cfg, "transformer_path")
    if transformer is None:
        transformer = (
            root
            if root.is_file()
            else root / "diffusion_models" / "anima-preview3-base.safetensors"
        )
    if not transformer.exists():
        raise FileNotFoundError(
            "Anima replay runtime requires only the transformer checkpoint; "
            f"missing={transformer}",
        )
    return {"transformer": str(transformer)}


def _load_anima_transformer_component(spec: RuntimeBuildSpec) -> Any:
    from safetensors.torch import load_file

    from vrl.models.diffusion.cosmos.anima.model import _load_anima_transformer

    path = (spec.extra or {}).get("resolved_paths", {}).get("transformer")
    if not path:
        raise ValueError("Anima runtime spec is missing resolved_paths.transformer")
    return _load_anima_transformer(
        load_file(str(path), device="cpu"),
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


def _optional_path(model_cfg: Any, name: str) -> Path | None:
    value = getattr(model_cfg, name, None)
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


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
