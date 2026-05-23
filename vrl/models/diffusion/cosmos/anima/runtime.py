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
    return {
        "transformer": _resolve_anima_artifact_ref(
            model_cfg,
            field_name="transformer_path",
            relative_path="diffusion_models/anima-preview3-base.safetensors",
        ),
        "text_encoder": _resolve_anima_artifact_ref(
            model_cfg,
            field_name="text_encoder_path",
            relative_path="text_encoders/qwen_3_06b_base.safetensors",
        ),
        "vae": _resolve_anima_artifact_ref(
            model_cfg,
            field_name="vae_path",
            relative_path="vae/qwen_image_vae.safetensors",
        ),
        "qwen_tokenizer": _resolve_tokenizer_source(
            model_cfg,
            "qwen_tokenizer_path",
            default="Qwen/Qwen2.5-0.5B",
        ),
        "t5_tokenizer": _resolve_tokenizer_source(
            model_cfg,
            "t5_tokenizer_path",
            default="google-t5/t5-base",
        ),
    }


def _resolve_anima_replay_paths(model_cfg: Any) -> dict[str, str]:
    return {
        "transformer": _resolve_anima_artifact_ref(
            model_cfg,
            field_name="transformer_path",
            relative_path="diffusion_models/anima-preview3-base.safetensors",
        ),
    }


def _load_anima_transformer_component(spec: RuntimeBuildSpec) -> Any:
    from safetensors.torch import load_file

    from vrl.models.diffusion.cosmos.anima.model import _load_anima_transformer

    path = (spec.extra or {}).get("resolved_paths", {}).get("transformer")
    if not path:
        raise ValueError("Anima runtime spec is missing resolved_paths.transformer")
    return _load_anima_transformer(
        load_file(_materialize_anima_artifact(path), device="cpu"),
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


def _resolve_anima_artifact_ref(
    model_cfg: Any,
    *,
    field_name: str,
    relative_path: str,
) -> str:
    explicit = _optional_path(model_cfg, field_name)
    if explicit is not None:
        return _filesystem_file_ref(explicit, field_name=field_name)

    root_text = str(getattr(model_cfg, "path", "") or "").strip()
    if not root_text:
        raise ValueError("model.path is required for Anima checkpoint resolution")

    root = Path(root_text).expanduser()
    if root.exists():
        if root.is_file():
            if relative_path != "diffusion_models/anima-preview3-base.safetensors":
                raise FileNotFoundError(
                    "Anima filesystem file model.path can only supply the transformer; "
                    f"set {field_name} explicitly for {relative_path}",
                )
            return str(root)
        return _filesystem_file_ref(
            root / relative_path,
            field_name=f"model.path/{relative_path}",
        )

    return f"hf://{root_text}/{relative_path}"


def _filesystem_file_ref(path: Path, *, field_name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Anima {field_name} points to a missing file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Anima {field_name} must be a file: {path}")
    return str(path)


def _resolve_tokenizer_source(model_cfg: Any, name: str, *, default: str) -> str:
    value = getattr(model_cfg, name, None)
    text = str(value or "").strip() or default
    path = Path(text).expanduser()
    if path.exists():
        if not path.is_dir():
            raise FileNotFoundError(f"Anima {name} must be a tokenizer directory: {path}")
        return str(path)
    return text


def _materialize_anima_artifact(ref: str) -> str:
    text = str(ref)
    if not text.startswith("hf://"):
        return text
    repo_and_file = text[len("hf://"):]
    if "/" not in repo_and_file:
        raise ValueError(f"invalid Anima Hugging Face artifact reference: {ref!r}")
    owner, rest = repo_and_file.split("/", 1)
    repo_name, filename = rest.split("/", 1)
    repo_id = f"{owner}/{repo_name}"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to load Anima artifacts from Hugging Face "
            f"reference {ref!r}",
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=filename)


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
