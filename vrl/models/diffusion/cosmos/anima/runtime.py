"""Cosmos Predict2 Anima family runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vrl.generation.diffusion import DiffusionChunkExecutorBase
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.models.loader import apply_rollout_quantization
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.runtime_config import (
    extract_runtime_spec,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

ANIMA_FAMILY = "cosmos-predict2-anima"


def extract_anima_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Slice the Anima-specific runtime fields out of a whole RL config."""

    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="text_to_image",
    )


def extract_anima_replay_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Trainer replay-only Anima runtime spec.

    With the whole ``cfg.model`` block carried wholesale, the replay model only
    reads the artifact paths / scheduler_shift / torch_compile it needs; the
    remaining fields ride along inertly, so no trimming is required.
    """

    return extract_anima_runtime_spec(cfg, device, weight_dtype)


def build_anima_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    from vrl.models.diffusion.cosmos.anima.model import AnimaModel

    logger.info("Building Anima runtime bundle from %s", spec.model_name_or_path)
    model_config = spec.model_config or {}
    root = str(spec.model_name_or_path or "").strip()
    for path_field, file_field in (
        ("transformer_path", "transformer_file"),
        ("text_encoder_path", "text_encoder_file"),
        ("vae_path", "vae_file"),
    ):
        resolved = _resolve_artifact(
            root,
            explicit_path=model_config.get(path_field, ""),
            relative_file=model_config.get(file_field, ""),
            field_name=path_field,
        )
        if resolved:
            model_config[path_field] = resolved

    use_lora = spec.use_lora
    model = AnimaModel.from_spec(spec)
    if use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    apply_rollout_quantization(model, spec)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(int(num_steps))

    metadata = {
        "model_path": spec.model_name_or_path,
        "family": ANIMA_FAMILY,
        "task_variant": spec.task_variant,
        "dtype": str(spec.dtype),
        "use_lora": use_lora,
        "transformer_path": model_config.get("transformer_path"),
        "text_encoder_path": model_config.get("text_encoder_path"),
        "vae_path": model_config.get("vae_path"),
        **full_generation_bundle_metadata(),
    }
    metadata.update(apply_generation_memory_policy(
        model,
        memory_config=getattr(spec, "memory", None),
        owner="Anima VAE",
    ))
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        runtime_caps={
            "supports_reference_conditioning": False,
        },
        metadata=metadata,
    )


def build_anima_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without Anima generation-only modules."""

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.models.diffusion.cosmos.anima.model import AnimaReplayModel
    from vrl.models.dtypes import resolve_torch_dtype

    logger.info("Building Anima replay runtime bundle from %s", spec.model_name_or_path)

    model_config = spec.model_config or {}
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=float(model_config.get("scheduler_shift", 3.0)),
    )
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    num_steps = spec.num_steps
    if num_steps is not None:
        scheduler.set_timesteps(int(num_steps), device=spec.device)

    model = AnimaReplayModel(
        transformer=load_anima_transformer(spec),
        scheduler=scheduler,
        device=spec.device,
        dtype=resolve_torch_dtype(spec.dtype),
    )

    use_lora = spec.use_lora
    if use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    if num_steps is not None:
        model.set_num_steps(int(num_steps))

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        runtime_caps={
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": ANIMA_FAMILY,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            "transformer_path": model_config.get("transformer_path"),
            **minimal_replay_bundle_metadata(),
        },
    )


class AnimaChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Anima text-to-image rollouts."""

    family: str = ANIMA_FAMILY
    task: str = "t2i"
    family_capability = diffusion_family_capability(ANIMA_FAMILY, "t2i")
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512

    def __init__(self, model: Any, *, sample_batch_size: int = 1) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))


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
    # Search HF hub cache for the most recent snapshot containing required_file.
    hub_cache = os.environ.get("HF_HUB_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if hub_cache:
        hf_root = Path(hub_cache).expanduser()
    elif hf_home:
        hf_root = Path(hf_home).expanduser() / "hub"
    else:
        hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots_dir = hf_root / ("models--" + root.replace("/", "--")) / "snapshots"
    if snapshots_dir.is_dir():
        candidates = [
            p for p in snapshots_dir.iterdir()
            if p.is_dir() and (p / relative_file).exists()
        ]
        if candidates:
            return str(max(candidates, key=lambda p: p.stat().st_mtime) / relative_file)
    raise ValueError(
        f"model.path={root!r} is not a local root and no cached HF snapshot "
        f"contains {relative_file!r}; set model.{field_name}",
    )


def load_anima_transformer(spec: RuntimeBuildSpec) -> Any:
    from safetensors.torch import load_file

    from vrl.models.diffusion.cosmos.anima.model import (
        _load_anima_transformer,
    )
    from vrl.models.dtypes import resolve_torch_dtype

    model_config = spec.model_config or {}
    path = model_config.get("transformer_path") or _resolve_artifact(
        str(spec.model_name_or_path or ""),
        explicit_path="",
        relative_file=model_config.get("transformer_file", ""),
        field_name="transformer_path",
    )
    if not path:
        raise ValueError("Anima replay runtime spec is missing transformer_path")
    dtype = resolve_torch_dtype(spec.dtype)
    return _load_anima_transformer(load_file(path, device="cpu"), dtype=dtype).to(spec.device, dtype=dtype)


__all__ = [
    "ANIMA_FAMILY",
    "AnimaChunkExecutor",
    "build_anima_replay_runtime_bundle",
    "build_anima_runtime_bundle",
    "extract_anima_replay_runtime_spec",
    "extract_anima_runtime_spec",
    "load_anima_transformer",
]
