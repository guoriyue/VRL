"""SD 3.5 GRPO training recipe."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.recipes.online import run_online_recipe
from vrl.scripts.recipes.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME


async def train_sd3_5_grpo(cfg: DictConfig) -> None:
    """Run SD 3.5 GRPO training driven by the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="sd3_5",
            build_bundle=_build_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
            weight_dtype_getter=_resolve_weight_dtype,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.families.sd3_5.runtime import build_sd3_5_runtime_bundle_from_cfg

    return build_sd3_5_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    transformer = bundle.policy.transformer
    if bool(cfg.actor.gradient_checkpointing):
        transformer.enable_gradient_checkpointing()
    _offload_driver_frozen_modules(bundle.policy)


def _reference_model(bundle: Any, cfg: DictConfig) -> Any | None:
    init_kl_coef = float(getattr(cfg.algorithm, "init_kl_coef", 0.0))
    if bool(cfg.model.use_lora) and init_kl_coef > 0:
        return bundle.policy
    return None


def _export_modules(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    transformer = bundle.policy.transformer
    if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained"):
        return {LORA_WEIGHTS_NAME: transformer}
    return None


def _resolve_weight_dtype(cfg: DictConfig, trainer_config: Any, torch: Any) -> Any:
    del cfg
    precision = str(trainer_config.mixed_precision or "").lower().strip()
    aliases = {
        "none": "no",
        "off": "no",
        "fp32": "no",
        "float32": "no",
        "float16": "fp16",
        "bfloat16": "bf16",
    }
    precision = aliases.get(precision, precision)
    if not precision:
        precision = "bf16" if trainer_config.bf16 else "no"
    if precision == "no":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(
        "actor.mixed_precision must be one of 'no', 'fp16', or 'bf16', "
        f"got {trainer_config.mixed_precision!r}",
    )


def _offload_driver_frozen_modules(policy: object) -> None:
    """Move frozen driver-only modules off CUDA before Ray workers load."""

    import torch

    pipeline = getattr(policy, "pipeline", None)
    if pipeline is None:
        return

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["train_sd3_5_grpo"]
