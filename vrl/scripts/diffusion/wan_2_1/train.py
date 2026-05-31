"""Wan 2.1 GRPO training recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME
from vrl.trainers.data.artifacts import ArtifactManifestError, resolve_artifact_path


async def train_wan_2_1_grpo(cfg: DictConfig) -> None:
    """Run Wan-family GRPO training driven by a merged YAML config."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="wan_2_1",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
        ),
    )


async def train_wan_2_1_i2v_grpo(cfg: DictConfig) -> None:
    """Run Wan I2V GRPO training driven by a merged YAML config."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="wan_2_1_i2v",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
            collector_kwargs_getter=_i2v_collector_kwargs,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.wan_2_1.runtime import build_wan_2_1_runtime_bundle_from_cfg

    return build_wan_2_1_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.wan_2_1.runtime import (
        build_wan_2_1_replay_runtime_bundle_from_cfg,
    )

    return build_wan_2_1_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    transformer = bundle.model.transformer
    if bool(cfg.actor.gradient_checkpointing):
        transformer.enable_gradient_checkpointing()


def _reference_model(bundle: Any, cfg: DictConfig) -> Any | None:
    init_kl_coef = float(getattr(cfg.algorithm, "init_kl_coef", 0.0))
    if bool(cfg.model.use_lora) and init_kl_coef > 0:
        return bundle.model
    return None


def _export_modules(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    transformer = bundle.model.transformer
    if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained"):
        return {LORA_WEIGHTS_NAME: transformer}
    return None


def _i2v_collector_kwargs(cfg: DictConfig, examples: list[Any]) -> dict[str, Any]:
    """Normalize per-sample reference images before rollout collection."""

    global_reference = str(OmegaConf.select(cfg, "model.reference_image", default="") or "")
    data_root = OmegaConf.select(cfg, "data.artifact_data_root", default=None)
    allow_absolute = bool(
        OmegaConf.select(cfg, "data.allow_absolute_artifact_paths", default=False),
    )
    manifest_path = Path(str(OmegaConf.select(cfg, "data.manifest", default="manifest")))

    for row_index, example in enumerate(examples):
        raw_path = str(getattr(example, "reference_image", "") or "").strip()
        if not raw_path:
            if global_reference:
                continue
            raise ValueError(
                f"{manifest_path}: row {row_index} is missing required field "
                "reference_image",
            )
        try:
            ref_path = resolve_artifact_path(
                raw_path,
                data_root=data_root,
                allow_absolute=allow_absolute,
            )
        except ArtifactManifestError as exc:
            raise ValueError(
                f"{manifest_path}: row {row_index} invalid reference_image",
            ) from exc
        if not ref_path.exists():
            raise FileNotFoundError(
                f"{manifest_path}: row {row_index} reference_image does not exist: "
                f"{ref_path}",
            )
        example.reference_image = str(ref_path)
        metadata = dict(getattr(example, "metadata", None) or {})
        metadata["reference_image"] = str(ref_path)
        metadata.setdefault("conditioning", "reference_image")
        example.metadata = metadata
    return {}


__all__ = ["train_wan_2_1_grpo", "train_wan_2_1_i2v_grpo"]
