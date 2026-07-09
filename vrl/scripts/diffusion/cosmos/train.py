"""Cosmos online training recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.common.online import (
    default_reference_model,
    enable_transformer_gradient_checkpointing,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.data.artifacts import ArtifactManifestError, resolve_artifact_path


async def train_cosmos_predict2_grpo(cfg: DictConfig) -> None:
    """Run Cosmos Predict2 GRPO through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2",
            build_bundle=_build_predict2_bundle,
            build_replay_bundle=_build_predict2_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
            collector_kwargs_getter=_predict2_collector_kwargs,
        ),
    )


async def train_cosmos_predict25_grpo(cfg: DictConfig) -> None:
    """Run Cosmos Predict2.5 GRPO through the common online recipe.

    The paper's RL recipe (§4.2.2) is GRPO; the NFT entrypoint below is the
    likelihood-free variant. Predict2.5 is text-to-world, so unlike the
    Predict2 Video2World entrypoint there is no reference-image collector
    wiring here.
    """

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2.5",
            build_bundle=_build_predict25_bundle,
            build_replay_bundle=_build_predict25_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
        ),
    )


async def train_cosmos_predict25_diffusion_nft(cfg: DictConfig) -> None:
    """Run Cosmos Predict2.5 DiffusionNFT through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2.5",
            build_bundle=_build_predict25_bundle,
            build_replay_bundle=_build_predict25_replay_bundle,
            after_bundle_built=_after_bundle_built,
            export_modules_getter=export_transformer_lora,
        ),
    )


def _build_predict2_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.build import build_family_runtime_bundle_from_cfg

    return build_family_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_predict2_replay_bundle(
    cfg: DictConfig,
    device: Any,
    weight_dtype: Any,
) -> Any:
    from vrl.models.diffusion.build import (
        build_family_replay_runtime_bundle_from_cfg,
    )

    return build_family_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_predict25_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.build import build_family_runtime_bundle_from_cfg

    return build_family_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_predict25_replay_bundle(
    cfg: DictConfig,
    device: Any,
    weight_dtype: Any,
) -> Any:
    from vrl.models.diffusion.build import (
        build_family_replay_runtime_bundle_from_cfg,
    )

    return build_family_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    enable_transformer_gradient_checkpointing(bundle, cfg, require_method=False)


def _predict2_collector_kwargs(cfg: DictConfig, examples: Any) -> dict[str, Any]:
    mode = str(OmegaConf.select(cfg, "cosmos.reference_mode", default="global"))
    if mode not in {"global", "per_sample"}:
        raise ValueError("cosmos.reference_mode must be 'global' or 'per_sample'")
    if mode == "global":
        # Pass the validated path, not a loaded PIL image: the Ray launch
        # contract only accepts primitive executor kwargs, and the worker-side
        # executor loads paths itself (load_reference_image in its __init__).
        path_text = str(cfg.model.reference_image or "").strip()
        if not path_text:
            raise ValueError(
                "Cosmos Predict2 Video2World GRPO requires model.reference_image "
                "unless cosmos.reference_mode=per_sample",
            )
        path = Path(path_text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"model.reference_image does not exist: {path}")
        return {"reference_image": str(path)}
    _normalize_per_sample_reference_images(
        examples,
        manifest_path=Path(str(cfg.data.manifest)),
        prompts_per_batch=int(cfg.rollout.prompts_per_batch),
    )
    return {}


def _normalize_per_sample_reference_images(
    examples: Any,
    *,
    manifest_path: Path,
    prompts_per_batch: int,
) -> None:
    if prompts_per_batch != 1:
        raise ValueError(
            "cosmos.reference_mode=per_sample currently requires "
            "trainer.prompts_per_batch=1",
        )
    for idx, example in enumerate(examples):
        raw_path = str(getattr(example, "reference_image", "") or "").strip()
        if not raw_path:
            raise ValueError(
                f"{manifest_path}: row {idx} is missing required field reference_image",
            )
        try:
            ref_path = resolve_artifact_path(raw_path)
        except ArtifactManifestError as exc:
            raise ValueError(
                f"{manifest_path}: row {idx} invalid reference_image",
            ) from exc
        if not ref_path.exists():
            raise FileNotFoundError(
                f"{manifest_path}: row {idx} reference_image does not exist: {ref_path}",
            )
        example.reference_image = str(ref_path)
        metadata = dict(getattr(example, "metadata", None) or {})
        metadata["reference_image"] = str(ref_path)
        example.metadata = metadata


__all__ = [
    "train_cosmos_predict2_grpo",
    "train_cosmos_predict25_diffusion_nft",
    "train_cosmos_predict25_grpo",
]
