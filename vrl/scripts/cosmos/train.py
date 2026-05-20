"""Cosmos online training recipes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME


async def train_cosmos_predict2_grpo(cfg: DictConfig) -> None:
    """Run Cosmos Predict2 GRPO through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2",
            build_bundle=_build_predict2_bundle,
            build_replay_bundle=_build_predict2_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
            collector_kwargs_getter=_predict2_collector_kwargs,
        ),
    )


async def train_cosmos_predict25_diffusion_nft(cfg: DictConfig) -> None:
    """Run Cosmos Predict2.5 DiffusionNFT through the common online recipe."""

    validation_state: dict[str, Any] = {}
    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2.5",
            build_bundle=_build_predict25_bundle,
            build_replay_bundle=_build_predict25_replay_bundle,
            after_bundle_built=_after_bundle_built,
            export_modules_getter=_export_modules,
            before_step=lambda stack, epoch, examples: _capture_optimization_check_before(
                validation_state,
                stack,
                epoch,
            ),
            after_step=lambda stack, epoch, examples: _write_optimization_check_after(
                validation_state,
                stack,
                epoch,
            ),
            metric_row_hook=lambda row, metrics: _capture_optimization_check_metrics(
                validation_state,
                row,
                metrics,
            ),
        ),
    )


async def train_anima_grpo(cfg: DictConfig) -> None:
    """Run Anima Preview3 GRPO through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2-anima",
            build_bundle=_build_anima_bundle,
            build_replay_bundle=_build_anima_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
        ),
    )


def _build_predict2_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.cosmos.predict2.runtime import (
        build_cosmos_predict2_runtime_bundle_from_cfg,
    )

    return build_cosmos_predict2_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_predict2_replay_bundle(
    cfg: DictConfig,
    device: Any,
    weight_dtype: Any,
) -> Any:
    from vrl.models.diffusion.cosmos.predict2.runtime import (
        build_cosmos_predict2_replay_runtime_bundle_from_cfg,
    )

    return build_cosmos_predict2_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_predict25_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.cosmos.predict2_5.runtime import (
        build_cosmos_predict25_runtime_bundle_from_cfg,
    )

    return build_cosmos_predict25_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_predict25_replay_bundle(
    cfg: DictConfig,
    device: Any,
    weight_dtype: Any,
) -> Any:
    from vrl.models.diffusion.cosmos.predict2_5.runtime import (
        build_cosmos_predict25_replay_runtime_bundle_from_cfg,
    )

    return build_cosmos_predict25_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_anima_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.cosmos.anima.runtime import (
        build_anima_runtime_bundle_from_cfg,
    )

    return build_anima_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_anima_replay_bundle(
    cfg: DictConfig,
    device: Any,
    weight_dtype: Any,
) -> Any:
    from vrl.models.diffusion.cosmos.anima.runtime import (
        build_anima_replay_runtime_bundle_from_cfg,
    )

    return build_anima_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    transformer = bundle.model.transformer
    if bool(cfg.actor.gradient_checkpointing) and hasattr(
        transformer,
        "enable_gradient_checkpointing",
    ):
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


def _capture_optimization_check_before(
    state: dict[str, Any],
    stack: Any,
    epoch: int,
) -> None:
    if epoch != 0 or not _uses_video_reward(stack.cfg):
        return
    state.setdefault("trainable_sha256_before", _trainable_modules_sha256(stack.bundle))


def _capture_optimization_check_metrics(
    state: dict[str, Any],
    row: dict[str, Any],
    metrics: Any,
) -> None:
    state["last_metric_row"] = dict(row)
    state["last_metrics"] = {
        "reward_mean": float(getattr(metrics, "reward_mean", 0.0)),
        "reward_std": float(getattr(metrics, "reward_std", 0.0)),
        "advantage_mean": float(getattr(metrics, "advantage_mean", 0.0)),
        "grad_norm": float(getattr(metrics, "grad_norm", 0.0)),
    }


def _write_optimization_check_after(
    state: dict[str, Any],
    stack: Any,
    epoch: int,
) -> None:
    if epoch != 0 or not _uses_video_reward(stack.cfg) or state.get("wrote_optimization_check"):
        return

    before = state.get("trainable_sha256_before")
    if before is None:
        before = _trainable_modules_sha256(stack.bundle)
    after = _trainable_modules_sha256(stack.bundle)
    metrics = dict(state.get("last_metrics", {}))
    row = dict(state.get("last_metric_row", {}))
    video_kwargs = _video_reward_kwargs(stack.cfg)
    artifact_dir = Path(str(video_kwargs.get("artifact_dir", "")))
    debug_dir = Path(str(video_kwargs.get("debug_dir", "")))
    output_dir = Path(stack.output_dir)
    payload = {
        "global_step": int(getattr(stack.trainer.state, "global_step", 0)),
        "trainer_step": int(getattr(stack.trainer.state, "step", 0)),
        "epoch": int(epoch),
        "grad_norm": float(metrics.get("grad_norm", row.get("grad_norm", 0.0))),
        "reward_mean": float(metrics.get("reward_mean", row.get("reward_mean", 0.0))),
        "reward_std": float(metrics.get("reward_std", row.get("reward_std", 0.0))),
        "advantage_mean": float(
            metrics.get("advantage_mean", row.get("advantage_mean", 0.0)),
        ),
        "trainable_sha256_before": str(before),
        "trainable_sha256_after": str(after),
        "trainable_sha256_changed": str(before) != str(after),
        "reward_runtime": str(video_kwargs.get("inference_runtime", "")),
        "reward_name": str(video_kwargs.get("reward_name", "")),
        "reward_model_version": str(
            (video_kwargs.get("worker_config") or {}).get("reward_model_version", ""),
        ),
        "reward_artifacts_manifest": str(artifact_dir / "manifest.jsonl"),
        "reward_debug_requests": str(debug_dir / "video_reward_requests.jsonl"),
        "reward_debug_results": str(debug_dir / "video_reward_results.jsonl"),
        "nonzero_advantage_batch": bool(
            abs(float(metrics.get("advantage_mean", row.get("advantage_mean", 0.0)))) > 0.0
            or float(metrics.get("reward_std", row.get("reward_std", 0.0))) > 0.0
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimization_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state["wrote_optimization_check"] = True


def _trainable_modules_sha256(bundle: Any) -> str:
    modules = getattr(bundle, "trainable_modules", None)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Cosmos optimization check requires bundle.trainable_modules")
    digest = hashlib.sha256()
    for module_name in sorted(modules):
        state_dict = getattr(modules[module_name], "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"trainable module {module_name!r} does not expose state_dict()")
        state = state_dict()
        if not isinstance(state, dict):
            raise TypeError(f"trainable module {module_name!r} state_dict() must return a dict")
        for key in sorted(state):
            value = state[key]
            digest.update(module_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(key).encode("utf-8"))
            digest.update(b"\0")
            _hash_state_value(digest, value)
    return digest.hexdigest()


def _hash_state_value(digest: Any, value: Any) -> None:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        import io

        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        digest.update(buffer.getvalue())
        return
    digest.update(repr(value).encode("utf-8"))


def _uses_video_reward(cfg: DictConfig) -> bool:
    try:
        weight = float(OmegaConf.select(cfg, "reward.components.video_reward", default=0.0))
    except (TypeError, ValueError):
        return False
    runtime = str(
        OmegaConf.select(
            cfg,
            "reward.kwargs.video_reward.inference_runtime",
            default="",
        ),
    )
    return weight > 0 and runtime == "ray"


def _video_reward_kwargs(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.select(cfg, "reward.kwargs.video_reward", default={}) or {}
    if hasattr(raw, "_metadata"):
        return dict(OmegaConf.to_container(raw, resolve=True) or {})
    return dict(raw)


def _predict2_collector_kwargs(cfg: DictConfig, examples: Any) -> dict[str, Any]:
    mode = str(OmegaConf.select(cfg, "cosmos.reference_mode", default="global"))
    if mode not in {"global", "per_sample"}:
        raise ValueError("cosmos.reference_mode must be 'global' or 'per_sample'")
    if mode == "global":
        return {"reference_image": _load_required_reference_image(str(cfg.model.reference_image))}
    _normalize_per_sample_reference_images(
        examples,
        manifest_path=Path(str(cfg.data.manifest)),
        rollout_batch_size=int(cfg.trainer.rollout_batch_size),
    )
    return {}


def _load_required_reference_image(reference_image_path: str) -> Any:
    path_text = str(reference_image_path or "").strip()
    if not path_text:
        raise ValueError(
            "Cosmos Predict2 Video2World GRPO requires model.reference_image "
            "unless cosmos.reference_mode=per_sample",
        )

    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model.reference_image does not exist: {path}")

    from PIL import Image

    return Image.open(path).convert("RGB")


def _normalize_per_sample_reference_images(
    examples: Any,
    *,
    manifest_path: Path,
    rollout_batch_size: int,
) -> None:
    if rollout_batch_size != 1:
        raise ValueError(
            "cosmos.reference_mode=per_sample currently requires "
            "trainer.rollout_batch_size=1",
        )
    base_dir = manifest_path.parent
    for idx, example in enumerate(examples):
        raw_path = str(getattr(example, "reference_image", "") or "").strip()
        if not raw_path:
            raise ValueError(
                f"{manifest_path}: row {idx} is missing required field reference_image",
            )
        ref_path = Path(raw_path).expanduser()
        if not ref_path.is_absolute():
            ref_path = (base_dir / ref_path).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(
                f"{manifest_path}: row {idx} reference_image does not exist: {ref_path}",
            )
        example.reference_image = str(ref_path)
        metadata = dict(getattr(example, "metadata", None) or {})
        metadata["reference_image"] = str(ref_path)
        example.metadata = metadata


__all__ = [
    "train_anima_grpo",
    "train_cosmos_predict2_grpo",
    "train_cosmos_predict25_diffusion_nft",
]
