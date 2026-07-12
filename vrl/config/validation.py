"""Validation and required-access helpers for merged training configs.

Structural validation flows through parse_config() -> RootConfig (schema.py).
This module owns the dotted-path access helpers (require / optional_none /
path_exists) and the production gates whose file-existence and manifest checks
must not enter the Pydantic schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from vrl.config.schema import (
    AlgorithmConfig,
    RewardConfig,
    _extract_error_message,
    parse_config,
)

_MISSING = object()
_REQUIRED = object()


def _select_field(cfg: DictConfig, path: str, *, on_none: Any) -> Any:
    """Resolve a dotted path, sharing the missing/None/container handling.

    ``require`` and ``optional_none`` differ only in what happens when the
    resolved node is ``None``: pass the ``_REQUIRED`` sentinel to reject it
    like a missing field, or any other value to return it.
    """
    try:
        node = OmegaConf.select(cfg, path, default=_MISSING, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", path) or path
        raise ValueError(f"config missing required field: {missing_path}") from exc
    if node is _MISSING:
        raise ValueError(f"config missing required field: {path}")
    if node is None:
        if on_none is _REQUIRED:
            raise ValueError(f"config missing required field: {path} (got None)")
        return on_none
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    return node


def require(cfg: DictConfig, path: str) -> Any:
    """Fetch a required dotted path from a config.

    YAML should declare experiment-owned required values with ``???``. This
    helper keeps a stable repo-level error message around OmegaConf's missing
    value semantics.
    """
    return _select_field(cfg, path, on_none=_REQUIRED)


def optional_none(cfg: DictConfig, path: str) -> Any | None:
    """Fetch a dotted path that may explicitly be ``null``."""
    return _select_field(cfg, path, on_none=None)


def path_exists(cfg: DictConfig, path: str) -> bool:
    """Return true if a dotted path is present, even when its value is ``???``."""
    keys = path.split(".")
    node: Any = cfg
    for key in keys:
        if not isinstance(node, DictConfig) or key not in node:
            return False
        node = node[key]
    return True


def resolve_algorithm_kind(algo: DictConfig) -> str:
    """Resolve and validate the algorithm dispatch key."""
    kind = algo.get("kind", None)
    if kind is None:
        raise ValueError("algorithm.kind required")
    kind = str(kind)
    valid = frozenset(get_args(AlgorithmConfig.model_fields["kind"].annotation))
    if kind not in valid:
        expected = " / ".join(sorted(valid))
        raise ValueError(f"unknown algorithm.kind={kind!r}; expected {expected}")
    return kind


def validate_reward_config(cfg: DictConfig) -> None:
    """Validate reward component shape and model-backed reward kwargs."""
    if "reward" not in cfg:
        raise ValueError("config missing required field: reward")
    from vrl.config.unknown_keys import warn_unknown_keys

    warn_unknown_keys(cfg.reward, section="reward")
    reward_raw = OmegaConf.to_container(cfg.reward, resolve=True, throw_on_missing=True) or {}
    try:
        RewardConfig.model_validate(reward_raw)
    except ValidationError as exc:
        raise ValueError(_extract_error_message(exc)) from exc


def validate_production_kling_video_reward_config(cfg: DictConfig) -> None:
    """Production Kling VideoReward gate: structural contract + path existence."""
    validate_production_reward_contract(cfg)
    for path_name in ("data.manifest", "data.eval_manifest", "data.source_report"):
        value = str(require(cfg, path_name)).strip()
        if not value:
            raise ValueError(f"config missing required field: {path_name}")
        if not Path(value).exists():
            raise ValueError(f"{path_name} does not exist: {value}")
    task_type = str(OmegaConf.select(cfg, "data.task_type", default="") or "")
    if task_type == "video2world":
        _validate_video_world_production_data(cfg)
    if task_type == "image_to_video":
        _validate_image_to_video_production_data(cfg)


def validate_production_reward_contract(cfg: DictConfig) -> None:
    """Structural production contract for the Kling VideoReward.

    Reads the raw cfg directly — per-reward config knowledge deliberately does
    not live in the schema (rewards own their contracts at construction; this
    gate exists because production misconfiguration is unrecoverable mid-run).
    """
    vr_kwargs = OmegaConf.select(cfg, "reward.kwargs.kling_video_reward") or {}
    if str(vr_kwargs.get("media_type", "")) != "video":
        raise ValueError(
            "production.kling_video_reward requires "
            "reward.kwargs.kling_video_reward.media_type=video"
        )
    if str(vr_kwargs.get("artifact_format", "")) != "mp4":
        raise ValueError("production.kling_video_reward requires artifact_format=mp4")
    if not str(vr_kwargs.get("reward_name", "")).strip():
        raise ValueError(
            "production.kling_video_reward requires reward.kwargs.kling_video_reward.reward_name"
        )
    worker_config = vr_kwargs.get("worker_config") or {}
    forbidden = sorted(
        k
        for k in (
            "backend",
            "backend_import_path",
            "backend_code_dir",
            "import_path",
            "model_subdir",
            "score_key_map",
            "model_factory",
        )
        if k in worker_config
    )
    if forbidden:
        raise ValueError(
            "production.kling_video_reward worker_config should name the reward "
            "model directly; "
            f"remove extra loader fields: {', '.join(forbidden)}",
        )
    task_type = str(OmegaConf.select(cfg, "data.task_type", default="") or "")
    if task_type not in {"text_to_video", "image_to_video", "video2world"}:
        raise ValueError(
            "production.kling_video_reward requires "
            "data.task_type=text_to_video, image_to_video, or video2world"
        )


def _validate_video_world_production_data(cfg: DictConfig) -> None:
    from vrl.trainers.data.artifacts import validate_source_backed_video_world_manifest_pair

    data_root = str(OmegaConf.select(cfg, "data.artifact_data_root", default="") or "").strip()
    kwargs = {"data_root": data_root} if data_root else {}
    reward_components = OmegaConf.select(cfg, "reward.components", default={}) or {}
    # The target-clip-reading reward is target_dino_similarity (successor to the deleted
    # pixel-L1 target_video_similarity); it consumes metadata['target_video'], so its
    # presence is what makes target clips a hard manifest requirement.
    require_target_video = "target_dino_similarity" in reward_components
    validate_source_backed_video_world_manifest_pair(
        str(require(cfg, "data.manifest")),
        str(require(cfg, "data.eval_manifest")),
        require_target_video=require_target_video,
        **kwargs,
    )
    _validate_video_world_source_report(Path(str(require(cfg, "data.source_report"))))


def _validate_video_world_source_report(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "dataset",
        "source",
        "repo_id",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
        "validation_summary",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(
            f"data.source_report is missing Video2World provenance fields: {missing}",
        )
    if int(payload.get("train_rows") or 0) <= 0 or int(payload.get("eval_rows") or 0) <= 0:
        raise ValueError("data.source_report must record non-empty train and eval rows")
    validation_summary = payload.get("validation_summary")
    if not isinstance(validation_summary, dict) or not validation_summary:
        raise ValueError("data.source_report must include a non-empty validation_summary")


def _validate_image_to_video_production_data(cfg: DictConfig) -> None:
    data_root = str(OmegaConf.select(cfg, "data.artifact_data_root", default="") or "").strip()
    if not data_root:
        raise ValueError("config missing required field: data.artifact_data_root")
    preprocessing = OmegaConf.select(cfg, "data.preprocessing", default={}) or {}
    image_field = str(preprocessing.get("image_field", "image"))
    caption_field = str(preprocessing.get("caption_field", "caption"))
    train_count = _validate_image_to_video_manifest(
        Path(str(require(cfg, "data.manifest"))),
        data_root=Path(data_root),
        image_field=image_field,
        caption_field=caption_field,
    )
    eval_count = _validate_image_to_video_manifest(
        Path(str(require(cfg, "data.eval_manifest"))),
        data_root=Path(data_root),
        image_field=image_field,
        caption_field=caption_field,
    )
    _validate_image_to_video_source_report(
        Path(str(require(cfg, "data.source_report"))),
        train_count=train_count,
        eval_count=eval_count,
    )


def _validate_image_to_video_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    image_field: str,
    caption_field: str,
) -> int:
    from vrl.utils.artifacts import resolve_artifact_path

    # Shares the {source_repo, source_frame_index, decode_method, conditioning}
    # provenance sub-vocabulary with
    # vrl/trainers/data/artifacts.py SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS —
    # keep in sync. This is a separate schema (Image2Video manifest rows use
    # source_video_url, not source_video), so the two are not unified.
    required_metadata = {
        "source_repo",
        "source_video_url",
        "source_frame_index",
        "decode_method",
        "conditioning",
    }
    row_count = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}: row {row_index} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}: row {row_index} must be an object")
            image = str(row.get(image_field, "")).strip()
            caption = str(row.get(caption_field, "")).strip()
            if not image:
                raise ValueError(f"{manifest_path}: row {row_index} missing {image_field}")
            if not caption:
                raise ValueError(f"{manifest_path}: row {row_index} missing {caption_field}")
            resolved_image = resolve_artifact_path(image, data_root=data_root)
            if not resolved_image.exists():
                raise ValueError(
                    f"{manifest_path}: row {row_index} image does not exist: {resolved_image}",
                )
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{manifest_path}: row {row_index} metadata is required")
            missing = sorted(
                field
                for field in required_metadata
                if metadata.get(field) is None or str(metadata.get(field)).strip() == ""
            )
            if missing:
                raise ValueError(
                    f"{manifest_path}: row {row_index} missing source metadata: {missing}",
                )
    if row_count == 0:
        raise ValueError(f"{manifest_path} must contain at least one image-to-video row")
    return row_count


def _validate_image_to_video_source_report(
    path: Path,
    *,
    train_count: int,
    eval_count: int,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Report-level (dataset-wide) schema — distinct from the per-row provenance
    # vocabulary in _validate_image_to_video_manifest and
    # vrl/trainers/data/artifacts.py SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS.
    # Do NOT fold this into those: these keys describe the whole source dump
    # (source_csv, train_rows, reference_dir), not a single manifest row.
    required_keys = {
        "dataset",
        "source_repo",
        "source_csv",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(
            f"data.source_report is missing Image2Video provenance fields: {missing}",
        )
    if payload.get("dataset") != "videophy_i2v":
        raise ValueError("data.source_report dataset must be videophy_i2v")
    if int(payload.get("train_rows") or 0) != train_count:
        raise ValueError("data.source_report train_rows does not match data.manifest")
    if int(payload.get("eval_rows") or 0) != eval_count:
        raise ValueError("data.source_report eval_rows does not match data.eval_manifest")


def validate_training_config(cfg: DictConfig) -> None:
    """Validate unresolved mandatory values and cross-field contracts."""
    from vrl.config.unknown_keys import warn_unknown_keys

    warn_unknown_keys(cfg)
    parse_config(cfg)
    from vrl.config.precision import resolve_precision_policy

    resolve_precision_policy(cfg)
    # compile x grad-checkpointing is refused at trainer startup; check it here
    # too so the collision fails at config load (where the all-experiments test
    # sees it) — a model-layer torch_compile.enable=true default can silently
    # flip compile on underneath an experiment that needs checkpointing.
    from vrl.trainers.activation_checkpointing import require_compile_checkpointing_compatible

    require_compile_checkpointing_compatible(cfg)
    if bool(OmegaConf.select(cfg, "production.kling_video_reward.enabled", default=False)):
        validate_production_kling_video_reward_config(cfg)


__all__ = [
    "optional_none",
    "path_exists",
    "require",
    "resolve_algorithm_kind",
    "validate_production_kling_video_reward_config",
    "validate_production_reward_contract",
    "validate_reward_config",
    "validate_training_config",
]
