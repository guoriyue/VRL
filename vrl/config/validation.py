"""Validation and required-access helpers for merged training configs.

Public API is preserved unchanged. Whitelist sets (_ALGORITHM_KINDS, _DATA_LOADERS,
_PROMPT_SAMPLERS, _REWARD_REQUIRED_KWARGS, _REWARD_KWARGS_VALIDATORS) have moved to
schema.py. Structural validation now flows through parse_config() -> RootConfig.
File-existence checks remain here because they must not enter the Pydantic schema (D4).
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
    DataConfig,
    RewardConfig,
    _extract_error_message,
    parse_config,
)

_MISSING = object()


def require(cfg: DictConfig, path: str) -> Any:
    """Fetch a required dotted path from a config.

    YAML should declare experiment-owned required values with ``???``. This
    helper keeps a stable repo-level error message around OmegaConf's missing
    value semantics.
    """
    try:
        node = OmegaConf.select(cfg, path, default=_MISSING, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", path) or path
        raise ValueError(f"config missing required field: {missing_path}") from exc
    if node is _MISSING:
        raise ValueError(f"config missing required field: {path}")
    if node is None:
        raise ValueError(f"config missing required field: {path} (got None)")
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    return node


def optional_none(cfg: DictConfig, path: str) -> Any | None:
    """Fetch a dotted path that may explicitly be ``null``."""
    try:
        node = OmegaConf.select(cfg, path, default=_MISSING, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", path) or path
        raise ValueError(f"config missing required field: {missing_path}") from exc
    if node is _MISSING:
        raise ValueError(f"config missing required field: {path}")
    if node is None:
        return None
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    return node


def path_exists(cfg: DictConfig, path: str) -> bool:
    """Return true if a dotted path is present, even when its value is ``???``."""
    keys = path.split(".")
    node: Any = cfg
    for key in keys:
        if not isinstance(node, DictConfig) or key not in node:
            return False
        node = node[key]
    return True


def assert_no_missing(cfg: DictConfig) -> None:
    """Fail if a merged config still contains OmegaConf mandatory values."""
    try:
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", None) or str(exc)
        raise ValueError(f"config missing required field: {missing_path}") from exc


def resolve_algorithm_kind(algo: DictConfig) -> str:
    """Resolve and validate the algorithm dispatch key."""
    if "adv_estimator" in algo:
        raise ValueError("algorithm.adv_estimator is no longer supported; use algorithm.kind")
    if "per_prompt_stat_tracking" in algo:
        raise ValueError(
            "algorithm.per_prompt_stat_tracking is no longer supported; "
            "use actor.drop_zero_advantage",
        )
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
    reward_raw = (
        OmegaConf.to_container(cfg.reward, resolve=True, throw_on_missing=True) or {}
    )
    try:
        RewardConfig.model_validate(reward_raw)
    except ValidationError as exc:
        raise ValueError(_extract_error_message(exc)) from exc


def validate_data_config(cfg: DictConfig) -> None:
    """Validate dataset loader, preprocessing, and sampler contracts."""
    if "data" not in cfg:
        raise ValueError("config missing required field: data")

    loader = str(require(cfg, "data.loader"))
    valid_loaders = frozenset(get_args(DataConfig.model_fields["loader"].annotation))
    if loader not in valid_loaders:
        expected = " / ".join(sorted(valid_loaders))
        raise ValueError(f"unknown data.loader={loader!r}; expected {expected}")

    if loader == "prompt_manifest":
        require(cfg, "data.manifest")
        require(cfg, "data.preprocessing")
        sampler_type = str(require(cfg, "data.sampler.type"))
        valid_samplers = {"random_without_replacement", "sequential_window"}
        if sampler_type not in valid_samplers:
            expected = " / ".join(sorted(valid_samplers))
            raise ValueError(f"unknown data.sampler.type={sampler_type!r}; expected {expected}")
        return

    if loader == "prompt_image_manifest":
        require(cfg, "data.manifest")
        require(cfg, "data.eval_manifest")
        require(cfg, "data.preprocessing.format")
        require(cfg, "data.preprocessing.image_field")
        require(cfg, "data.preprocessing.caption_field")
        require(cfg, "data.preprocessing.media_type")
        require(cfg, "data.preprocessing.conditioning")
        sampler_type = str(require(cfg, "data.sampler.type"))
        valid_samplers = {"random_without_replacement", "sequential_window"}
        if sampler_type not in valid_samplers:
            expected = " / ".join(sorted(valid_samplers))
            raise ValueError(f"unknown data.sampler.type={sampler_type!r}; expected {expected}")
        return

    if loader == "pickapic_preference":
        require(cfg, "data.dataset_name")
        require(cfg, "data.split")
        require(cfg, "data.cache_dir")
        optional_none(cfg, "data.max_train_samples")
        require(cfg, "data.preprocessing.resolution")
        require(cfg, "data.preprocessing.random_crop")
        require(cfg, "data.preprocessing.horizontal_flip")
        require(cfg, "data.sampler.shuffle")
        require(cfg, "data.sampler.drop_last")
        require(cfg, "data.sampler.dataloader_num_workers")
        return

    raise AssertionError(f"unreachable: loader={loader}")  # pragma: no cover


def validate_production_kling_video_reward_config(cfg: DictConfig) -> None:
    """Check file existence for production Kling VideoReward paths.

    Structural rules (media_type, artifact_format, reward_name, worker_config
    forbidden fields, data.task_type) are handled by the typed schema inside
    parse_config(). This function only checks path existence and data integrity,
    which must not enter the schema per D4.
    """
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


def _validate_video_world_production_data(cfg: DictConfig) -> None:
    from vrl.trainers.data.artifacts import validate_source_backed_video_world_manifest_pair

    data_root = str(OmegaConf.select(cfg, "data.artifact_data_root", default="") or "").strip()
    kwargs = {"data_root": data_root} if data_root else {}
    validate_source_backed_video_world_manifest_pair(
        str(require(cfg, "data.manifest")),
        str(require(cfg, "data.eval_manifest")),
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
    from vrl.trainers.data.artifacts import resolve_artifact_path

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
    parse_config(cfg)
    if bool(
        OmegaConf.select(cfg, "production.kling_video_reward.enabled", default=False)
        or OmegaConf.select(cfg, "production.video_reward.enabled", default=False)
    ):
        validate_production_kling_video_reward_config(cfg)


validate_production_video_reward_config = validate_production_kling_video_reward_config


__all__ = [
    "assert_no_missing",
    "optional_none",
    "path_exists",
    "require",
    "resolve_algorithm_kind",
    "validate_data_config",
    "validate_production_kling_video_reward_config",
    "validate_production_video_reward_config",
    "validate_reward_config",
    "validate_training_config",
]
