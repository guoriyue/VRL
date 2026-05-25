"""Validation and required-access helpers for merged training configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

_MISSING = object()

_ALGORITHM_KINDS = {
    "grpo",
    "token_grpo",
    "token_grpo_multisegment",
    "diffusion_dpo",
    "diffusion_nft",
}

_DATA_LOADERS = {
    "pickapic_preference",
    "prompt_manifest",
}

_PROMPT_SAMPLERS = {
    "random_without_replacement",
    "sequential_window",
}

_REWARD_REQUIRED_KWARGS: dict[str, tuple[str, ...]] = {
    "aesthetic": ("model_name",),
    "clipscore": ("model_name",),
    "codex_image_qa": ("command",),
    "geneval": ("evaluator", "import_path"),
    "image_qa_cli": ("command",),
    "nsfw_safety": ("model_name", "threshold"),
    "pickscore": ("processor_name", "model_name"),
    "ocr": ("debug_dir",),
    "video_reward": ("inference_runtime", "reward_name", "score_key"),
}


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
    kind = algo.get("kind", None)
    if kind is None:
        raise ValueError("algorithm.kind required")
    kind = str(kind)
    if kind not in _ALGORITHM_KINDS:
        expected = " / ".join(sorted(_ALGORITHM_KINDS))
        raise ValueError(f"unknown algorithm.kind={kind!r}; expected {expected}")
    return kind


def validate_reward_config(cfg: DictConfig) -> None:
    """Validate reward component shape and model-backed reward kwargs."""

    if "reward" not in cfg:
        raise ValueError("config missing required field: reward")
    reward = cfg.reward
    if "components" not in reward:
        raise ValueError("config missing required field: reward.components")

    components_raw = (
        OmegaConf.to_container(
            reward.components,
            resolve=True,
            throw_on_missing=True,
        )
        or {}
    )
    if not isinstance(components_raw, dict):
        raise ValueError("reward.components must be a mapping of name -> weight")

    raw_kwargs_node = reward.get("kwargs", None) if "kwargs" in reward else None
    kwargs_dict: dict[str, Any] = (
        OmegaConf.to_container(raw_kwargs_node, resolve=True, throw_on_missing=True) or {}
        if raw_kwargs_node is not None
        else {}
    )

    for name, sub in kwargs_dict.items():
        if sub is None:
            continue
        if not isinstance(sub, dict):
            raise ValueError(
                f"reward.kwargs.{name} must be a mapping, got {type(sub).__name__}",
            )

    for name, required_subkeys in _REWARD_REQUIRED_KWARGS.items():
        if name not in components_raw:
            continue
        try:
            weight = float(components_raw[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reward.components.{name} must be numeric, got {components_raw[name]!r}",
            ) from exc
        if weight < 0:
            raise ValueError(f"reward.components.{name} must be >= 0, got {weight}")
        if weight == 0:
            continue
        sub = kwargs_dict.get(name)
        if not isinstance(sub, dict):
            raise ValueError(
                f"config missing required field: reward.kwargs.{name} "
                f"(component {name!r} has non-zero weight {weight})",
            )
        for subkey in required_subkeys:
            if subkey not in sub:
                raise ValueError(f"config missing required field: reward.kwargs.{name}.{subkey}")
        extra_validator = _REWARD_KWARGS_VALIDATORS.get(name)
        if extra_validator is not None:
            extra_validator(sub)


def _validate_video_reward_kwargs(sub: dict) -> None:
    if "backend" in sub:
        raise ValueError(
            "reward.kwargs.video_reward.backend is no longer supported; "
            "use reward.kwargs.video_reward.inference_runtime=ray",
        )
    if str(sub.get("inference_runtime")) != "ray":
        raise ValueError("reward.kwargs.video_reward.inference_runtime must be 'ray'")
    if not isinstance(sub.get("worker_config"), dict):
        raise ValueError("reward.kwargs.video_reward.worker_config must be a mapping")
    if str(sub.get("scheduling", "sync")) != "sync":
        raise ValueError("reward.kwargs.video_reward.scheduling currently supports only 'sync'")
    removed_fields = [
        k for k in ("enqueue_url", "fetch_url", "token", "poll_interval_s", "max_wait_s", "stub_scale", "device")
        if k in sub
    ]
    if removed_fields:
        raise ValueError(
            "reward.kwargs.video_reward no longer supports external reward endpoint fields: "
            + ", ".join(sorted(removed_fields)),
        )


_REWARD_KWARGS_VALIDATORS: dict[str, Any] = {
    "video_reward": _validate_video_reward_kwargs,
}


def validate_production_video_reward_config(cfg: DictConfig) -> None:
    """Validate production-only video reward gates.

    validate_reward_config() must run first — it asserts weight > 0, video_kwargs
    is a dict, and worker_config is a mapping.
    """

    video_kwargs = require(cfg, "reward.kwargs.video_reward")
    if str(video_kwargs.get("media_type", "")) != "video":
        raise ValueError("production.video_reward requires reward.kwargs.video_reward.media_type=video")
    if str(video_kwargs.get("artifact_format", "")) != "mp4":
        raise ValueError("production.video_reward requires artifact_format=mp4")
    if not str(video_kwargs.get("reward_name", "")).strip():
        raise ValueError("production.video_reward requires reward.kwargs.video_reward.reward_name")

    worker_config = video_kwargs.get("worker_config")
    forbidden_worker_keys = [
        key
        for key in (
            "backend",
            "backend_import_path",
            "backend_code_dir",
            "import_path",
            "model_subdir",
            "score_key_map",
            "model_factory",
        )
        if key in worker_config
    ]
    if forbidden_worker_keys:
        fields = ", ".join(sorted(forbidden_worker_keys))
        raise ValueError(
            "production.video_reward worker_config should name the reward model directly; "
            f"remove extra loader fields: {fields}",
        )

    task_type = str(require(cfg, "data.task_type"))
    if task_type not in {"text_to_video", "video2world"}:
        raise ValueError(
            "production.video_reward requires data.task_type=text_to_video or video2world",
        )
    for path_name in ("data.manifest", "data.eval_manifest", "data.source_report"):
        value = str(require(cfg, path_name)).strip()
        if not value:
            raise ValueError(f"config missing required field: {path_name}")
        if not Path(value).exists():
            raise ValueError(f"{path_name} does not exist: {value}")
    if task_type == "video2world":
        _validate_video_world_production_data(cfg)


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


def validate_data_config(cfg: DictConfig) -> None:
    """Validate dataset loader, preprocessing, and sampler contracts."""

    if "data" not in cfg:
        raise ValueError("config missing required field: data")

    loader = str(require(cfg, "data.loader"))
    if loader not in _DATA_LOADERS:
        expected = " / ".join(sorted(_DATA_LOADERS))
        raise ValueError(f"unknown data.loader={loader!r}; expected {expected}")

    if loader == "prompt_manifest":
        require(cfg, "data.manifest")
        require(cfg, "data.preprocessing")
        sampler_type = str(require(cfg, "data.sampler.type"))
        if sampler_type not in _PROMPT_SAMPLERS:
            expected = " / ".join(sorted(_PROMPT_SAMPLERS))
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


def validate_training_config(cfg: DictConfig) -> None:
    """Validate only unresolved mandatory values and cross-field contracts."""

    assert_no_missing(cfg)
    if "algorithm" not in cfg:
        raise ValueError("config missing required field: algorithm")
    kind = resolve_algorithm_kind(cfg.algorithm)

    validate_data_config(cfg)

    if "reward" in cfg:
        validate_reward_config(cfg)
    if bool(OmegaConf.select(cfg, "production.video_reward.enabled", default=False)):
        validate_production_video_reward_config(cfg)

    if kind in {"grpo", "diffusion_nft"}:
        sde_type = str(require(cfg, "rollout.sde.type"))
        if sde_type not in {"sde", "cps"}:
            raise ValueError("rollout.sde.type must be 'sde' or 'cps'")

    if kind == "token_grpo":
        family = cfg.model.family if path_exists(cfg, "model.family") else None
        if family == "nextstep_1":
            require(cfg, "rollout.noise_level")

    if kind == "token_grpo_multisegment":
        family = str(require(cfg, "model.family"))
        if family != "janus_pro":
            raise ValueError("token_grpo_multisegment currently requires model.family=janus_pro")
        final_image_policy = str(require(cfg, "rollout.final_image_policy"))
        if final_image_policy not in {"always_generate", "use_selfcheck"}:
            raise ValueError(
                "rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'",
            )
        sampling_final_policy = str(require(cfg, "sampling.r1.final_image_policy"))
        if sampling_final_policy != final_image_policy:
            raise ValueError(
                "sampling.r1.final_image_policy must match rollout.final_image_policy",
            )

    if kind == "diffusion_dpo":
        optional_none(cfg, "data.max_train_samples")


__all__ = [
    "assert_no_missing",
    "optional_none",
    "path_exists",
    "require",
    "resolve_algorithm_kind",
    "validate_data_config",
    "validate_production_video_reward_config",
    "validate_reward_config",
    "validate_training_config",
]
