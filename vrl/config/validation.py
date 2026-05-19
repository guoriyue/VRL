"""Validation and required-access helpers for merged training configs."""

from __future__ import annotations

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

_REWARD_REQUIRED_KWARGS: dict[str, tuple[str, ...]] = {
    "aesthetic": ("model_name",),
    "clipscore": ("model_name",),
    "codex_image_qa": ("command",),
    "image_qa_cli": ("command",),
    "pickscore": ("processor_name", "model_name"),
    "ocr": ("debug_dir",),
    "video_reward": ("backend", "reward_name", "score_key"),
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

    components_raw = OmegaConf.to_container(
        reward.components,
        resolve=True,
        throw_on_missing=True,
    ) or {}
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
        if weight <= 0:
            continue
        sub = kwargs_dict.get(name)
        if not isinstance(sub, dict):
            raise ValueError(
                f"config missing required field: reward.kwargs.{name} "
                f"(component {name!r} has weight {weight} > 0)",
            )
        for subkey in required_subkeys:
            if subkey not in sub:
                raise ValueError(f"config missing required field: reward.kwargs.{name}.{subkey}")
        if name == "video_reward":
            backend = str(sub.get("backend"))
            if backend != "stub":
                raise ValueError(
                    "reward.kwargs.video_reward.backend only supports 'stub' until the "
                    "Ray reward inference runtime is implemented",
                )
            removed_endpoint_fields = [
                key
                for key in ("enqueue_url", "fetch_url", "token", "poll_interval_s", "max_wait_s")
                if key in sub
            ]
            if removed_endpoint_fields:
                fields = ", ".join(sorted(removed_endpoint_fields))
                raise ValueError(
                    "reward.kwargs.video_reward no longer supports external reward "
                    f"endpoint fields: {fields}",
                )


def validate_training_config(cfg: DictConfig) -> None:
    """Validate only unresolved mandatory values and cross-field contracts."""

    assert_no_missing(cfg)
    if "algorithm" not in cfg:
        raise ValueError("config missing required field: algorithm")
    kind = resolve_algorithm_kind(cfg.algorithm)

    if "reward" in cfg:
        validate_reward_config(cfg)

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
    "validate_reward_config",
    "validate_training_config",
]
