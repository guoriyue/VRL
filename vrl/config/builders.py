"""Build typed runtime config objects from merged YAML."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any, get_type_hints

from omegaconf import DictConfig, OmegaConf

from vrl.config.precision import resolve_precision_policy
from vrl.config.validation import (
    path_exists,
    require,
    resolve_algorithm_kind,
    validate_reward_config,
    validate_training_config,
)


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} must be a dataclass type")
    return {field.name for field in fields(cls) if field.init}


def _required_field_names(cls: type[Any]) -> set[str]:
    """Fields without a default — required, torch signature semantics."""

    return {
        field.name
        for field in fields(cls)
        if field.init and field.default is MISSING and field.default_factory is MISSING
    }


def _section_payload(cfg: DictConfig, path: str) -> dict[str, Any]:
    """Resolve a YAML section to a plain dict; absent section -> {}.

    An explicitly null section (``actor.ema: null``) raises instead of
    silently replacing the section with all-defaults — only true absence
    means "use the dataclass defaults".
    """

    node = OmegaConf.select(cfg, path)
    if node is None:
        if path_exists(cfg, path):
            raise ValueError(
                f"config section {path} is null; delete the key or fill the section",
            )
        return {}
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise ValueError(f"config section {path} must be a mapping")
    return raw


def _section_payload_and_missing(
    cls: type[Any],
    cfg: DictConfig,
    path: str,
) -> tuple[dict[str, Any], list[str]]:
    """Select ``cls`` fields from the section; report missing required paths.

    Unknown keys raise: for a typed section the dataclass is the complete
    vocabulary, so a typo'd hyperparameter must refuse to start rather than
    silently train with the default behind a lint warning.
    """

    raw = _section_payload(cfg, path)
    allowed = _dataclass_field_names(cls)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ValueError(f"unknown {cls.__name__} key(s): {keys}")
    payload = {key: value for key, value in raw.items() if key in allowed}
    missing = sorted(
        f"{path}.{name}" for name in _required_field_names(cls) - set(payload)
    )
    return payload, missing


def section_to_dataclass(cls: type[Any], cfg: DictConfig, path: str) -> Any:
    """Construct ``cls`` from the YAML section at ``path``.

    Requiredness comes from the dataclass field list alone: a field without a
    default is required, and every missing required key is reported at once
    with its full YAML path.
    """

    payload, missing = _section_payload_and_missing(cls, cfg, path)
    if missing:
        raise ValueError("config missing required key(s): " + ", ".join(missing))
    return cls(**payload)


def _dataclass_payload(cls: type[Any], node: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    allowed = _dataclass_field_names(cls)
    ignored_keys = {"kind", "kl_reward"}
    unknown = sorted(set(raw) - allowed - ignored_keys)
    if unknown:
        fields_text = ", ".join(f"algorithm.{key}" for key in unknown)
        raise ValueError(f"unknown {cls.__name__} config field(s): {fields_text}")
    return {key: value for key, value in raw.items() if key in allowed}


def _validate_yaml_home(field_name: str, home: str) -> None:
    """Reject metadata addresses whose top-level section is not a known one.

    Guards the silent failure mode of a typo'd address on an OPTIONAL field
    (it would fall back to the default instead of reading the user's value).
    The valid-section vocabulary is derived from the schema's RootConfig —
    the existing source of truth for top-level sections — so this check can
    never drift into rejecting a legitimately added section.
    """

    from vrl.config.schema import RootConfig

    top = home.split(".", 1)[0]
    if top not in RootConfig.model_fields:
        expected = ", ".join(sorted(RootConfig.model_fields))
        raise AssertionError(
            f"TrainerConfig.{field_name} declares unknown yaml home {home!r}; "
            f"the top-level section must be one of: {expected}",
        )


def build_trainer_config(cfg: DictConfig):
    """Slice merged YAML into ``TrainerConfig``.

    The layout is derived from each field's ``metadata={"yaml": ...}`` on the
    dataclass (section name for scalars, dotted path for nested sections,
    "bridged" for builder-computed values), and requiredness from the field
    defaults — the dataclass is the single declaration of both. Missing
    required keys across sections and scalars are
    collected and reported together with full YAML paths.
    """

    from vrl.trainers.core.types import TrainerConfig

    hints = get_type_hints(TrainerConfig)
    payload: dict[str, Any] = {}
    missing: list[str] = []

    for f in fields(TrainerConfig):
        if not f.init:
            continue
        home = f.metadata.get("yaml")
        if home is None:
            raise AssertionError(
                f"TrainerConfig.{f.name} does not declare its YAML home "
                "(field metadata {'yaml': ...})",
            )
        if home == "bridged":
            continue
        _validate_yaml_home(f.name, home)
        field_cls = hints[f.name]
        if is_dataclass(field_cls):
            section_payload, section_missing = _section_payload_and_missing(
                field_cls, cfg, home,
            )
            if section_missing:
                missing.extend(section_missing)
            else:
                payload[f.name] = field_cls(**section_payload)
        else:
            path = f"{home}.{f.name}"
            if path_exists(cfg, path):
                payload[f.name] = require(cfg, path)
            elif f.default is MISSING and f.default_factory is MISSING:
                missing.append(path)

    if missing:
        raise ValueError("config missing required key(s): " + ", ".join(sorted(missing)))

    # The unified precision policy expands into four fields.
    precision = resolve_precision_policy(cfg)
    payload["mixed_precision"] = precision.compute
    payload["bf16"] = precision.compute == "bf16"
    payload["rollout_precision"] = precision.rollout
    payload["math_precision"] = precision.math

    return TrainerConfig(**payload)


def build_algorithm_config(cfg: DictConfig):
    """Dispatch on ``algorithm.kind`` and return the typed algorithm config."""

    if "algorithm" not in cfg:
        raise ValueError("config missing `algorithm` section")
    kind = resolve_algorithm_kind(cfg.algorithm)

    if kind == "grpo":
        from vrl.algorithms.grpo.continuous import GRPOConfig

        return GRPOConfig(**_dataclass_payload(GRPOConfig, cfg.algorithm))

    if kind == "token_grpo":
        from vrl.algorithms.grpo.token import TokenGRPOConfig

        return TokenGRPOConfig(**_dataclass_payload(TokenGRPOConfig, cfg.algorithm))

    if kind == "token_grpo_multisegment":
        from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig

        return MultiSegmentTokenGRPOConfig(
            **_dataclass_payload(MultiSegmentTokenGRPOConfig, cfg.algorithm),
        )

    if kind == "diffusion_dpo":
        from vrl.algorithms.dpo import DiffusionDPOConfig

        return DiffusionDPOConfig(**_dataclass_payload(DiffusionDPOConfig, cfg.algorithm))

    if kind == "diffusion_nft":
        from vrl.algorithms.diffusion_nft import DiffusionNFTConfig

        return DiffusionNFTConfig(**_dataclass_payload(DiffusionNFTConfig, cfg.algorithm))

    raise AssertionError(f"unreachable: kind={kind}")  # pragma: no cover


def build_reward_config(cfg: DictConfig) -> tuple[dict[str, float], dict[str, dict]]:
    """Slice ``cfg.reward`` into ``(weights, kwargs)``."""

    validate_reward_config(cfg)
    reward = cfg.reward

    components = OmegaConf.to_container(
        reward.components,
        resolve=True,
        throw_on_missing=True,
    ) or {}
    weights: dict[str, float] = {}
    for name, weight in components.items():
        component_weight = float(weight)
        if component_weight > 0:
            weights[name] = component_weight

    raw_kwargs = reward.get("kwargs", None)
    kwargs: dict[str, dict] = (
        OmegaConf.to_container(raw_kwargs, resolve=True, throw_on_missing=True) or {}
        if raw_kwargs
        else {}
    )

    return weights, kwargs


def build_configs(cfg: DictConfig) -> dict[str, Any]:
    """Bundle typed configs for downstream training scripts."""

    validate_training_config(cfg)
    out: dict[str, Any] = {
        "trainer": build_trainer_config(cfg),
        "algorithm": build_algorithm_config(cfg),
        "raw": cfg,
    }
    if "reward" in cfg:
        out["reward"] = build_reward_config(cfg)
    return out


__all__ = [
    "build_algorithm_config",
    "build_configs",
    "build_reward_config",
    "build_trainer_config",
]
