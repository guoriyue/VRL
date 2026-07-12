"""Build typed runtime config objects from merged YAML."""

from __future__ import annotations

import math
from dataclasses import MISSING, fields, is_dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from omegaconf import DictConfig, OmegaConf

from vrl.config.algorithm import algorithm_config_class
from vrl.config.precision import (
    PrecisionPolicy,
    resolve_precision_policy,
)
from vrl.config.validation import (
    path_exists,
    require,
    resolve_algorithm_kind,
    validate_reward_config,
    validate_training_config,
)

if TYPE_CHECKING:
    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig


def build_precision_split_safety_configs() -> tuple[
    PrecisionCorrectionConfig,
    PrecisionDriftGuardConfig,
]:
    """Build the production correction and guard policy for a precision split.

    Hardware validation probes consume this same typed source so a measured gate
    cannot silently validate thresholds different from live training.
    """

    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig

    return (
        PrecisionCorrectionConfig(
            tis_mode="truncate",
            rs_mode="seq_mean_k1",
        ),
        PrecisionDriftGuardConfig(
            mode="fail",
            max_abs_log_ratio=math.log(10.0),
            max_ratio_abs_dev=9.0,
            fail_on_nonfinite=True,
        ),
    )


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} must be a dataclass type")
    return {field.name for field in fields(cls) if field.init}


def _section_payload_and_missing(
    cls: type[Any],
    cfg: DictConfig,
    path: str,
) -> tuple[dict[str, Any], list[str]]:
    """Select ``cls`` fields from the section; report missing required paths.

    An explicitly null section (``actor.ema: null``) raises instead of
    silently replacing the section with all-defaults — only true absence
    means "use the dataclass defaults". Unknown keys raise: for a typed
    section the dataclass is the complete vocabulary, so a typo'd
    hyperparameter must refuse to start rather than silently train with the
    default behind a lint warning.
    """

    node = OmegaConf.select(cfg, path)
    if node is None:
        if path_exists(cfg, path):
            raise ValueError(
                f"config section {path} is null; delete the key or fill the section",
            )
        raw: dict[str, Any] = {}
    else:
        raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
        if not isinstance(raw, dict):
            raise ValueError(f"config section {path} must be a mapping")

    allowed = _dataclass_field_names(cls)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ValueError(f"unknown {cls.__name__} key(s): {keys}")
    payload = {key: value for key, value in raw.items() if key in allowed}
    # Required = no default, torch signature semantics.
    required = {
        field.name
        for field in fields(cls)
        if field.init and field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(f"{path}.{name}" for name in required - set(payload))
    return payload, missing


def _dataclass_payload(cls: type[Any], node: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    allowed = _dataclass_field_names(cls)
    ignored_keys = {"kind", "kl_reward_coef"}
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


def build_trainer_config(
    cfg: DictConfig,
    *,
    precision: PrecisionPolicy | None = None,
):
    """Slice merged YAML into ``TrainerConfig``.

    The layout is derived from each field's ``metadata={"yaml": ...}`` on the
    dataclass (section name for scalars, dotted path for nested sections,
    "bridged" for builder-computed values), and requiredness from the field
    defaults — the dataclass is the single declaration of both. Missing
    required keys across sections and scalars are
    collected and reported together with full YAML paths.
    """

    from vrl.trainers.core.types import TrainerConfig

    trainer_block = getattr(cfg, "trainer", None)
    if trainer_block is not None and "eval" in trainer_block:
        raise ValueError(
            "trainer.eval was removed: online training no longer runs fixed eval. "
            "Evaluate saved checkpoints with a script under vrl/scripts/eval; "
            "for Cosmos Predict2.5 + Kling use "
            "`python -m vrl.scripts.eval.cosmos_predict25_kling_eval`.",
        )
    orchestration_block = (
        getattr(trainer_block, "rollout_orchestration", None)
        if trainer_block is not None
        else None
    )
    if orchestration_block is not None and "require_separate_gpus" in orchestration_block:
        raise ValueError(
            "trainer.rollout_orchestration.require_separate_gpus was removed: "
            "rollout topology is derived from resolved GPU ownership. Shared GPUs "
            "use strict_on_policy with distributed.resources.rollout.gpu_pool=trainer; "
            "continuous requires disjoint trainer and rollout GPUs.",
        )

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
                field_cls,
                cfg,
                home,
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

    # Resolve the public policy once; trainer fields are its runtime projection.
    precision = precision or resolve_precision_policy(cfg)
    payload.update(
        train_precision=precision.training.label,
        rollout_precision=precision.rollout.label,
    )
    # On a rollout/train precision split, the correction mechanism is an
    # implementation detail the user should not have to spell out: default to
    # TIS/RS correction plus a catastrophic-drift guard. Explicit expert
    # trainer.precision_* blocks are still respected.
    if not precision.stages_match:
        correction, guard = build_precision_split_safety_configs()
        if not path_exists(cfg, "trainer.precision_correction"):
            payload["precision_correction"] = correction
        if not path_exists(cfg, "trainer.precision_drift_guard"):
            payload["precision_drift_guard"] = guard

    return TrainerConfig(**payload)


def build_algorithm_config(cfg: DictConfig):
    """Dispatch on ``algorithm.kind`` and return the typed algorithm config."""

    if "algorithm" not in cfg:
        raise ValueError("config missing `algorithm` section")
    kind = resolve_algorithm_kind(cfg.algorithm)
    cls = algorithm_config_class(kind)
    return cls(**_dataclass_payload(cls, cfg.algorithm))


def build_reward_config(cfg: DictConfig) -> tuple[dict[str, float], dict[str, dict]]:
    """Slice ``cfg.reward`` into ``(weights, kwargs)``.

    Zero-weight components remain present so they can be scored and logged as
    observation-only safeguards without changing the optimization reward.
    """

    validate_reward_config(cfg)
    reward = cfg.reward

    components = (
        OmegaConf.to_container(
            reward.components,
            resolve=True,
            throw_on_missing=True,
        )
        or {}
    )
    weights = {name: float(weight) for name, weight in components.items()}

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
    precision = resolve_precision_policy(cfg)
    out: dict[str, Any] = {
        "trainer": build_trainer_config(cfg, precision=precision),
        "algorithm": build_algorithm_config(cfg),
        "precision": precision,
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
