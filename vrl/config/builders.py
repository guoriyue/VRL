"""Build typed runtime config objects from merged YAML."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

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


def _dataclass_payload(cls: type[Any], node: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    allowed = _dataclass_field_names(cls)
    return {key: value for key, value in raw.items() if key in allowed}


def build_trainer_config(cfg: DictConfig):
    """Slice merged YAML into ``TrainerConfig``."""

    from vrl.trainers.core.types import (
        DebugConfig,
        EMAConfig,
        OptimConfig,
        TorchProfilerConfig,
        TrainerConfig,
    )

    optim_dict = require(cfg, "actor.optim")
    ema_dict = require(cfg, "actor.ema")
    debug_dict = require(cfg, "trainer.debug")
    torch_profiler_dict = require(cfg, "trainer.torch_profiler")

    if path_exists(cfg, "rollout.n_samples_per_prompt"):
        n_value = require(cfg, "rollout.n_samples_per_prompt")
    elif path_exists(cfg, "rollout.n"):
        n_value = require(cfg, "rollout.n")
    else:
        raise ValueError(
            "config missing rollout group size: expected rollout.n or "
            "rollout.n_samples_per_prompt",
        )

    return TrainerConfig(
        optim=OptimConfig(**optim_dict),
        ema=EMAConfig(**ema_dict),
        debug=DebugConfig(**debug_dict),
        torch_profiler=TorchProfilerConfig(**torch_profiler_dict),
        max_norm=require(cfg, "actor.max_norm"),
        ppo_epochs=require(cfg, "actor.ppo_epochs"),
        gradient_accumulation_steps=require(cfg, "actor.gradient_accumulation_steps"),
        mixed_precision=require(cfg, "actor.mixed_precision"),
        bf16=require(cfg, "actor.bf16"),
        gradient_checkpointing=require(cfg, "actor.gradient_checkpointing"),
        n=n_value,
        rollout_batch_size=require(cfg, "rollout.rollout_batch_size"),
        timestep_fraction=require(cfg, "actor.timestep_fraction"),
        total_epochs=require(cfg, "trainer.total_epochs"),
        save_freq=require(cfg, "trainer.save_freq"),
        log_freq=require(cfg, "trainer.log_freq"),
        output_dir=require(cfg, "trainer.output_dir"),
        seed=require(cfg, "trainer.seed"),
        resume_from=require(cfg, "trainer.resume_from"),
        resume_strict=require(cfg, "trainer.resume_strict"),
        profile=require(cfg, "trainer.profile"),
    )


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
    weights = {name: float(weight) for name, weight in components.items() if float(weight) > 0}

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
