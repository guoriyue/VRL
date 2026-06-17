"""Factory functions for common online training recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import build_configs
from vrl.config.precision import resolve_precision_policy
from vrl.models.dtypes import resolve_torch_dtype
from vrl.ray.resources import (
    resolve_distributed_resources,
    reward_runtime_resource_kwargs,
)
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import (
    build_rollout_config_from_cfg as build_collector_rollout_config_from_cfg,
)
from vrl.rollouts.families import (
    RolloutFamilyEntry,
    get_rollout_family_entry,
    normalize_rollout_family,
)
from vrl.utils.config import cfg_get


class UnsupportedOnlineRecipeError(ValueError):
    """Raised when a YAML config targets a non-online or unsupported recipe."""


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluatorPair:
    """Algorithm instance plus its optional evaluator."""

    algorithm: Any
    evaluator: Any | None


@dataclass(frozen=True, slots=True)
class OnlineRecipeFactoryOutput:
    """Typed objects built from YAML and the canonical rollout family registry."""

    built: dict[str, Any]
    trainer_config: Any
    family: str
    family_entry: RolloutFamilyEntry
    collector_config: Any
    reward_fn: Any
    algorithm: Any
    evaluator: Any | None


def resolve_online_family(cfg: DictConfig) -> str:
    """Resolve a merged training config to the canonical online rollout family."""

    raw_family = OmegaConf.select(cfg, "model.family", default=None)
    if raw_family is None:
        raise ValueError("config missing required field: model.family")
    family = normalize_rollout_family(str(raw_family))
    algorithm_kind = str(OmegaConf.select(cfg, "algorithm.kind", default=""))
    if family == "janus_pro" and algorithm_kind == "token_grpo_multisegment":
        return "janus_pro_r1"
    return family


def build_rollout_config_from_cfg(
    cfg: DictConfig,
    family: str | RolloutFamilyEntry | None = None,
) -> Any:
    """Build resolved rollout config from YAML."""

    entry = _entry_from_family(cfg, family)
    return _rollout_config_for_entry(cfg, entry)


def _rollout_config_for_entry(
    cfg: DictConfig,
    entry: RolloutFamilyEntry,
) -> Any:
    return build_collector_rollout_config_from_cfg(
        cfg,
        family=entry.family,
    )


def build_reward_from_cfg(
    cfg: DictConfig,
    *,
    built: dict[str, Any] | None = None,
    device: str = "cuda",
    reward_placement: Any | None = None,
) -> Any:
    """Build the online reward function from the shared config loader output.

    ``reward_placement`` is the reward role's placement handle from a
    ``GlobalRayPlacementOwner``; when given, the Ray reward runtime schedules its
    actors into the shared run-level placement group instead of building its own.
    """

    built = built or build_configs(cfg)
    if "reward" not in built:
        raise UnsupportedOnlineRecipeError(
            "online recipe requires a reward section; diffusion_dpo is offline-only",
        )
    reward_weights, reward_kwargs = built["reward"]
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    reward_kwargs = _with_resolved_reward_runtime_kwargs(
        cfg,
        reward_weights,
        reward_kwargs,
        reward_placement=reward_placement,
    )
    from vrl.rewards.functions.registry import MultiReward

    return MultiReward.from_dict(
        reward_weights,
        device=str(device),
        reward_kwargs=reward_kwargs,
    )


def _with_resolved_reward_runtime_kwargs(
    cfg: DictConfig,
    reward_weights: dict[str, float],
    reward_kwargs: dict[str, dict],
    *,
    reward_placement: Any | None = None,
) -> dict[str, dict]:
    ray_reward_keys = [
        name
        for name, weight in reward_weights.items()
        if float(weight) > 0.0
        and str((reward_kwargs.get(name) or {}).get("execution", "")) == "pool"
    ]
    if not ray_reward_keys:
        return reward_kwargs

    resources = resolve_distributed_resources(cfg)
    resolved_runtime = reward_runtime_resource_kwargs(resources)

    out = dict(reward_kwargs)
    for reward_key in ray_reward_keys:
        merged = dict(reward_kwargs.get(reward_key) or {})
        merged.update(resolved_runtime)
        if reward_placement is not None:
            # Run-level placement owner supplied: the reward runtime schedules its
            # actors into the shared group's reward bundle.
            merged["placement"] = reward_placement
        out[reward_key] = merged
    return out


def build_algorithm_and_evaluator_from_cfg(
    cfg: DictConfig,
    *,
    family: str | RolloutFamilyEntry | None = None,
    built: dict[str, Any] | None = None,
    collector_config: Any | None = None,
    scheduler: Any | None = None,
) -> AlgorithmEvaluatorPair:
    """Build the algorithm/evaluator pair for a strict online recipe."""

    built = built or build_configs(cfg)
    entry = _entry_from_family(cfg, family)
    algorithm_config = built["algorithm"]
    kind = str(OmegaConf.select(cfg, "algorithm.kind", default=""))

    if kind == "grpo":
        from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
        from vrl.algorithms.grpo.token import TokenGRPOConfig
        from vrl.rollouts.evaluators.diffusion.sde_logprob import (
            DiffusionSDELogProbEvaluator,
        )

        if not isinstance(algorithm_config, GRPOConfig) or isinstance(
            algorithm_config,
            TokenGRPOConfig,
        ):
            raise TypeError(
                f"{entry.family} GRPO expects GRPOConfig, got "
                f"{type(algorithm_config).__name__}",
            )
        collector_config = collector_config or _rollout_config_for_entry(cfg, entry)
        math_dtype = resolve_torch_dtype(resolve_precision_policy(cfg).math)
        return AlgorithmEvaluatorPair(
            algorithm=GRPO(algorithm_config),
            evaluator=DiffusionSDELogProbEvaluator(
                scheduler,
                noise_level=float(
                    cfg_get(collector_config, "noise_level", 1.0),
                ),
                sde_type=str(
                    cfg_get(collector_config, "sde_type", "sde"),
                ),
                math_dtype=math_dtype,
            ),
        )

    if kind == "token_grpo":
        from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig

        if not isinstance(algorithm_config, TokenGRPOConfig):
            raise TypeError(
                f"{entry.family} token GRPO expects TokenGRPOConfig, got "
                f"{type(algorithm_config).__name__}",
            )
        if entry.collector.kind == "ar_continuous":
            from vrl.rollouts.evaluators.ar import ContinuousTokenLogProbEvaluator

            evaluator = ContinuousTokenLogProbEvaluator()
        else:
            from vrl.rollouts.evaluators.ar import TokenLogProbEvaluator

            evaluator = TokenLogProbEvaluator()
        return AlgorithmEvaluatorPair(algorithm=TokenGRPO(algorithm_config), evaluator=evaluator)

    if kind == "token_grpo_multisegment":
        from vrl.algorithms.grpo.multisegment import (
            MultiSegmentTokenGRPO,
            MultiSegmentTokenGRPOConfig,
        )
        from vrl.rollouts.evaluators.ar import MultiSegmentTokenLogProbEvaluator

        if entry.family != "janus_pro_r1":
            raise UnsupportedOnlineRecipeError(
                "token_grpo_multisegment currently requires rollout family janus_pro_r1",
            )
        if not isinstance(algorithm_config, MultiSegmentTokenGRPOConfig):
            raise TypeError(
                "multi-segment token GRPO expects MultiSegmentTokenGRPOConfig, "
                f"got {type(algorithm_config).__name__}",
            )
        segment_flags = dict(_cfg_select(cfg, "algorithm.train_segments", {}) or {})
        enabled_segments = tuple(
            name for name, enabled in segment_flags.items() if bool(enabled)
        )
        return AlgorithmEvaluatorPair(
            algorithm=MultiSegmentTokenGRPO(algorithm_config),
            evaluator=MultiSegmentTokenLogProbEvaluator(enabled_segments=enabled_segments),
        )

    if kind == "diffusion_nft":
        from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
        from vrl.algorithms.grpo.token import TokenGRPOConfig

        if not isinstance(algorithm_config, DiffusionNFTConfig) or isinstance(
            algorithm_config,
            TokenGRPOConfig,
        ):
            raise TypeError(
                f"{entry.family} DiffusionNFT expects DiffusionNFTConfig, got "
                f"{type(algorithm_config).__name__}",
            )
        return AlgorithmEvaluatorPair(
            algorithm=DiffusionNFT(algorithm_config),
            evaluator=None,
        )

    if kind == "diffusion_dpo":
        raise UnsupportedOnlineRecipeError(
            "diffusion_dpo is an offline recipe and is not supported by common online recipe",
        )

    raise UnsupportedOnlineRecipeError(f"unsupported online algorithm.kind: {kind!r}")


def build_collector_from_cfg(
    cfg: DictConfig,
    *,
    model: Any,
    reward_fn: Any,
    family: str | RolloutFamilyEntry | None = None,
    collector_config: Any | None = None,
    runtime: Any | None = None,
) -> Any:
    """Build a rollout collector through the canonical family registry."""

    entry = _entry_from_family(cfg, family)
    collector_config = collector_config or _rollout_config_for_entry(cfg, entry)
    # Topology-derived release policy so the collector reads its own handoff
    # rather than asking the runtime. Absent for in-process runs with no
    # distributed.resources, where there is no shared GPU to hand off.
    lifecycle = None
    if OmegaConf.select(cfg, "distributed.resources", default=None) is not None:
        lifecycle = resolve_distributed_resources(cfg).lifecycle
    return build_rollout_collector(
        entry.family,
        model=model,
        reward_fn=reward_fn,
        config=collector_config,
        runtime=runtime,
        lifecycle=lifecycle,
    )


def build_online_recipe_components(
    cfg: DictConfig,
    *,
    device: str = "cuda",
    family: str | RolloutFamilyEntry | None = None,
    scheduler: Any | None = None,
    built: dict[str, Any] | None = None,
    reward_placement: Any | None = None,
) -> OnlineRecipeFactoryOutput:
    """Build config-derived online recipe components without loading a model.

    ``reward_placement`` is forwarded to the reward runtime so Ray reward actors
    share the run-level placement group owned by a ``GlobalRayPlacementOwner``.
    """

    built = built or build_configs(cfg)
    entry = _entry_from_family(cfg, family)
    collector_config = _rollout_config_for_entry(cfg, entry)
    reward_fn = build_reward_from_cfg(
        cfg,
        built=built,
        device=device,
        reward_placement=reward_placement,
    )
    pair = build_algorithm_and_evaluator_from_cfg(
        cfg,
        family=entry,
        built=built,
        collector_config=collector_config,
        scheduler=scheduler,
    )
    return OnlineRecipeFactoryOutput(
        built=built,
        trainer_config=built["trainer"],
        family=entry.family,
        family_entry=entry,
        collector_config=collector_config,
        reward_fn=reward_fn,
        algorithm=pair.algorithm,
        evaluator=pair.evaluator,
    )


def _entry_from_family(
    cfg: DictConfig,
    family: str | RolloutFamilyEntry | None,
) -> RolloutFamilyEntry:
    if isinstance(family, RolloutFamilyEntry):
        return family
    return get_rollout_family_entry(family or resolve_online_family(cfg))


def _cfg_select(cfg: DictConfig, path: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    if isinstance(value, (dict, list, tuple)):
        return value
    if value is default:
        return default
    container = value
    if hasattr(value, "_metadata"):
        container = OmegaConf.to_container(value, resolve=True)
    return container


__all__ = [
    "AlgorithmEvaluatorPair",
    "OnlineRecipeFactoryOutput",
    "UnsupportedOnlineRecipeError",
    "build_algorithm_and_evaluator_from_cfg",
    "build_collector_from_cfg",
    "build_online_recipe_components",
    "build_reward_from_cfg",
    "build_rollout_config_from_cfg",
    "resolve_online_family",
]
