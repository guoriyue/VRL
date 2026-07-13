"""Factory functions for common online training recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import build_configs
from vrl.config.precision import resolve_precision_policy
from vrl.models.dtypes import resolve_torch_dtype
from vrl.ray.resources import resolve_distributed_resources, reward_torch_device
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import build_rollout_config_from_cfg
from vrl.rollouts.families import (
    RolloutFamilyEntry,
    get_rollout_family_entry,
    resolve_rollout_family_from_config,
)
from vrl.utils.config import cfg_get


class UnsupportedOnlineRecipeError(ValueError):
    """Raised when a YAML config targets a non-online or unsupported recipe."""


def _validate_topology_derived_reward_kwargs(
    reward_kwargs: dict[str, Any],
) -> None:
    """Reject public knobs whose values come from resolved GPU ownership."""

    for name, kwargs in reward_kwargs.items():
        extra = dict(kwargs or {})
        for key in ("sleep_offload", "memory_parking_residual_bytes_limit"):
            if key in extra:
                raise ValueError(
                    f"reward.kwargs.{name}.{key} is topology-derived and cannot "
                    "be set in YAML; remove it and select shared or dedicated "
                    "reward GPU ownership under distributed.resources.reward",
                )


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
    """Public recipe facade over the registry-owned family selection rule."""

    return resolve_rollout_family_from_config(cfg)


def build_reward_from_cfg(
    cfg: DictConfig,
    *,
    built: dict[str, Any] | None = None,
    device: str = "cuda",
) -> Any:
    """Build the online reward function from the shared config loader output.

    In-process GPU ownership decides parking: a shared reward is automatically
    pooled and must publish a release proof, while a dedicated reward stays
    resident. HTTP components own their deployment externally and receive no
    local parking policy. YAML selects transport, not lifecycle behavior.
    """

    built = built or build_configs(cfg)
    if "reward" not in built:
        raise UnsupportedOnlineRecipeError(
            "online recipe requires a reward section; diffusion_dpo is offline-only",
        )
    reward_weights, reward_kwargs = built["reward"]
    if not any(weight > 0 for weight in reward_weights.values()):
        raise ValueError("At least one reward component must have weight > 0.")
    _validate_topology_derived_reward_kwargs(dict(reward_kwargs))
    from vrl.rewards.functions.registry import MultiReward

    memory_parking_required: bool | None = None
    if OmegaConf.select(cfg, "distributed.resources", default=None) is not None:
        resources = resolve_distributed_resources(cfg)
        expected_device = reward_torch_device(resources)
        if str(device).strip().lower() != expected_device.strip().lower():
            raise ValueError(
                f"reward device {str(device)!r} conflicts with distributed resources "
                f"resolved device {expected_device!r}; resource topology is the "
                "execution-device source of truth.",
            )
        validate_reward_memory_parking_from_cfg(
            cfg,
            resources=resources,
            built=built,
            device=str(device),
        )
        memory_parking_required = bool(
            resources.lifecycle.handoff.release_reward_after_score,
        )

    return MultiReward.from_dict(
        reward_weights,
        device=str(device),
        reward_kwargs=reward_kwargs,
        memory_parking_required=memory_parking_required,
    )


def validate_reward_memory_parking_from_cfg(
    cfg: DictConfig,
    *,
    resources: Any,
    built: dict[str, Any] | None = None,
    device: str | None = None,
) -> None:
    """Validate shared reward parking without constructing a reward model."""

    if built is not None and "reward" in built:
        reward_kwargs = built["reward"][1]
        names = tuple(str(name) for name in built["reward"][0])
    else:
        raw_kwargs = OmegaConf.select(cfg, "reward.kwargs", default={})
        reward_kwargs = (
            OmegaConf.to_container(raw_kwargs, resolve=True)
            if OmegaConf.is_config(raw_kwargs)
            else raw_kwargs
        )
        components = OmegaConf.select(cfg, "reward.components", default={})
        plain = (
            OmegaConf.to_container(components, resolve=True)
            if OmegaConf.is_config(components)
            else components
        )
        names = tuple(str(name) for name in dict(plain or {}))
    reward_kwargs = dict(reward_kwargs or {})
    _validate_topology_derived_reward_kwargs(reward_kwargs)
    if not bool(resources.lifecycle.handoff.release_reward_after_score):
        return
    if not names:
        return
    from vrl.rewards.functions.registry import (
        validate_reward_memory_parking_components,
    )

    validate_reward_memory_parking_components(
        names,
        device=str(device or reward_torch_device(resources)),
        reward_kwargs=reward_kwargs,
    )


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
    diffusion_logprob_kinds = {"grpo", "dance_grpo", "flow_dppo", "grpo_guard"}
    # Production callers pass the complete build result. Unit-level callers may
    # inject only an algorithm config to exercise type rejection; resolve the
    # same public source directly in that narrow path.
    precision = built.get("precision") or resolve_precision_policy(cfg)
    if precision.diffusion_math != "fp32" and kind not in diffusion_logprob_kinds:
        raise ValueError(
            "precision.diffusion_math.dtype overrides are supported only by "
            "diffusion log-prob "
            f"objectives; algorithm.kind={kind!r} keeps its protected math in fp32",
        )

    if kind in diffusion_logprob_kinds:
        # All four are flow-matching GRPO-family algorithms on the same SDE
        # evaluator. dance_grpo reuses FlowGRPO unchanged (its delta is the
        # trainer's random timestep selection + multi-reward); flow_dppo /
        # grpo_guard are trust-region variants whose loss reads the rollout
        # proposal mean (sampling.return_prev_sample_mean).
        from vrl.algorithms.grpo.continuous import (
            GRPO,
            FlowDPPO,
            FlowDPPOConfig,
            GRPOConfig,
            GRPOGuard,
            GRPOGuardConfig,
        )
        from vrl.rollouts.evaluators.diffusion.sde_logprob import (
            DiffusionSDELogProbEvaluator,
        )

        expected_config_type = {
            "grpo": GRPOConfig,
            "dance_grpo": GRPOConfig,
            "flow_dppo": FlowDPPOConfig,
            "grpo_guard": GRPOGuardConfig,
        }[kind]
        if not isinstance(algorithm_config, expected_config_type):
            raise TypeError(
                f"{entry.family} {kind} expects {expected_config_type.__name__}, got "
                f"{type(algorithm_config).__name__}",
            )
        if kind == "flow_dppo":
            algorithm: object = FlowDPPO(algorithm_config)
        elif kind == "grpo_guard":
            algorithm = GRPOGuard(algorithm_config)
        else:
            algorithm = GRPO(algorithm_config)
        collector_config = collector_config or build_rollout_config_from_cfg(
            cfg, family=entry.family
        )
        math_dtype = resolve_torch_dtype(precision.diffusion_math)
        return AlgorithmEvaluatorPair(
            algorithm=algorithm,
            evaluator=DiffusionSDELogProbEvaluator(
                scheduler,
                noise_level=float(
                    cfg_get(collector_config, "noise_level", 1.0),
                ),
                sde_type=str(
                    cfg_get(collector_config, "sde_type", "flow_grpo"),
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
        enabled_segments = tuple(name for name, enabled in segment_flags.items() if bool(enabled))
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
    reward_fn: Any,
    family: str | RolloutFamilyEntry | None = None,
    collector_config: Any | None = None,
    runtime: Any | None = None,
) -> Any:
    """Build a rollout collector through the canonical family registry."""

    entry = _entry_from_family(cfg, family)
    collector_config = collector_config or build_rollout_config_from_cfg(cfg, family=entry.family)
    # Topology-derived release policy so the collector reads its own handoff
    # rather than asking the runtime. Absent for in-process runs with no
    # distributed.resources, where there is no shared GPU to hand off.
    lifecycle = None
    if OmegaConf.select(cfg, "distributed.resources", default=None) is not None:
        lifecycle = resolve_distributed_resources(cfg).lifecycle
    return build_rollout_collector(
        entry.family,
        reward_fn=reward_fn,
        config=collector_config,
        runtime=runtime,
        lifecycle=lifecycle,
    )


def build_online_recipe_components(
    cfg: DictConfig,
    *,
    reward_device: str = "cuda",
    family: str | RolloutFamilyEntry | None = None,
    scheduler: Any | None = None,
    built: dict[str, Any] | None = None,
) -> OnlineRecipeFactoryOutput:
    """Build config-derived online recipe components without loading a model."""

    built = built or build_configs(cfg)
    entry = _entry_from_family(cfg, family)
    collector_config = build_rollout_config_from_cfg(cfg, family=entry.family)
    reward_fn = build_reward_from_cfg(
        cfg,
        built=built,
        device=reward_device,
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
    "resolve_online_family",
]
