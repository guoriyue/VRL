"""Factory functions for common online training recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.config.builders import BuiltConfigs
from vrl.families.registry import ModelFamilyEntry
from vrl.models.dtypes import resolve_torch_dtype
from vrl.ray.resources import ResolvedDistributedResources, require_reward_device


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluatorPair:
    """Algorithm instance plus its optional evaluator."""

    algorithm: Algorithm
    evaluator: Any | None


def build_reward(
    *,
    built: BuiltConfigs,
    resources: ResolvedDistributedResources | None,
    device: str = "cuda",
) -> Any:
    """Build the online reward function from the shared config loader output.

    In-process GPU ownership decides parking: a shared reward is automatically
    pooled and must publish a release proof, while a dedicated reward stays
    resident. HTTP components own their deployment externally and receive no
    local parking policy. YAML selects transport, not lifecycle behavior.
    """

    reward = built.reward
    if reward is None:
        raise ValueError(
            "online recipe requires a reward section; diffusion_dpo is offline-only",
        )
    if not any(weight > 0 for weight in reward.weights.values()):
        raise ValueError("At least one reward component must have weight > 0.")
    from vrl.rewards.functions.registry import MultiReward

    memory_parking_required: bool | None = None
    if resources is not None and not reward.all_external_inference:
        require_reward_device(resources, str(device))
        validate_reward_memory_parking(
            resources=resources,
            built=built,
            device=str(device),
        )
        memory_parking_required = bool(
            resources.lifecycle.handoff.release_reward_after_score,
        )
    elif resources is not None:
        # HTTP components execute as CPU clients in this process; their model
        # device belongs to the standalone service and is absent from the local
        # resource plan. A torchrun rank's logical CUDA name is irrelevant here.
        memory_parking_required = False

    return MultiReward.from_dict(
        reward.weights,
        device=str(device),
        reward_kwargs=reward.kwargs,
        memory_parking_required=memory_parking_required,
        inference_configs=reward.inference_configs,
    )


def validate_reward_memory_parking(
    *,
    resources: ResolvedDistributedResources,
    built: BuiltConfigs,
    device: str | None = None,
) -> None:
    """Validate shared reward parking without constructing a reward model."""

    if not bool(resources.lifecycle.handoff.release_reward_after_score):
        return
    reward = built.reward
    if reward is None or reward.all_external_inference:
        return
    names = tuple(reward.weights)
    if not names:
        return
    from vrl.rewards.functions.registry import (
        validate_reward_memory_parking_components,
    )

    validate_reward_memory_parking_components(
        names,
        device=str(device or resources.reward_torch_device()),
        reward_kwargs=reward.kwargs,
        inference_configs=reward.inference_configs,
    )


def build_algorithm_and_evaluator(
    *,
    family_entry: ModelFamilyEntry,
    built: BuiltConfigs,
    collector_config: Any,
    scheduler: Any | None = None,
) -> AlgorithmEvaluatorPair:
    """Build the algorithm/evaluator pair for a strict online recipe."""

    if not family_entry.supports_policy_replay:
        raise RuntimeError(
            f"{family_entry.family} is generation-only: its runtime exposes no "
            "trainable actions, transition likelihoods, or policy replay evaluator",
        )
    algorithm_config = built.algorithm
    algorithm_section = built.root.algorithm
    if algorithm_section is None:
        raise ValueError("online recipe requires an algorithm section")
    kind = algorithm_section.kind
    diffusion_logprob_kinds = {"grpo", "dance_grpo", "flow_dppo", "grpo_guard"}
    precision = built.precision
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

        expected_config_type = {
            "grpo": GRPOConfig,
            "dance_grpo": GRPOConfig,
            "flow_dppo": FlowDPPOConfig,
            "grpo_guard": GRPOGuardConfig,
        }[kind]
        if not isinstance(algorithm_config, expected_config_type):
            raise TypeError(
                f"{family_entry.family} {kind} expects {expected_config_type.__name__}, got "
                f"{type(algorithm_config).__name__}",
            )
        is_chunk_autoregressive = (
            family_entry.policy_semantics.generation_regime == "chunk_autoregressive"
        )
        if is_chunk_autoregressive and float(getattr(algorithm_config, "sft_weight", 0.0)) > 0:
            raise ValueError(
                f"{family_entry.family} grouped causal-chunk replay does not "
                "implement the full-sequence scheduler target required by "
                "algorithm.sft_weight; set sft_weight=0",
            )
        if kind == "flow_dppo":
            algorithm: object = FlowDPPO(algorithm_config)
        elif kind == "grpo_guard":
            algorithm = GRPOGuard(algorithm_config)
        else:
            algorithm = GRPO(algorithm_config)
        if is_chunk_autoregressive:
            if precision.diffusion_math != "fp32":
                raise ValueError(
                    f"{family_entry.family} uses an exact fp32 Gaussian re-noise "
                    "policy; precision.diffusion_math.dtype overrides are not "
                    "implemented for grouped causal-chunk replay",
                )
            if kind == "dance_grpo":
                raise ValueError(
                    f"{family_entry.family} uses one ordered full-trajectory replay; "
                    "DanceGRPO's random denoise-timestep subset is not defined for "
                    "the [temporal_chunk, denoise_transition] policy axes. Use grpo.",
                )
            if kind in {"flow_dppo", "grpo_guard"}:
                raise ValueError(
                    f"{family_entry.family} uses grouped causal-chunk replay; "
                    f"algorithm.kind={kind!r} requires reverse-SDE dt signals that "
                    "the chunk re-noise policy does not expose. Use grpo.",
                )
            from vrl.rollouts.evaluators.denoise import (
                ChunkAutoregressiveDenoiseLogProbEvaluator,
            )

            return AlgorithmEvaluatorPair(
                algorithm=algorithm,
                evaluator=ChunkAutoregressiveDenoiseLogProbEvaluator(),
            )

        math_dtype = resolve_torch_dtype(precision.diffusion_math)
        from vrl.rollouts.evaluators.denoise.sde_logprob import (
            DiffusionSDELogProbEvaluator,
        )

        return AlgorithmEvaluatorPair(
            algorithm=algorithm,
            evaluator=DiffusionSDELogProbEvaluator(
                scheduler,
                noise_level=float(
                    collector_config.request_sampling.get("noise_level", 1.0),
                ),
                sde_type=str(
                    collector_config.request_sampling.get("sde_type", "flow_grpo"),
                ),
                math_dtype=math_dtype,
            ),
        )

    if kind == "token_grpo":
        from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig

        if not isinstance(algorithm_config, TokenGRPOConfig):
            raise TypeError(
                f"{family_entry.family} token GRPO expects TokenGRPOConfig, got "
                f"{type(algorithm_config).__name__}",
            )
        if family_entry.policy_semantics.action_distribution == "continuous":
            from vrl.rollouts.evaluators.token import ContinuousTokenLogProbEvaluator

            evaluator = ContinuousTokenLogProbEvaluator()
        else:
            from vrl.rollouts.evaluators.token import TokenLogProbEvaluator

            evaluator = TokenLogProbEvaluator()
        return AlgorithmEvaluatorPair(algorithm=TokenGRPO(algorithm_config), evaluator=evaluator)

    if kind == "token_grpo_multisegment":
        from vrl.algorithms.grpo.multisegment import (
            MultiSegmentTokenGRPO,
            MultiSegmentTokenGRPOConfig,
        )
        from vrl.rollouts.evaluators.token import MultiSegmentTokenLogProbEvaluator

        if family_entry.family != "janus_pro_r1":
            raise ValueError(
                "token_grpo_multisegment currently requires model family janus_pro_r1",
            )
        if not isinstance(algorithm_config, MultiSegmentTokenGRPOConfig):
            raise TypeError(
                "multi-segment token GRPO expects MultiSegmentTokenGRPOConfig, "
                f"got {type(algorithm_config).__name__}",
            )
        segment_flags = dict(algorithm_config.train_segments or {})
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
                f"{family_entry.family} DiffusionNFT expects DiffusionNFTConfig, got "
                f"{type(algorithm_config).__name__}",
            )
        return AlgorithmEvaluatorPair(
            algorithm=DiffusionNFT(algorithm_config),
            evaluator=None,
        )

    if kind == "diffusion_dpo":
        raise ValueError(
            "diffusion_dpo is an offline recipe and is not supported by common online recipe",
        )

    raise ValueError(f"unsupported online algorithm.kind: {kind!r}")


__all__ = [
    "AlgorithmEvaluatorPair",
    "build_algorithm_and_evaluator",
    "build_reward",
    "validate_reward_memory_parking",
]
