"""Factory functions for common online training recipes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.algorithms.base import Algorithm
from vrl.config.builders import BuiltConfigs
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.registry import ModelFamilyEntry
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rewards import RewardRuntime
from vrl.rewards.base import RewardFunction
from vrl.rollouts.evaluators.base import Evaluator
from vrl.run import ResolvedReward

if TYPE_CHECKING:
    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.rewards.ray import RayRewardPlacement


def resolve_reward_actor_placement(
    reward: ResolvedReward,
    owner: GlobalRayPlacementOwner,
) -> RayRewardPlacement:
    """Bind a component deployment to the run's acquired resource ownership."""
    from vrl.rewards.ray import RayRewardPlacement

    if reward.device.startswith("cuda") and reward.memory_parking_required:
        import torch

        from vrl.ray.dependencies import require_ray

        # CUDA ordinals in a driver mask are not Ray's physical GPU IDs.
        device_index = torch.device(reward.device).index
        ordinal = torch.cuda.current_device() if device_index is None else device_index
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_gpu = int(mask.split(",")[ordinal]) if mask else ordinal
        return RayRewardPlacement(
            shared_gpu_id=physical_gpu,
            node_id=str(require_ray().get_runtime_context().get_node_id()),
        )
    return RayRewardPlacement(placement=owner.reward_placement)


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluatorPair:
    """Algorithm instance plus its optional evaluator."""

    algorithm: Algorithm
    evaluator: Evaluator | None

    @classmethod
    def from_configs(
        cls,
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
        if kind == "diffusion_dpo":
            raise ValueError(
                "diffusion_dpo is an offline recipe and is not supported by common online recipe",
            )
        reward = built.reward
        if reward is None:
            raise ValueError("online recipe requires reward configuration")
        diffusion_logprob_kinds = {"grpo", "dance_grpo", "flash_grpo", "flow_dppo", "grpo_guard"}
        precision = built.precision
        if precision.denoise_math != "fp32" and kind not in diffusion_logprob_kinds:
            raise ValueError(
                "precision.denoise_math.dtype overrides are supported only by "
                "diffusion log-prob "
                f"objectives; algorithm.kind={kind!r} keeps its protected math in fp32",
            )

        if kind in diffusion_logprob_kinds:
            # These are flow-matching GRPO-family algorithms on the same SDE
            # evaluator. dance_grpo reuses FlowGRPO unchanged (its delta is the
            # trainer's random timestep selection + multi-reward); flow_dppo /
            # grpo_guard are trust-region variants whose loss reads the rollout
            # proposal mean, checked against the recipe below.
            from vrl.algorithms.grpo.continuous import GRPO, FlashGRPO, FlowDPPO, GRPOGuard

            is_chunk_autoregressive = (
                family_entry.policy_semantics.generation_regime == "chunk_autoregressive"
            )
            if is_chunk_autoregressive and float(getattr(algorithm_config, "sft_weight", 0.0)) > 0:
                raise ValueError(
                    f"{family_entry.family} grouped causal-chunk replay does not "
                    "implement the full-sequence scheduler target required by "
                    "algorithm.sft_weight; set sft_weight=0",
                )
            advantage_estimator = algorithm_config.build_estimator(
                component_weights=reward.weights,
            )
            if kind == "flow_dppo":
                algorithm_type = FlowDPPO
            elif kind == "grpo_guard":
                algorithm_type = GRPOGuard
            elif kind == "flash_grpo":
                algorithm_type = FlashGRPO
            else:
                algorithm_type = GRPO
            algorithm = algorithm_type(
                algorithm_config,
                advantage_estimator=advantage_estimator,
            )
            if is_chunk_autoregressive:
                if precision.denoise_math != "fp32":
                    raise ValueError(
                        f"{family_entry.family} uses an exact fp32 Gaussian re-noise "
                        "policy; precision.denoise_math.dtype overrides are not "
                        "implemented for grouped causal-chunk replay",
                    )
                if kind == "dance_grpo":
                    raise ValueError(
                        f"{family_entry.family} uses one ordered full-trajectory replay; "
                        "DanceGRPO's random denoise-timestep subset is not defined for "
                        "the [temporal_chunk, denoise_transition] policy axes. Use grpo.",
                    )
                if kind in {"flash_grpo", "flow_dppo", "grpo_guard"}:
                    raise ValueError(
                        f"{family_entry.family} uses grouped causal-chunk replay; "
                        f"algorithm.kind={kind!r} requires reverse-SDE dt signals that "
                        "the batch re-noise policy does not expose. Use grpo.",
                    )
                from vrl.rollouts.evaluators.denoise import (
                    ChunkAutoregressiveDenoiseLogProbEvaluator,
                )

                return cls(
                    algorithm=algorithm,
                    evaluator=ChunkAutoregressiveDenoiseLogProbEvaluator(),
                )

            math_dtype = resolve_torch_dtype(precision.denoise_math)
            denoise = collector_config.denoise or DenoiseRequestOptions()
            if algorithm.requires_active_trust_region and not denoise.return_prev_sample_mean:
                raise ValueError(
                    f"algorithm.kind={kind!r} measures the current-vs-rollout proposal "
                    "drift, which needs the rollout mean stored at generation; set "
                    "rollout.return_prev_sample_mean=true",
                )
            from vrl.rollouts.evaluators.denoise.sde_logprob import (
                DenoiseSDELogProbEvaluator,
            )

            return cls(
                algorithm=algorithm,
                evaluator=DenoiseSDELogProbEvaluator(
                    scheduler,
                    noise_level=denoise.noise_level,
                    sde_type=denoise.sde_type or "flow_grpo",
                    math_dtype=math_dtype,
                ),
            )

        if kind == "diffusion_nft":
            from vrl.algorithms.diffusion_nft import DiffusionNFT

            return cls(
                algorithm=DiffusionNFT(algorithm_config),
                evaluator=None,
            )

        if kind == "v_grpo":
            from vrl.algorithms.v_grpo import VGRPO

            return cls(
                algorithm=VGRPO(algorithm_config),
                evaluator=None,
            )

        raise ValueError(f"unsupported online algorithm.kind: {kind!r}")


def build_reward_function(
    reward: ResolvedReward, *, ray_placement: RayRewardPlacement | None = None
) -> RewardFunction:
    """Build the online reward function from the resolved reward inputs.

    Device and parking policy are decided once by ``ResolvedOnlineRun.
    reward_inputs``; this factory validates the components and constructs.
    In-process GPU ownership decides parking: a shared reward must completely
    park its model memory after scoring, while a dedicated reward stays resident.
    HTTP components own their deployment externally and receive no local parking
    policy. YAML selects transport, not lifecycle behavior.
    """

    config = reward.config
    config.require_online_training()
    from vrl.rewards.functions.registry import MultiReward

    combination, axis_mapping = None, None
    if config.calibration is not None:
        from vrl.rewards.deployment import RewardDeployment

        deployment = RewardDeployment.load(config)
        combination, axis_mapping = deployment.combination, deployment.axis_mapping

    if reward.memory_parking_required and not config.all_external_inference:
        from vrl.rewards.functions.registry import (
            validate_reward_memory_parking_components,
        )

        validate_reward_memory_parking_components(
            tuple(config.weights),
            device=reward.device,
            reward_kwargs=config.kwargs,
            inference_configs=config.inference_configs,
        )

    return MultiReward.from_dict(
        config.weights,
        device=reward.device,
        reward_kwargs=config.kwargs,
        memory_parking_required=reward.memory_parking_required,
        inference_configs=config.inference_configs,
        ray_placement=ray_placement,
        combination=combination,
        axis_mapping=axis_mapping,
        image_float32_inputs=combination is not None,
    )


def build_reward_runtime(
    reward: ResolvedReward, *, ray_placement: RayRewardPlacement | None = None
) -> RewardRuntime:
    """Build the collector-facing runtime around the configured reward function."""

    from vrl.rewards.runtime import RewardFunctionRuntime

    return RewardFunctionRuntime(build_reward_function(reward, ray_placement=ray_placement))


def validate_reward_memory_parking(
    *,
    resources: ResolvedDistributedResources,
    built: BuiltConfigs,
    device: str | None = None,
) -> None:
    """Validate shared reward parking without constructing a reward model."""

    if not bool(resources.lifecycle.offload_reward):
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


__all__ = [
    "AlgorithmEvaluatorPair",
    "build_reward_function",
    "build_reward_runtime",
    "validate_reward_memory_parking",
]
