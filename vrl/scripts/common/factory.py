"""The online recipe's algorithm/evaluator pairing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.config.builders import BuiltConfigs
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.registry import ModelFamilyEntry
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.evaluators.base import Evaluator


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
            # proposal mean (config rules require it to be stored).
            from vrl.algorithms.grpo.continuous import GRPO, FlashGRPO, FlowDPPO, GRPOGuard

            is_chunk_autoregressive = (
                family_entry.policy_semantics.generation_regime == "chunk_autoregressive"
            )
            advantage_estimator = algorithm_config.build_estimator(
                component_weights=reward.weights,
            )
            trainer_config = built.trainer
            correction = None if trainer_config is None else trainer_config.precision_correction
            denoise = (
                RolloutCollectorConfig.from_root(built.root).denoise or DenoiseRequestOptions()
            )
            if kind == "flash_grpo":
                # The rectification weight is defined over the rollout SDE the
                # replay evaluator integrates: same scheduler, same noise.
                algorithm = FlashGRPO(
                    algorithm_config,
                    scheduler=scheduler,
                    noise_level=denoise.noise_level,
                    sde_type=denoise.sde_type or "flow_grpo",
                    advantage_estimator=advantage_estimator,
                    precision_correction=correction,
                )
            else:
                algorithm_type = {"flow_dppo": FlowDPPO, "grpo_guard": GRPOGuard}.get(kind, GRPO)
                algorithm = algorithm_type(
                    algorithm_config,
                    advantage_estimator=advantage_estimator,
                    precision_correction=correction,
                )
            if is_chunk_autoregressive:
                if algorithm.sft_weight > 0:
                    raise ValueError(
                        f"{family_entry.family} grouped causal-chunk replay does not "
                        "implement the full-sequence scheduler target required by "
                        "algorithm.sft_weight; set sft_weight=0",
                    )
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
                algorithm=DiffusionNFT(
                    algorithm_config,
                    advantage_estimator=algorithm_config.build_estimator(
                        component_weights=reward.weights,
                    ),
                ),
                evaluator=None,
            )

        if kind == "v_grpo":
            from vrl.algorithms.v_grpo import VGRPO

            return cls(
                algorithm=VGRPO(algorithm_config),
                evaluator=None,
            )

        raise ValueError(f"unsupported online algorithm.kind: {kind!r}")


__all__ = [
    "AlgorithmEvaluatorPair",
]
