"""DiffusionNFT-style objective for diffusion world-model RL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vrl.algorithms.advantages import GroupAdvantageConfig, GroupRelativeObjective
from vrl.algorithms.previous_policy import PreviousPolicyObjective
from vrl.algorithms.requirements import AlgorithmRequirements
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
from vrl.models.precision import model_autocast


@dataclass(slots=True)
class DiffusionNFTConfig(GroupAdvantageConfig):
    """Hyper-parameters for the DiffusionNFT training objective."""

    requirements: ClassVar[AlgorithmRequirements] = AlgorithmRequirements(
        needs_sde_rollout=True,
        sft_source="unsupported",
        requires_previous_policy=True,
        requires_reference_policy=True,
    )

    eps: float = 1e-8
    # Keep the existing positional constructor fields in their original order.
    advantage_combine: str = field(default="normalized_sum", kw_only=True)
    nft_beta: float = 1.0
    kl_coef: float = 1.0
    advantage_scale: float = 5.0

    def __post_init__(self) -> None:
        GroupAdvantageConfig.__post_init__(self)
        if float(self.nft_beta) <= 0:
            raise ValueError(f"DiffusionNFTConfig.nft_beta must be > 0, got {self.nft_beta}")
        if float(self.advantage_scale) <= 0:
            raise ValueError(
                f"DiffusionNFTConfig.advantage_scale must be > 0, got {self.advantage_scale}",
            )


class DiffusionNFT(PreviousPolicyObjective, GroupRelativeObjective):
    """DiffusionNFT-style GRPO objective.

    This objective does not consume evaluator log-prob signals. It trains from
    generated clean latents, prompt embeddings, sampled diffusion timesteps, and
    video-level rewards. This algorithm is diffusion-specific and owns its
    model-forward objective assembly. Likelihood-free: it computes no
    importance-sampling ratio, and its positive/negative decomposition is taken
    against ``theta_old``, the detached current prediction.
    """

    config: DiffusionNFTConfig

    def __init__(
        self,
        config: DiffusionNFTConfig,
        *,
        component_weights: Mapping[str, float] | None = None,
    ) -> None:
        GroupRelativeObjective.__init__(self, config, component_weights=component_weights)

    @property
    def kl_coef(self) -> float:
        return float(self.config.kl_coef)

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        return self.compute_batch_timestep_loss(
            inputs.model,
            inputs.rollout_batch,
            inputs.timestep_index,
            inputs.advantages,
        )

    def compute_batch_timestep_loss(
        self,
        model: Any,
        batch: Any,
        timestep_index: int,
        advantages: Any,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute one DiffusionNFT loss slice for a rollout batch/timestep."""

        import torch

        from vrl.trajectory.reader import TrajectoryReader

        cfg = self.config
        advantage_scale = float(cfg.advantage_scale)
        replay = TrajectoryReader.from_batch(batch).forward_process_replay(
            "denoise", timestep_index
        )
        x0 = replay.latents_clean
        t = self.flow_time(replay.timestep, x0).to(dtype=x0.dtype)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        if replay.noise is None:
            noise = torch.randn_like(x0.float())
        else:
            noise = replay.noise.to(device=x0.device, dtype=torch.float32)
        xt = (1 - t_expanded) * x0.float() + t_expanded * noise
        xt_input = xt.to(x0.dtype)

        # Two evaluations of the family's own conditional forward at this
        # trajectory step: the pre-training reference for the KL term, then the
        # trainable policy. The frozen reference may swap weights in place, so
        # it runs before the live forward whose graph backward will read.
        # theta_old is the policy as of this optimizer step: the detached live
        # prediction, not a third forward.
        with (
            model.reference_policy(),
            torch.no_grad(),
            model_autocast(model, x0.device),
        ):
            ref_prediction = model.replay_forward_with_latents(
                batch, timestep_index, xt_input, classifier_free_guidance=False
            )["noise_pred"].detach()
        with model_autocast(model, x0.device):
            forward_prediction = model.replay_forward_with_latents(
                batch, timestep_index, xt_input, classifier_free_guidance=False
            )["noise_pred"]
        previous_prediction = forward_prediction.detach()

        # Advantages are already clamped to ±adv_clip_max upstream in
        # the shared advantage estimator. The final
        # .clamp(0.0, 1.0) on reward_mix makes any second ±advantage_scale clamp
        # on `adv` provably redundant — clamp(clamp(a,-s,s)/s/2+0.5, 0, 1) equals
        # clamp(a/s/2+0.5, 0, 1) for all a — so read advantages directly.
        # Keep complementary loss weights in FP32: BF16 rounding can make
        # mix + (1 - mix) differ from one even with identical branch losses.
        adv = advantages.to(device=x0.device, dtype=torch.float32)
        while adv.ndim < forward_prediction.ndim:
            adv = adv.unsqueeze(-1)
        reward_mix = ((adv / advantage_scale) / 2.0 + 0.5).clamp(0.0, 1.0)

        beta = float(cfg.nft_beta)
        positive_prediction = beta * forward_prediction + (1.0 - beta) * previous_prediction
        negative_prediction = (1.0 + beta) * previous_prediction - beta * forward_prediction

        x0_float = x0.float()
        positive_x0 = xt - t_expanded * positive_prediction.float()
        negative_x0 = xt - t_expanded * negative_prediction.float()
        positive_loss = self.normalized_mse(positive_x0, x0_float)
        negative_loss = self.normalized_mse(negative_x0, x0_float)

        flat_mix = reward_mix.flatten(start_dim=1).mean(dim=1)
        original_policy_loss = (
            flat_mix * positive_loss / beta + (1.0 - flat_mix) * negative_loss / beta
        )
        policy_loss = original_policy_loss.mean() * advantage_scale
        auxiliary_loss = self.compute_clean_latent_auxiliary_loss(
            batch, xt - t_expanded * forward_prediction.float()
        )
        if auxiliary_loss is not None:
            if (
                not isinstance(auxiliary_loss, torch.Tensor)
                or auxiliary_loss.ndim != 0
                or not bool(torch.isfinite(auxiliary_loss))
            ):
                raise ValueError("NFT clean-latent auxiliary loss must be a finite scalar tensor")
            policy_loss = policy_loss + auxiliary_loss
        kl_loss = ((forward_prediction.float() - ref_prediction.float()) ** 2).mean()
        kl_term = float(cfg.kl_coef) * kl_loss
        loss = policy_loss + kl_term
        kl_value = float(kl_loss.detach().item())

        return loss, TrainStepMetrics(
            loss=float(loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            kl_penalty=kl_value,
            weighted_kl_loss=float(kl_term.detach().item()),
            update=PolicyUpdateStats(
                approx_kl=kl_value,
            ),
        )

    def compute_clean_latent_auxiliary_loss(self, batch: Any, predicted_clean: Any) -> Any | None:
        """Optional task loss on the live policy's FP32 clean prediction.

        A task adapter can constrain selected spatial cells without duplicating
        NFT's frozen-policy forwards or inserting targets at inference time.
        The default contributes no loss. Targets and masks belong to the task.
        """

        return None


__all__ = ["DiffusionNFT", "DiffusionNFTConfig"]
