"""DiffusionNFT-style objective for diffusion world-model RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.config_contract import AlgorithmConfigContract
from vrl.algorithms.previous_policy import PreviousPolicyObjective
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
from vrl.models.precision import model_autocast


@dataclass(slots=True)
class DiffusionNFTConfig:
    """Hyper-parameters for the DiffusionNFT training objective."""

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=True,
        sft_source="unsupported",
        requires_previous_policy=True,
        requires_reference_policy=True,
    )

    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False
    nft_beta: float = 1.0
    kl_coef: float = 1.0
    advantage_scale: float = 5.0
    weight_copy_decay: float = 0.0

    def __post_init__(self) -> None:
        if float(self.nft_beta) <= 0:
            raise ValueError(f"DiffusionNFTConfig.nft_beta must be > 0, got {self.nft_beta}")
        if float(self.advantage_scale) <= 0:
            raise ValueError(
                f"DiffusionNFTConfig.advantage_scale must be > 0, got {self.advantage_scale}",
            )


class DiffusionNFT(PreviousPolicyObjective):
    """DiffusionNFT-style GRPO objective.

    This objective does not consume evaluator log-prob signals. It trains from
    generated clean latents, prompt embeddings, sampled diffusion timesteps, and
    video-level rewards. This algorithm is diffusion-specific and owns its
    model-forward objective assembly. Likelihood-free: it computes no
    importance-sampling ratio, and its positive/negative decomposition is taken
    against the previous policy the parent refreshes every step.
    """

    def __init__(self, config: DiffusionNFTConfig | None = None) -> None:
        self.config = config or DiffusionNFTConfig()

    def compute_advantages_from_tensors(
        self,
        rewards: Any,
        group_ids: Any,
    ) -> Any:
        cfg = self.config
        return group_relative_advantages(
            rewards,
            group_ids,
            eps=cfg.eps,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
        )

    def first_step_invariant_check(
        self,
        *,
        model: Any,
        batch: Any,
        advantages: Any,
        timestep_index: int = 0,
        threshold: float = 1.0e-6,
    ) -> dict[str, Any]:
        """lr=0 invariant: flipping the advantages must not change the loss."""

        loss, flipped_loss = self._flipped_advantage_losses(
            model, batch, advantages, timestep_index
        )
        abs_diff = abs(loss - flipped_loss)
        return {
            "loss": loss,
            "flipped_loss": flipped_loss,
            "abs_diff": abs_diff,
            "threshold": threshold,
            "passed": abs_diff <= threshold,
        }

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

        # Three evaluations of the family's own conditional forward at this
        # trajectory step: the frozen behaviour policy, the pre-training
        # reference for the KL term, and — last — the trainable policy. The two
        # frozen policies swap weights in place, so they run before the live
        # forward whose graph backward will read.
        with (
            model.previous_policy(),
            torch.no_grad(),
            model_autocast(model, x0.device),
        ):
            previous_prediction = model.replay_forward_with_latents(
                batch, timestep_index, xt_input, classifier_free_guidance=False
            )["noise_pred"].detach()
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

        # Advantages are already clamped to ±adv_clip_max upstream in
        # compute_advantages_from_tensors (group_relative_advantages). The final
        # .clamp(0.0, 1.0) on reward_mix makes any second ±advantage_scale clamp
        # on `adv` provably redundant — clamp(clamp(a,-s,s)/s/2+0.5, 0, 1) equals
        # clamp(a/s/2+0.5, 0, 1) for all a — so read advantages directly.
        adv = advantages.to(device=x0.device, dtype=forward_prediction.dtype)
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


__all__ = ["DiffusionNFT", "DiffusionNFTConfig"]
