"""DiffusionNFT-style objective for diffusion world-model RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.config_contract import AlgorithmConfigContract
from vrl.algorithms.previous_adapter import (
    flipped_advantage_losses,
    flow_time,
    forward_process_prediction,
    sync_previous_policy_adapter,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
from vrl.models.precision import model_autocast


def normalized_mse(prediction: Any, target: Any) -> Any:
    """Return per-sample MSE normalized by detached mean absolute error."""

    import torch

    reduce_dims = tuple(range(1, target.ndim))
    with torch.no_grad():
        weight = (
            torch.abs(prediction.double() - target.double())
            .mean(
                dim=reduce_dims,
                keepdim=True,
            )
            .clip(min=1e-5)
        )
    return ((prediction - target) ** 2 / weight).mean(dim=reduce_dims)


@dataclass(slots=True)
class DiffusionNFTConfig:
    """Hyper-parameters for the DiffusionNFT training objective."""

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=True,
        supports_step_kl_reward=True,
        sft_source="unsupported",
        requires_previous_adapter=True,
    )

    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False
    nft_beta: float = 1.0
    kl_coef: float = 1.0
    advantage_scale: float = 5.0
    weight_copy_decay: float = 0.0


class DiffusionNFT:
    """DiffusionNFT-style GRPO objective.

    This objective does not consume evaluator log-prob signals. It trains from
    generated clean latents, prompt embeddings, sampled diffusion timesteps, and
    video-level rewards. This algorithm is diffusion-specific and owns its
    model-forward objective assembly.
    """

    uses_evaluator = False
    # Replay-branch contract (AlgorithmAdapter.validate_inputs): NFT trains the
    # forward process from these rollout tensors only — no reverse-SDE
    # trajectory, no log-probs. Declaring them lets the adapter fail fast with
    # available-vs-missing diagnostics, replacing the old inline per-key check.
    required_data_keys = ("latents_clean", "prompt_embeds", "timesteps")
    required_signal_keys: tuple[str, ...] = ()
    needs_kl_intermediates = False
    requires_active_trust_region = False
    # DiffusionNFT is likelihood-free: it computes no importance-sampling ratio
    # to reweight off-policy samples, and its positive/negative decomposition is
    # taken against a previous-policy adapter that ``after_optimizer_step``
    # refreshes every step. Training on rollouts generated under a superseded
    # policy is therefore silently biased, not just noisy. So unlike GRPO (whose
    # IS ratio absorbs a bounded version lag), NFT requires strictly on-policy
    # data — the continuous-rollout staleness window must be 0. Consumed by
    # build_rollout_schedule to fail fast on an unsound max_stale>0 config.
    tolerates_off_policy_staleness = False

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
        """NFT's lr=0 invariant: flipping advantages must not change the loss.

        Ratio-style parity is blind to NFT (it computes no log-prob ratio); the
        equivalent collection-time check is advantage antisymmetry — with the
        previous adapter freshly synced, the loss is invariant to flipping the
        advantage signs.

        Called by the trainer's debug.first_step branch through this optional
        protocol method, keeping algorithm-specific checks out of the trainer.
        """

        loss, flipped_loss = flipped_advantage_losses(
            self,
            model=model,
            batch=batch,
            advantages=advantages,
            timestep_index=timestep_index,
        )
        abs_diff = abs(loss - flipped_loss)
        return {
            "event": "first_step_nft_invariant",
            "invariant": "advantage_flip",
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
        if advantage_scale <= 0:
            raise RuntimeError("DiffusionNFTConfig.advantage_scale must be > 0")
        replay_tensors = TrajectoryReader.from_batch(batch).replay_tensor_dict("denoise")
        # Presence + tensor-type of these keys is enforced upstream by
        # AlgorithmAdapter.validate_inputs (declared in required_data_keys);
        # read them directly here.
        x0 = replay_tensors["latents_clean"]
        prompt_embeds = replay_tensors["prompt_embeds"]
        timesteps = replay_tensors["timesteps"]
        timestep_width = 1 if timesteps.ndim == 1 else int(timesteps.shape[1])
        if not 0 <= timestep_index < timestep_width:
            raise RuntimeError(
                "DiffusionNFT timestep_index out of range: "
                f"timestep_index={timestep_index}, width={timestep_width}, "
                f"timesteps.shape={tuple(timesteps.shape)}",
            )
        t_raw = timesteps if timesteps.ndim == 1 else timesteps[:, timestep_index]

        if x0.shape[0] != prompt_embeds.shape[0]:
            raise RuntimeError(
                "DiffusionNFT batch mismatch: latents_clean and prompt_embeds "
                f"have leading dims {x0.shape[0]} and {prompt_embeds.shape[0]}",
            )

        if advantages.shape[0] != x0.shape[0]:
            raise RuntimeError(
                "DiffusionNFT batch mismatch: advantages and latents_clean "
                f"have leading dims {advantages.shape[0]} and {x0.shape[0]}",
            )

        t = flow_time(t_raw, x0, owner="DiffusionNFT")
        t = t.to(dtype=x0.dtype)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = replay_tensors.get("diffusion_nft_noise")
        if noise is None:
            noise = torch.randn_like(x0.float())
        else:
            noise = noise.to(device=x0.device, dtype=torch.float32)
        xt = (1 - t_expanded) * x0.float() + t_expanded * noise
        xt_input = xt.to(x0.dtype)

        # Three evaluations of the family's own conditional forward at this
        # trajectory step: the frozen behaviour policy, the trainable policy,
        # and the adapter-free base as the KL reference.
        with (
            model.activate_adapter("previous"),
            torch.no_grad(),
            model_autocast(model, x0.device),
        ):
            previous_prediction = forward_process_prediction(
                model, batch, timestep_index, xt_input, owner="DiffusionNFT"
            ).detach()
        with model_autocast(model, x0.device):
            forward_prediction = forward_process_prediction(
                model, batch, timestep_index, xt_input, owner="DiffusionNFT"
            )
        with (
            model.disable_adapter(),
            torch.no_grad(),
            model_autocast(model, x0.device),
        ):
            ref_prediction = forward_process_prediction(
                model, batch, timestep_index, xt_input, owner="DiffusionNFT"
            ).detach()

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
        if beta <= 0:
            raise RuntimeError("DiffusionNFTConfig.nft_beta must be > 0")
        positive_prediction = beta * forward_prediction + (1.0 - beta) * previous_prediction
        negative_prediction = (1.0 + beta) * previous_prediction - beta * forward_prediction

        x0_float = x0.float()
        positive_x0 = xt - t_expanded * positive_prediction.float()
        negative_x0 = xt - t_expanded * negative_prediction.float()
        positive_loss = normalized_mse(positive_x0, x0_float)
        negative_loss = normalized_mse(negative_x0, x0_float)

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

    def after_optimizer_step(self, model: Any, global_step: int) -> None:
        """Refresh the previous-policy adapter after an optimizer step."""

        del global_step
        sync_previous_policy_adapter(
            model, decay=self.config.weight_copy_decay, owner="DiffusionNFT"
        )


__all__ = ["DiffusionNFT", "DiffusionNFTConfig"]
