"""DiffusionNFT-style objective for diffusion world-model RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.base import Algorithm
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import TrainStepMetrics
from vrl.math.diffusion.nft import normalized_mse


@dataclass(slots=True)
class DiffusionNFTConfig:
    """Hyper-parameters for the DiffusionNFT training objective."""

    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False
    nft_beta: float = 1.0
    kl_beta: float = 1.0
    advantage_high: float = 5.0
    advantage_low: float = -5.0
    weight_copy_decay: float = 0.0


class DiffusionNFT(Algorithm):
    """DiffusionNFT-style GRPO objective.

    This objective does not consume evaluator log-prob signals. It trains from
    generated clean latents, prompt embeddings, sampled diffusion timesteps, and
    video-level rewards. This algorithm is diffusion-specific and owns its
    model-forward objective assembly.
    """

    uses_evaluator = False

    def __init__(self, config: DiffusionNFTConfig | None = None) -> None:
        self.config = config or DiffusionNFTConfig()
        self._last_policy_loss_tensor: Any = None
        self._last_kl_term_tensor: Any = None

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
        advantage signs. The RNG is forked and seeded so both evaluations draw
        the same NFT noise when the trajectory carries none.

        Called by the trainer's debug.first_step branch through this optional
        protocol method, keeping algorithm-specific checks out of the trainer.
        """

        import torch

        def _loss(adv: Any) -> float:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                loss, _ = self.compute_batch_timestep_loss(
                    model,
                    batch,
                    timestep_index,
                    adv,
                )
            return float(loss.detach().float().item())

        loss = _loss(advantages)
        flipped_loss = _loss(-advantages)
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
        model = inputs.metadata.get("model")
        batch = inputs.metadata.get("rollout_batch")
        timestep_index = int(inputs.metadata.get("timestep_index", 0))
        if model is None:
            raise RuntimeError("DiffusionNFT AlgorithmInput.metadata['model'] is required")
        if batch is None:
            raise RuntimeError("DiffusionNFT AlgorithmInput.metadata['rollout_batch'] is required")
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for DiffusionNFT")
        return self.compute_batch_timestep_loss(
            model,
            batch,
            timestep_index,
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

        from vrl.trajectory import TrajectoryResolver

        cfg = self.config
        replay_tensors = TrajectoryResolver.from_batch(batch).replay_tensor_dict("denoise")
        required_tensors = {}
        for key in ("latents_clean", "prompt_embeds", "timesteps"):
            value = replay_tensors.get(key)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"DiffusionNFT requires trajectory replay tensor {key!r}; "
                    f"got {type(value).__name__}",
                )
            required_tensors[key] = value
        x0 = required_tensors["latents_clean"]
        prompt_embeds = required_tensors["prompt_embeds"]
        timesteps = required_tensors["timesteps"]
        if timesteps.ndim == 1:
            t_raw = timesteps
        else:
            if timestep_index >= timesteps.shape[1]:
                raise RuntimeError(
                    "DiffusionNFT timestep_index out of range: "
                    f"timestep_index={timestep_index}, timesteps.shape={tuple(timesteps.shape)}",
                )
            t_raw = timesteps[:, timestep_index]

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

        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise RuntimeError("DiffusionNFT model must expose a transformer module")

        prepare = getattr(model, "diffusion_nft_prepare_transformer_input", None)
        if not callable(prepare):
            raise RuntimeError(
                "DiffusionNFT model must expose diffusion_nft_prepare_transformer_input(...)",
            )

        t = t_raw.to(device=x0.device, dtype=torch.float32)
        if bool((t > 1.0).any()):
            t = t / 1000.0
        # The /1000 heuristic assumes a [0, 1] or [0, 1000] timestep grid.
        # EDM-style grids (e.g. Cosmos Predict2 FlowMatch, timesteps up to
        # 80000) would land far outside [0, 1] and silently push the
        # xt = (1-t)*x0 + t*noise interpolation off the data manifold — the
        # same failure shape as the predict2 sigma-domain incident. Fail loud.
        if bool((t > 1.0).any()) or bool((t < 0.0).any()):
            raise RuntimeError(
                "DiffusionNFT timestep grid must normalize into [0, 1]; got "
                f"min={float(t.min()):.4g}, max={float(t.max()):.4g} after "
                "the /1000 heuristic. EDM-scale timestep grids are not "
                "supported by this normalization.",
            )
        t = t.to(dtype=x0.dtype)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = replay_tensors.get("diffusion_nft_noise")
        if noise is None:
            noise = torch.randn_like(x0.float())
        else:
            noise = noise.to(device=x0.device, dtype=torch.float32)
        xt = (1 - t_expanded) * x0.float() + t_expanded * noise

        transformer_inputs = prepare(
            latents=xt.to(x0.dtype),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            pooled_prompt_embeds=replay_tensors.get("pooled_prompt_embeds"),
            timestep=t_raw,
            num_frames=int(
                batch.context.get(
                    "num_frames",
                    int(x0.shape[2]) if getattr(x0, "ndim", 0) >= 3 else 1,
                )
            ),
            height=int(batch.context.get("height", 0)),
            width=int(batch.context.get("width", 0)),
        )

        previous_prediction = _forward_previous_policy_adapter(
            transformer,
            transformer_inputs,
        )
        forward_prediction = transformer(**transformer_inputs)[0]
        ref_prediction = _forward_reference(transformer, transformer_inputs)

        adv = torch.clamp(advantages, cfg.advantage_low, cfg.advantage_high)
        adv = adv.to(device=x0.device, dtype=forward_prediction.dtype)
        while adv.ndim < forward_prediction.ndim:
            adv = adv.unsqueeze(-1)
        reward_mix = ((adv / cfg.advantage_high) / 2.0 + 0.5).clamp(0.0, 1.0)

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
            flat_mix * positive_loss / beta
            + (1.0 - flat_mix) * negative_loss / beta
        )
        policy_loss = original_policy_loss.mean() * float(cfg.advantage_high)
        kl_loss = ((forward_prediction.float() - ref_prediction.float()) ** 2).mean()
        kl_term = float(cfg.kl_beta) * kl_loss
        loss = policy_loss + kl_term

        self._last_policy_loss_tensor = policy_loss
        self._last_kl_term_tensor = kl_term

        with torch.no_grad():
            previous_deviation = (
                (forward_prediction.float() - previous_prediction.float()) ** 2
            ).mean()
            approx_kl = ((forward_prediction.float() - ref_prediction.float()) ** 2).mean()

        return loss, TrainStepMetrics(
            loss=float(loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            kl_penalty=float(kl_loss.detach().item()),
            clip_fraction=0.0,
            approx_kl=float(approx_kl.detach().item()),
            advantage_mean=float(advantages.detach().float().mean().item()),
            grad_norm=0.0,
            adv_zero_rate=float((advantages.detach().abs() < 1e-6).float().mean().item()),
            adv_saturation=float(
                (advantages.detach().abs() >= cfg.adv_clip_max - 1e-6).float().mean().item(),
            ),
            phase_times={
                "diffusion_nft.previous_policy_deviation": float(
                    previous_deviation.item(),
                ),
            },
        )

    def after_optimizer_step(self, model: Any, global_step: int) -> None:
        """Refresh the previous-policy adapter after an optimizer step."""

        del global_step
        sync = getattr(model, "sync_previous_policy_adapter", None)
        if not callable(sync):
            raise RuntimeError(
                "DiffusionNFT model must expose "
                "sync_previous_policy_adapter(decay=...) "
                "for previous-policy refresh",
            )
        sync(decay=float(self.config.weight_copy_decay))


def _forward_previous_policy_adapter(transformer: Any, inputs: dict[str, Any]) -> Any:
    set_adapter = getattr(transformer, "set_adapter", None)
    if not callable(set_adapter):
        raise RuntimeError(
            "DiffusionNFT requires transformer.set_adapter('previous') "
            "for the previous-policy branch",
        )
    set_adapter("previous")
    import torch

    try:
        with torch.no_grad():
            return transformer(**inputs)[0].detach()
    finally:
        set_adapter("default")


def _forward_reference(transformer: Any, inputs: dict[str, Any]) -> Any:
    import torch

    disable_adapters = getattr(transformer, "disable_adapters", None)
    if callable(disable_adapters):
        transformer.disable_adapters()
        try:
            with torch.no_grad():
                return transformer(**inputs)[0].detach()
        finally:
            enable = getattr(transformer, "enable_adapters", None)
            if callable(enable):
                enable()
            set_adapter = getattr(transformer, "set_adapter", None)
            if callable(set_adapter):
                set_adapter("default")

    disable_adapter = getattr(transformer, "disable_adapter", None)
    if callable(disable_adapter):
        with disable_adapter(), torch.no_grad():
            return transformer(**inputs)[0].detach()

    raise RuntimeError(
        "DiffusionNFT requires a reference branch via transformer.disable_adapters() "
        "or transformer.disable_adapter()",
    )


__all__ = ["DiffusionNFT", "DiffusionNFTConfig"]
