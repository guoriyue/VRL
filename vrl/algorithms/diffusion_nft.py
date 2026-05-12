"""DiffusionNFT-style objective for diffusion world-model RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import SignalBatch


@dataclass(slots=True)
class DiffusionNFTConfig:
    """Hyper-parameters for the DiffusionNFT training objective."""

    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False
    per_prompt_stat_tracking: bool = True
    nft_beta: float = 1.0
    kl_beta: float = 1.0
    advantage_high: float = 5.0
    advantage_low: float = -5.0
    mini_batch: int = 1
    uncentralized_training: bool = True
    weight_copy_decay: float = 0.0


class DiffusionNFT(Algorithm):
    """DiffusionNFT-style GRPO objective.

    This objective does not consume evaluator log-prob signals. It trains from
    generated clean latents, prompt embeddings, sampled diffusion timesteps, and
    video-level rewards. Calling ``compute_signal_loss`` is therefore treated as
    a wiring error instead of silently falling back to continuous GRPO.
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
        import torch

        cfg = self.config
        advantages = torch.zeros_like(rewards)
        for gid in torch.unique(group_ids):
            mask = group_ids == gid
            group_rewards = rewards[mask]
            if group_rewards.numel() <= 1:
                advantages[mask] = 0.0
                continue
            mean = group_rewards.mean()
            std = rewards.std() if cfg.global_std else group_rewards.std()
            denom = torch.clamp(std, min=cfg.eps)
            group_adv = (group_rewards - mean) / denom
            advantages[mask] = torch.clamp(
                group_adv,
                -cfg.adv_clip_max,
                cfg.adv_clip_max,
            )
        return advantages

    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,
        old_log_probs: Any,
    ) -> tuple[Any, TrainStepMetrics]:
        del signals, advantages, old_log_probs
        raise RuntimeError(
            "DiffusionNFT does not consume evaluator log-prob signals. "
            "Use OnlineTrainer's DiffusionNFT batch/timestep loss path with "
            "latents_clean, prompt embeddings, timesteps, and video rewards.",
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

        cfg = self.config
        extras = batch.extras
        x0 = _required_tensor(extras, "latents_clean")
        prompt_embeds = _required_tensor(extras, "prompt_embeds")
        timesteps = _required_tensor(extras, "timesteps")
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
        if prepare is None:
            prepare = getattr(model, "nft_prepare_transformer_input", None)
        if not callable(prepare):
            raise RuntimeError(
                "DiffusionNFT model must expose diffusion_nft_prepare_transformer_input(...) "
                "or nft_prepare_transformer_input(...)",
            )

        t = _normalize_timesteps(t_raw, device=x0.device, dtype=x0.dtype)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = extras.get("diffusion_nft_noise")
        if noise is None:
            noise = torch.randn_like(x0.float())
        else:
            noise = noise.to(device=x0.device, dtype=torch.float32)
        xt = (1 - t_expanded) * x0.float() + t_expanded * noise

        transformer_inputs = prepare(
            latents=xt.to(x0.dtype),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=extras.get("prompt_attention_mask"),
            pooled_prompt_embeds=extras.get("pooled_prompt_embeds"),
            timestep=t_raw,
            num_frames=int(batch.context.get("num_frames", _infer_num_frames(x0))),
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
        positive_loss = _normalized_mse(positive_x0, x0_float)
        negative_loss = _normalized_mse(negative_x0, x0_float)

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


def _required_tensor(extras: dict[str, Any], key: str) -> Any:
    import torch

    value = extras.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"DiffusionNFT requires batch.extras[{key!r}] to be a tensor; "
            f"got {type(value).__name__}",
        )
    return value


def _normalize_timesteps(timesteps: Any, *, device: Any, dtype: Any) -> Any:
    import torch

    t = timesteps.to(device=device, dtype=torch.float32)
    if bool((t > 1.0).any()):
        t = t / 1000.0
    return t.to(dtype=dtype)


def _normalized_mse(prediction: Any, target: Any) -> Any:
    import torch

    reduce_dims = tuple(range(1, target.ndim))
    with torch.no_grad():
        weight = torch.abs(prediction.double() - target.double()).mean(
            dim=reduce_dims,
            keepdim=True,
        ).clip(min=1e-5)
    return ((prediction - target) ** 2 / weight).mean(dim=reduce_dims)


def _forward_previous_policy_adapter(transformer: Any, inputs: dict[str, Any]) -> Any:
    set_adapter = getattr(transformer, "set_adapter", None)
    if not callable(set_adapter):
        raise RuntimeError(
            "DiffusionNFT requires transformer.set_adapter('previous') "
            "for the previous-policy branch",
        )
    with _AdapterRestore(transformer):
        set_adapter("previous")
        import torch

        with torch.no_grad():
            return transformer(**inputs)[0].detach()


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
        with disable_adapter():
            with torch.no_grad():
                return transformer(**inputs)[0].detach()

    raise RuntimeError(
        "DiffusionNFT requires a reference branch via transformer.disable_adapters() "
        "or transformer.disable_adapter()",
    )


class _AdapterRestore:
    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        set_adapter = getattr(self.transformer, "set_adapter", None)
        if callable(set_adapter):
            set_adapter("default")


def _infer_num_frames(latents: Any) -> int:
    if getattr(latents, "ndim", 0) >= 3:
        return int(latents.shape[2])
    return 1


__all__ = ["DiffusionNFT", "DiffusionNFTConfig"]
