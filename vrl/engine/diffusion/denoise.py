"""Shared fused-stage helpers for diffusion family executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.engine.core.protocols import PipelineChunkResult
from vrl.math.diffusion.flow_matching import sde_step_with_logprob


@dataclass(frozen=True, slots=True)
class DiffusionDenoiseConfig:
    """Runtime knobs for one diffusion micro-batch denoise loop."""

    prompt_index: int
    sample_start: int
    sample_count: int
    seed: int | None
    same_latent: bool
    sde_window: tuple[int, int] | None
    return_kl: bool
    noise_level: float = 1.0
    sde_type: str = "sde"


@dataclass(slots=True)
class DiffusionChunkResult(PipelineChunkResult):
    """Output of one fused diffusion micro-batch."""

    prompt_index: int
    sample_start: int
    sample_count: int
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    video: Any
    replay_tensors: dict[str, Any]
    context: dict[str, Any]
    peak_memory_mb: float | None = None


def run_diffusion_denoise_chunk(
    *,
    model: Any,
    request: Any,
    encoded: dict[str, Any],
    config: DiffusionDenoiseConfig,
    prepare_kwargs: dict[str, Any] | None = None,
) -> DiffusionChunkResult:
    """Run one fused diffusion micro-batch: prepare -> denoise -> decode."""

    from vrl.trainers.profiling import record_function

    with record_function("engine.cache_write"):
        state = model.prepare_sampling(request, encoded, **(prepare_kwargs or {}))
    chunk_batch = state.latents.shape[0]
    if int(chunk_batch) != config.sample_count:
        raise ValueError(
            "Diffusion denoise chunk produced "
            f"{chunk_batch} rows, expected {config.sample_count}",
        )
    device = state.latents.device
    generator = _build_generator(
        device=device,
        sample_start=config.sample_start,
        seed=config.seed,
        same_latent=config.same_latent,
    )

    obs_steps: list[Any] = []
    act_steps: list[Any] = []
    lp_steps: list[Any] = []
    kl_steps: list[Any] = []
    t_steps: list[Any] = []

    prompt_embeds = encoded.get("prompt_embeds")
    transformer_dtype = (
        prompt_embeds.dtype if isinstance(prompt_embeds, torch.Tensor) else state.latents.dtype
    )
    autocast_ctx = _autocast_for_dtype(state.latents.device, transformer_dtype)

    with autocast_ctx, torch.no_grad():
        for step_idx in range(len(state.timesteps)):
            with record_function("engine.denoise_step"):
                latents_ori = state.latents.clone()
                timestep = state.timesteps[step_idx]
                with record_function("engine.cache_read"):
                    step_output = model.forward_step(state, step_idx)
                noise_pred = step_output["noise_pred"]

                in_sde_window = config.sde_window is None or (
                    config.sde_window[0] <= step_idx < config.sde_window[1]
                )
                sde_result = sde_step_with_logprob(
                    state.scheduler,
                    noise_pred.float(),
                    timestep.unsqueeze(0),
                    state.latents.float(),
                    generator=generator if in_sde_window else None,
                    deterministic=not in_sde_window,
                    return_dt=config.return_kl,
                    noise_level=config.noise_level,
                    sde_type=config.sde_type,
                )
                prev_latents = sde_result.prev_sample
                with record_function("engine.cache_write"):
                    state.latents = prev_latents

            obs_steps.append(latents_ori.detach())
            act_steps.append(prev_latents.detach())
            lp_steps.append(sde_result.log_prob.detach())
            t_steps.append(timestep.detach())
            if config.return_kl:
                kl_steps.append(sde_result.log_prob.detach().abs())
            else:
                kl_steps.append(torch.zeros(chunk_batch, device=device))

    observations = torch.stack(obs_steps, dim=1)
    actions = torch.stack(act_steps, dim=1)
    log_probs = torch.stack(lp_steps, dim=1)
    timesteps = torch.stack(
        [timestep.expand(chunk_batch) for timestep in t_steps],
        dim=1,
    )
    kl = torch.stack(kl_steps, dim=1)
    with record_function("engine.vq_decode"):
        video = model.decode_latents(state.latents)

    return DiffusionChunkResult(
        prompt_index=config.prompt_index,
        sample_start=config.sample_start,
        sample_count=config.sample_count,
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        timesteps=timesteps,
        kl=kl,
        video=video,
        replay_tensors=model.export_replay_tensors(state),
        context=model.export_batch_context(state),
        peak_memory_mb=_peak_memory_mb(),
    )


def _peak_memory_mb() -> float | None:
    """Return CUDA peak memory if available."""

    if not torch.cuda.is_available():
        return None
    try:
        peak_bytes = torch.cuda.max_memory_allocated()
    except Exception:
        return None
    return peak_bytes / (1024 * 1024)


def _build_generator(
    *,
    device: Any,
    sample_start: int,
    seed: int | None,
    same_latent: bool,
) -> torch.Generator | None:
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + sample_start)
        return generator
    if same_latent:
        raise ValueError("same_latent=True requires an explicit sampling.seed")
    return None


def _autocast_for_dtype(device: Any, dtype: Any) -> Any:
    """Use autocast only for CUDA half-precision rollout forwards."""

    if getattr(device, "type", None) != "cuda":
        return _NullCtx()
    if dtype not in (torch.float16, torch.bfloat16):
        return _NullCtx()
    return torch.amp.autocast("cuda", dtype=dtype)


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


__all__ = [
    "DiffusionChunkResult",
    "DiffusionDenoiseConfig",
    "run_diffusion_denoise_chunk",
]
