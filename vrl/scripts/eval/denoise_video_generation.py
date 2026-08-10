"""Shared deterministic video generation for denoise checkpoint evaluations."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.types import VideoGenerationRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob


def seed_for(
    *,
    base_seed: int,
    prompt_index: int,
    sample_index: int,
    samples_per_prompt: int,
) -> int:
    """Return one checkpoint-independent seed for a prompt/sample cell."""

    return int(base_seed) + int(prompt_index) * int(samples_per_prompt) + int(sample_index)


def generate_one_video(
    model: Any,
    *,
    prompt: str,
    seed: int,
    sampling: dict[str, Any],
) -> torch.Tensor:
    """Generate one video through the registered full-sequence model boundary."""

    encoded = model.encode_prompt(
        prompt,
        None,
        max_sequence_length=int(sampling["max_sequence_length"]),
        guidance_scale=float(sampling["guidance_scale"]),
    )
    request = VideoGenerationRequest(
        width=int(sampling["width"]),
        height=int(sampling["height"]),
        frame_count=int(sampling["num_frames"]),
        num_steps=int(sampling["num_steps"]),
        guidance_scale=float(sampling["guidance_scale"]),
        seed=int(seed),
        fps=int(sampling["fps"]),
    )
    state = model.prepare_sampling(request, encoded)
    generator = torch.Generator(device=state.latents.device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for step_idx, timestep in enumerate(state.timesteps):
            step_output = model.forward_step(state, step_idx)
            if str(sampling["denoise_mode"]) == "native":
                state.latents = state.scheduler.step(
                    step_output["noise_pred"].float(),
                    timestep,
                    state.latents.float(),
                    return_dict=False,
                )[0]
            else:
                state.latents = sde_step_with_logprob(
                    state.scheduler,
                    step_output["noise_pred"].float(),
                    timestep.unsqueeze(0),
                    state.latents.float(),
                    generator=generator,
                    deterministic=False,
                    return_dt=False,
                    noise_level=float(sampling["noise_level"]),
                    sde_type=str(sampling["sde_type"]),
                    step_index=step_idx,
                ).prev_sample
        decoded = model.decode_latents(state.latents)
    return video_to_cthw(decoded.detach().cpu())


def video_to_cthw(video: torch.Tensor) -> torch.Tensor:
    """Normalize a decoded video to channel-first ``[C,T,H,W]``."""

    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"expected one decoded video, got shape={tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"expected decoded video rank 4/5, got shape={tuple(video.shape)}")
    if video.shape[0] in {1, 3, 4}:
        return video[:3]
    if video.shape[1] in {1, 3, 4}:
        return video[:, :3].permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"cannot infer channel axis for decoded video shape={tuple(video.shape)}")


__all__ = ["generate_one_video", "seed_for", "video_to_cthw"]
