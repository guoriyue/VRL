"""Diffusion model timestep shape helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def expand_batch_timestep(timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return timestep as a batch vector without changing dtype/device."""

    if timestep.ndim == 0:
        return timestep.expand(batch_size)
    if timestep.shape[0] != batch_size:
        raise ValueError(
            f"timestep batch dim must be {batch_size}, got {tuple(timestep.shape)}",
        )
    return timestep


def pack_eval_timestep(timesteps: torch.Tensor, step_idx: int) -> torch.Tensor:
    """Pack per-sample eval timesteps so forward_step can index step 0."""

    timestep = timesteps[:, step_idx] if timesteps.ndim > 1 else timesteps
    return timestep.unsqueeze(0) if timestep.ndim == 1 else timestep


def broadcast_spatial_timestep(
    value: torch.Tensor,
    *,
    batch_size: int,
    frames: int,
) -> torch.Tensor:
    """Broadcast a scalar timestep to ``[B, 1, T, 1, 1]``."""

    return value.view(1, 1, 1, 1, 1).expand(batch_size, -1, frames, -1, -1)


def set_mu_shifted_timesteps(
    scheduler: Any,
    *,
    num_steps: int,
    image_seq_len: int,
    device: Any,
    calculate_shift: Callable[..., float],
) -> Any:
    """Set scheduler timesteps with the resolution-derived ``mu`` (diffusers parity).

    ``calculate_shift`` is passed in by the family so each keeps its own
    diffusers provenance and lazy import at the call site
    (``flux.pipeline_flux`` vs ``qwenimage.pipeline_qwenimage`` — byte-identical
    upstream, but the family owns the import). Returns ``scheduler.timesteps``.
    """

    config = scheduler.config
    if getattr(config, "use_dynamic_shifting", False):
        mu = calculate_shift(
            image_seq_len,
            getattr(config, "base_image_seq_len", 256),
            getattr(config, "max_image_seq_len", 4096),
            getattr(config, "base_shift", 0.5),
            getattr(config, "max_shift", 1.15),
        )
        scheduler.set_timesteps(num_steps, device=device, mu=mu)
    else:
        scheduler.set_timesteps(num_steps, device=device)
    return scheduler.timesteps
