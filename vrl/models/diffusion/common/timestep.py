"""Diffusion model timestep shape helpers."""

from __future__ import annotations

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
