"""DiffusionNFT tensor losses shared by diffusion algorithms."""

from __future__ import annotations

import torch


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-sample MSE normalized by detached mean absolute error."""

    reduce_dims = tuple(range(1, target.ndim))
    with torch.no_grad():
        weight = torch.abs(prediction.double() - target.double()).mean(
            dim=reduce_dims,
            keepdim=True,
        ).clip(min=1e-5)
    return ((prediction - target) ** 2 / weight).mean(dim=reduce_dims)


__all__ = ["normalized_mse"]
