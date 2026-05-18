"""Small CPU diffusion fixtures for model parity tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


def fake_transformer_output(
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor | None = None,
    fps: int | None = None,
    condition_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    return_dict: bool = False,
) -> torch.Tensor:
    del padding_mask, return_dict
    output = hidden_states.float()
    step = timestep.float()
    while step.ndim < output.ndim:
        step = step.unsqueeze(-1)
    output = output + step
    output = output + _batch_scalar(encoder_hidden_states, output.ndim)
    if pooled_projections is not None:
        output = output + _batch_scalar(pooled_projections, output.ndim)
    if condition_mask is not None:
        output = output + condition_mask.float() * 0.125
    if fps is not None:
        output = output + float(fps) / 100.0
    return output.to(hidden_states.dtype)


class FakeDiffusionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=1)
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor]:
        self.calls.append(kwargs)
        return (fake_transformer_output(**kwargs),)


def _batch_scalar(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    value = tensor.float()
    while value.ndim > 1:
        value = value.mean(dim=-1)
    return value.view(value.shape[0], *([1] * (target_ndim - 1)))
