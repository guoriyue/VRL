"""Classifier-free guidance tensor packing for diffusion model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

DiffusionCFGBase = Literal["uncond", "cond"]


@dataclass(slots=True)
class DiffusionBranch:
    """One diffusion transformer branch call."""

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_transformer_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_states": self.hidden_states,
            "timestep": self.timestep,
            "encoder_hidden_states": self.encoder_hidden_states,
            **self.extra_kwargs,
            "return_dict": False,
        }


def pack_batched_cfg(
    *,
    cond: DiffusionBranch,
    uncond: DiffusionBranch,
) -> DiffusionBranch:
    """Pack uncond+cond branches for a single batched CFG transformer call.

    Uncond rows precede cond rows in the packed batch dimension; the batched
    branch carries no ``metadata`` (its default empty mapping).
    """

    _validate_pair("hidden_states", cond.hidden_states, uncond.hidden_states)
    _validate_pair("timestep", cond.timestep, uncond.timestep)
    _validate_pair(
        "encoder_hidden_states",
        cond.encoder_hidden_states,
        uncond.encoder_hidden_states,
    )
    return DiffusionBranch(
        hidden_states=torch.cat([uncond.hidden_states, cond.hidden_states], dim=0),
        timestep=torch.cat([uncond.timestep, cond.timestep], dim=0),
        encoder_hidden_states=torch.cat(
            [uncond.encoder_hidden_states, cond.encoder_hidden_states],
            dim=0,
        ),
        extra_kwargs=_pack_extra_kwargs(cond.extra_kwargs, uncond.extra_kwargs),
    )


def split_batched_cfg_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split batched CFG output into ``(uncond, cond)`` branches."""

    if output.shape[0] % 2 != 0:
        raise ValueError(
            f"batched CFG output batch dimension must be even, got {output.shape[0]}",
        )
    uncond, cond = output.chunk(2, dim=0)
    return uncond, cond


def combine_cfg(
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    *,
    guidance_scale: float,
    do_cfg: bool,
    base: DiffusionCFGBase = "uncond",
) -> torch.Tensor:
    """Combine conditional and unconditional branch predictions."""

    if not do_cfg:
        return cond
    if uncond is None:
        raise ValueError("uncond branch is required when do_cfg=True")
    if base == "uncond":
        return uncond + float(guidance_scale) * (cond - uncond)
    if base == "cond":
        return cond + float(guidance_scale) * (cond - uncond)
    raise ValueError(f"unsupported CFG base: {base!r}")


def _validate_pair(name: str, cond: torch.Tensor, uncond: torch.Tensor) -> None:
    if cond.shape[1:] != uncond.shape[1:]:
        raise ValueError(
            f"{name} branch shapes must match after batch dim: "
            f"cond={tuple(cond.shape)}, uncond={tuple(uncond.shape)}",
        )


def _pack_extra_kwargs(
    cond: dict[str, Any],
    uncond: dict[str, Any],
) -> dict[str, Any]:
    if cond.keys() != uncond.keys():
        raise ValueError(
            "cond/uncond extra kwargs must have identical keys for batched CFG: "
            f"cond={sorted(cond)}, uncond={sorted(uncond)}",
        )
    packed: dict[str, Any] = {}
    for key in cond:
        cond_value = cond[key]
        uncond_value = uncond[key]
        if isinstance(cond_value, torch.Tensor) and isinstance(uncond_value, torch.Tensor):
            _validate_pair(key, cond_value, uncond_value)
            packed[key] = torch.cat([uncond_value, cond_value], dim=0)
        elif cond_value is uncond_value or cond_value == uncond_value:
            packed[key] = cond_value
        else:
            raise ValueError(
                f"extra kwarg {key!r} cannot be batched because branch values differ",
            )
    return packed
