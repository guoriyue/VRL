"""Classifier-free guidance around one diffusion transformer call."""

from __future__ import annotations

from typing import Any, Literal

import torch


def predict_with_cfg(
    transformer: Any,
    cond: dict[str, Any],
    uncond: dict[str, Any] | None,
    *,
    guidance_scale: float,
    batched: bool,
    dtype: torch.dtype,
    normalize: bool = False,
) -> torch.Tensor:
    """Run the conditional transformer call and, given ``uncond``, guide it.

    ``cond`` and ``uncond`` are the transformer kwargs of each branch. With
    ``batched`` the branches run as one call: every tensor kwarg is
    concatenated uncond-first along the batch dim and every other kwarg must
    be shared. Without it each branch is its own call, for families whose
    branches differ in sequence length or non-tensor inputs. Each branch
    prediction is cast to ``dtype`` before the combine.
    """

    if uncond is None:
        return transformer(**cond, return_dict=False)[0].to(dtype)
    if batched:
        packed: dict[str, Any] = {}
        for key, value in cond.items():
            other = uncond[key]
            if isinstance(value, torch.Tensor):
                if value.shape != other.shape:
                    raise ValueError(
                        f"{key} branch shapes must match including batch dim: "
                        f"cond={tuple(value.shape)}, uncond={tuple(other.shape)}",
                    )
                packed[key] = torch.cat([other, value], dim=0)
            elif value is other or value == other:
                packed[key] = value
            else:
                raise ValueError(f"{key!r} cannot be batched because branch values differ")
        pred_uncond, pred_cond = transformer(**packed, return_dict=False)[0].chunk(2, dim=0)
    else:
        pred_cond = transformer(**cond, return_dict=False)[0]
        pred_uncond = transformer(**uncond, return_dict=False)[0]
    return combine_cfg(
        pred_cond.to(dtype),
        pred_uncond.to(dtype),
        guidance_scale=guidance_scale,
        normalize=normalize,
    )


def combine_cfg(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    *,
    guidance_scale: float,
    base: Literal["uncond", "cond"] = "uncond",
    normalize: bool = False,
) -> torch.Tensor:
    """Combine conditional and unconditional branch predictions.

    ``base="cond"`` is the Cosmos form ``cond + g * (cond - uncond)``.
    ``normalize`` reproduces the norm-preserving combine of the Lumina2 and
    Qwen-Image pipelines: after the linear blend, rescale the result back to
    the conditional branch's norm along the channel axis.
    """

    anchor = uncond if base == "uncond" else cond
    combined = anchor + float(guidance_scale) * (cond - uncond)
    if not normalize:
        return combined
    cond_norm = torch.norm(cond, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (cond_norm / combined_norm)
