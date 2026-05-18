"""Torch SDPA attention kernel wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TorchSDPAAttentionKernel:
    """Thin wrapper over PyTorch scaled dot-product attention."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        if query.ndim not in (3, 4):
            raise ValueError("query must have shape [B, T, D] or [B, H, T, D]")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query and key batch sizes must match")
        kwargs: dict[str, object] = {
            "attn_mask": attention_mask,
            "dropout_p": float(dropout_p),
            "is_causal": bool(causal),
        }
        if scale is not None:
            kwargs["scale"] = float(scale)
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


__all__ = ["TorchSDPAAttentionKernel"]
