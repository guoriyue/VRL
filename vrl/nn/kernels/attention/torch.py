"""Torch reference attention kernel for parity and debug paths."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class TorchAttentionKernel:
    """Small scaled dot-product attention reference implementation."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must have shape [B, T, H]")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query and key batch sizes must match")
        scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        mask = None
        if causal:
            q_len = query.shape[1]
            k_len = key.shape[1]
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril()
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=scale,
        )


__all__ = ["TorchAttentionKernel"]

