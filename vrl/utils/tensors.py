"""Layer-neutral tensor helpers shared by generation and the model families."""

from __future__ import annotations

import torch


def expand_tensor_to_batch(
    value: torch.Tensor,
    batch_size: int,
    *,
    materialize: bool = False,
) -> torch.Tensor:
    """Match the leading batch dimension, broadcasting only singleton inputs.

    Matching inputs are returned unchanged. On expansion, materialize=True
    allocates independent contiguous rows; otherwise return a shared view.
    """
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(
            f"cannot broadcast tensor batch={value.shape[0]} to batch_size={batch_size}"
        )
    expanded = value.expand(batch_size, *value.shape[1:])
    return expanded.clone(memory_format=torch.contiguous_format) if materialize else expanded


__all__ = ["expand_tensor_to_batch"]
