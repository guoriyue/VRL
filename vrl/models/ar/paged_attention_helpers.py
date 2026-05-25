"""Shared paged-attention helpers reused by AR model runners."""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "append_attention_token",
    "scatter_paged_states",
    "select_paged_states",
]


def append_attention_token(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            attention_mask,
            torch.ones(
                attention_mask.shape[0],
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=1,
    )


def select_paged_states(
    states: list[Any] | None,
    row_indices: list[int],
) -> list[Any]:
    if states is None:
        raise RuntimeError("paged attention state is not initialized")
    return [states[index] for index in row_indices]


def scatter_paged_states(
    states: list[Any] | None,
    row_indices: list[int],
    values: list[Any],
) -> None:
    if states is None:
        raise RuntimeError("paged attention state is not initialized")
    if len(row_indices) != len(values):
        raise ValueError("paged attention state updates must match row indices")
    for row_index, value in zip(row_indices, values, strict=True):
        states[row_index] = value
