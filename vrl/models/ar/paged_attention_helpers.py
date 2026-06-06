"""Shared paged-attention helpers reused by AR model runners."""

from __future__ import annotations

from typing import Any

import torch

from vrl.nn.layers.attention.paged import ARAttentionBackend

__all__ = [
    "append_attention_token",
    "normalize_paged_last_hidden",
    "require_attention_backend",
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


def require_attention_backend(backend: ARAttentionBackend | None, *, family: str) -> ARAttentionBackend:
    """Return the attention backend, or raise if the runner has none.

    ``family`` names the model family (e.g. ``"Janus"``) for a clear error when the
    naive (no-backend) runner reaches a backend-only operation.
    """

    if backend is None:
        raise RuntimeError(f"{family} paged-attention path requires an attention backend")
    return backend


def normalize_paged_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
    """Squeeze a paged-attention ``[B, 1, H]`` last-hidden to ``[B, H]``.

    Accepts an already-``[B, H]`` tensor unchanged; rejects any other rank.
    """

    if last_hidden.ndim == 3:
        if last_hidden.shape[1] != 1:
            raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
        return last_hidden[:, 0, :]
    if last_hidden.ndim != 2:
        raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
    return last_hidden


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
