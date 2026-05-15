"""AR per-row cache helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def ar_split_rows(value: Any, batch_size: int) -> list[Any]:
    """Split a batched AR cache/value into one-row values.

    HF-style ``past_key_values`` are nested tuples whose tensors carry batch as
    dim 0. Splitting to per-row caches avoids invalid scatter when partial AR
    scheduling lets rows reach different sequence lengths.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    value = _to_tuple_cache_if_needed(value)
    if _is_tensor(value):
        if value.shape[0] != batch_size:
            raise ValueError(
                f"cannot split tensor with batch={value.shape[0]} into "
                f"{batch_size} rows",
            )
        return [value[row : row + 1] for row in range(batch_size)]
    if isinstance(value, Mapping):
        split_items = {
            key: ar_split_rows(inner, batch_size) for key, inner in value.items()
        }
        return [
            type(value)((key, parts[row]) for key, parts in split_items.items())
            for row in range(batch_size)
        ]
    if isinstance(value, tuple):
        split_items = [ar_split_rows(inner, batch_size) for inner in value]
        return [
            tuple(parts[row] for parts in split_items)
            for row in range(batch_size)
        ]
    if isinstance(value, list):
        split_items = [ar_split_rows(inner, batch_size) for inner in value]
        return [
            [parts[row] for parts in split_items]
            for row in range(batch_size)
        ]
    return [value for _ in range(batch_size)]


def ar_concat_rows(values: Sequence[Any]) -> Any:
    """Concatenate one-row AR cache/value objects along batch dim 0."""

    if not values:
        raise ValueError("values must be non-empty")
    first = _to_tuple_cache_if_needed(values[0])
    rest = [_to_tuple_cache_if_needed(value) for value in values[1:]]
    values = [first, *rest]
    if _is_tensor(first):
        return torch.cat(list(values), dim=0)
    if isinstance(first, Mapping):
        return type(first)(
            (key, ar_concat_rows([value[key] for value in values]))
            for key in first
        )
    if isinstance(first, tuple):
        return tuple(
            ar_concat_rows([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            ar_concat_rows([value[index] for value in values])
            for index in range(len(first))
        ]
    if any(value != first for value in values[1:]):
        raise ValueError("cannot concatenate non-tensor AR values that differ")
    return first


def _to_tuple_cache_if_needed(value: Any) -> Any:
    to_legacy_cache = getattr(value, "to_legacy_cache", None)
    if callable(to_legacy_cache):
        return to_legacy_cache()
    return value


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


__all__ = ["ar_concat_rows", "ar_split_rows"]
