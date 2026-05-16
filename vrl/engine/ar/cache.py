"""AR per-row cache helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache


def ar_split_rows(value: Any, batch_size: int) -> list[Any]:
    """Split a batched AR cache/value into one-row values.

    HF ``DynamicCache`` stays as ``DynamicCache``. The real transformers
    forward path rejects legacy tuple caches, so cache objects must not be
    normalized into tuple form after prefill/decode.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if isinstance(value, Cache):
        return _split_hf_cache_rows(value, batch_size)
    return _split_plain_rows(value, batch_size)


def _split_plain_rows(value: Any, batch_size: int) -> list[Any]:
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
    first = values[0]
    if isinstance(first, Cache):
        return _concat_hf_cache_rows(values)
    if any(isinstance(value, Cache) for value in values[1:]):
        raise TypeError("cannot concatenate mixed HF cache and non-cache rows")
    return _concat_plain_rows(values)


def _concat_plain_rows(values: Sequence[Any]) -> Any:
    first = values[0]
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


def _split_hf_cache_rows(value: Cache, batch_size: int) -> list[Cache]:
    if not isinstance(value, DynamicCache):
        raise TypeError(
            "AR KV row scheduling currently supports transformers DynamicCache; "
            f"got {type(value).__name__}",
        )
    rows = value.batch_split(full_batch_size=batch_size, split_size=1)
    if len(rows) != batch_size:
        raise RuntimeError(
            f"DynamicCache.batch_split returned {len(rows)} rows for batch={batch_size}",
        )
    return rows


def _concat_hf_cache_rows(values: Sequence[Any]) -> DynamicCache:
    if not all(isinstance(value, DynamicCache) for value in values):
        got = ", ".join(type(value).__name__ for value in values)
        raise TypeError(f"cannot concatenate mixed HF cache row types: {got}")
    return DynamicCache.from_batch_splits(list(values))


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


__all__ = ["ar_concat_rows", "ar_split_rows"]
