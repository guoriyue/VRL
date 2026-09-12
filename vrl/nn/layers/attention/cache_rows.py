"""Row-wise cache helpers shared by AR decoder backends and schedulers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:  # pragma: no cover - plain tensor lanes work without transformers.
    Cache = None
    DynamicCache = None


@dataclass(slots=True)
class ARCacheRows:
    """Mutable per-row AR cache store."""

    rows: list[Any]
    owner: str = "ar_cache"

    def __post_init__(self) -> None:
        self.rows = list(self.rows)
        if not self.rows:
            raise ValueError(f"{self.owner} requires at least one cache row")

    @classmethod
    def from_batched(
        cls,
        value: Any,
        batch_size: int,
        *,
        owner: str = "ar_cache",
    ) -> ARCacheRows:
        """Create a row cache store from one batched cache value."""

        return cls(cls.split_batched(value, batch_size), owner=owner)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        return self.rows[index]

    def gather(self, indices: Sequence[int]) -> Any:
        """Return selected rows as one batched cache value."""

        return self.concatenate(self.select_rows(indices))

    def scatter(self, indices: Sequence[int], value: Any) -> None:
        """Overwrite selected rows from one batched cache value."""

        row_indices = self._validate_indices(indices)
        row_values = self.split_batched(value, len(row_indices))
        for index, row_value in zip(row_indices, row_values, strict=True):
            self.rows[index] = row_value

    def select_rows(self, indices: Sequence[int]) -> list[Any]:
        """Return selected row cache objects without batching them."""

        row_indices = self._validate_indices(indices)
        return [self.rows[index] for index in row_indices]

    def scatter_rows(self, indices: Sequence[int], rows: Sequence[Any]) -> None:
        """Overwrite selected rows from already-split row cache objects."""

        row_indices = self._validate_indices(indices)
        row_values = list(rows)
        if len(row_values) != len(row_indices):
            raise ValueError(
                f"{self.owner} received {len(row_values)} rows for {len(row_indices)} row indices",
            )
        for index, row_value in zip(row_indices, row_values, strict=True):
            self.rows[index] = row_value

    def _validate_indices(self, indices: Sequence[int]) -> list[int]:
        row_indices = list(indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in row_indices):
            raise ValueError(f"{self.owner} row indices must be integers")
        if not row_indices:
            raise ValueError(f"{self.owner} requires at least one row index")
        size = len(self.rows)
        invalid = [index for index in row_indices if index < 0 or index >= size]
        if invalid:
            raise IndexError(
                f"{self.owner} row indices out of range for {size} rows: {invalid}",
            )
        return row_indices

    @classmethod
    def split_batched(cls, value: Any, batch_size: int) -> list[Any]:
        """Split a batched AR cache/value into one-row values."""

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if Cache is not None and isinstance(value, Cache):
            return cls._split_hf_cache_rows(value, batch_size)
        return cls._split_plain_rows(value, batch_size)

    @classmethod
    def concatenate(cls, values: Sequence[Any]) -> Any:
        """Concatenate one-row AR caches along dim 0 without dtype promotion."""

        if not values:
            raise ValueError("values must be non-empty")
        first = values[0]
        if Cache is not None and isinstance(first, Cache):
            return cls._concat_hf_cache_rows(values)
        if any((Cache is not None and isinstance(value, Cache)) for value in values[1:]):
            raise TypeError("cannot concatenate mixed HF cache and non-cache rows")
        return cls._concat_plain_rows(values)

    @classmethod
    def _split_plain_rows(cls, value: Any, batch_size: int) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                raise ValueError(
                    "cannot split scalar AR tensor: expected a leading batch dimension"
                )
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"cannot split tensor with batch={value.shape[0]} into {batch_size} rows",
                )
            return [value[row : row + 1] for row in range(batch_size)]
        if isinstance(value, Mapping):
            split_items = {
                key: cls.split_batched(inner, batch_size) for key, inner in value.items()
            }
            return [
                type(value)((key, parts[row]) for key, parts in split_items.items())
                for row in range(batch_size)
            ]
        if isinstance(value, tuple):
            split_items = [cls.split_batched(inner, batch_size) for inner in value]
            return [tuple(parts[row] for parts in split_items) for row in range(batch_size)]
        if isinstance(value, list):
            split_items = [cls.split_batched(inner, batch_size) for inner in value]
            return [[parts[row] for parts in split_items] for row in range(batch_size)]
        return [value for _ in range(batch_size)]

    @classmethod
    def _concat_plain_rows(cls, values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            for index, value in enumerate(values[1:], start=1):
                if not isinstance(value, torch.Tensor):
                    raise TypeError("cannot concatenate mixed tensor and non-tensor AR rows")
                if value.dtype != first.dtype:
                    raise ValueError(
                        f"AR cache row {index} has dtype {value.dtype}, expected {first.dtype}"
                    )
            return torch.cat(list(values), dim=0)
        if isinstance(first, Mapping):
            if any(
                not isinstance(value, Mapping) or value.keys() != first.keys() for value in values
            ):
                raise ValueError("cannot concatenate AR mappings with different keys or row types")
            return type(first)(
                (key, cls.concatenate([value[key] for value in values])) for key in first
            )
        if isinstance(first, (tuple, list)):
            container_type = tuple if isinstance(first, tuple) else list
            if any(
                not isinstance(value, container_type) or len(value) != len(first)
                for value in values
            ):
                raise ValueError(
                    "cannot concatenate AR sequences with different lengths or row types"
                )
        if isinstance(first, tuple):
            return tuple(
                cls.concatenate([value[index] for value in values]) for index in range(len(first))
            )
        if isinstance(first, list):
            return [
                cls.concatenate([value[index] for value in values]) for index in range(len(first))
            ]
        if any(value != first for value in values[1:]):
            raise ValueError("cannot concatenate non-tensor AR values that differ")
        return first

    @staticmethod
    def _cache_kv_pairs(cache: Any) -> list[tuple[Any, Any]]:
        """(key, value) per layer, tolerant of the transformers 4/5 cache API.

        transformers 5 removed to_legacy_cache/from_legacy_cache; the cache now
        exposes ``.layers`` with per-layer ``keys``/``values`` tensors.
        """
        if hasattr(cache, "to_legacy_cache"):
            return [(key, val) for key, val in cache.to_legacy_cache()]
        return [(layer.keys, layer.values) for layer in cache.layers]

    @staticmethod
    def _cache_from_kv_pairs(pairs: Sequence[tuple[Any, Any]]) -> Any:
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(tuple(pairs))
        cache = DynamicCache()
        for layer_idx, (key, val) in enumerate(pairs):
            cache.update(key, val, layer_idx)
        return cache

    @classmethod
    def _split_hf_cache_rows(cls, value: Any, batch_size: int) -> list[Any]:
        if DynamicCache is None or not isinstance(value, DynamicCache):
            raise TypeError(
                "AR KV row scheduling currently supports transformers DynamicCache; "
                f"got {type(value).__name__}",
            )
        split_layers = [
            (cls.split_batched(key, batch_size), cls.split_batched(val, batch_size))
            for key, val in cls._cache_kv_pairs(value)
        ]
        return [
            cls._cache_from_kv_pairs([(keys[row], values[row]) for keys, values in split_layers])
            for row in range(batch_size)
        ]

    @classmethod
    def _concat_hf_cache_rows(cls, values: Sequence[Any]) -> Any:
        if DynamicCache is None or not all(isinstance(value, DynamicCache) for value in values):
            got = ", ".join(type(value).__name__ for value in values)
            raise TypeError(f"cannot concatenate mixed HF cache row types: {got}")
        kv_rows = [cls._cache_kv_pairs(value) for value in values]
        layer_count = len(kv_rows[0])
        if any(len(row) != layer_count for row in kv_rows[1:]):
            raise ValueError("cannot concatenate DynamicCache rows with different layer counts")
        return cls._cache_from_kv_pairs(
            [
                (
                    cls.concatenate([row[layer_idx][0] for row in kv_rows]),
                    cls.concatenate([row[layer_idx][1] for row in kv_rows]),
                )
                for layer_idx in range(layer_count)
            ],
        )


__all__ = ["ARCacheRows"]
