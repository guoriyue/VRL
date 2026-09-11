"""Bounded wire chunks with complete host-side assembly before model installation.

This bounds individual transport objects, not source or receiver model-state RAM.
The receiver never exposes an incomplete mapping to a model's existing loader.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch

from vrl.models.dtypes import resolve_torch_dtype
from vrl.utils.config import require_exact_int


def weight_manifest(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not state:
        raise ValueError("weight transfer requires a nonempty state")
    manifest = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("weight transfer keys must be nonempty strings")
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.is_quantized
            or hasattr(value, "placements")
        ):
            raise TypeError(f"{name}: weight transfer requires plain, dense CPU tensors")
        manifest[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        resolve_torch_dtype(str(value.dtype))
    return manifest


def iter_weight_chunks(
    state: Mapping[str, torch.Tensor], bucket_bytes: int
) -> Iterator[tuple[str, int, torch.Tensor]]:
    """Clone each slice so serialization cannot retain the whole backing storage."""

    if type(bucket_bytes) is not int or bucket_bytes < 1:
        raise ValueError("bucket_bytes must be a positive integer")
    for name, value in state.items():
        per_chunk = bucket_bytes // value.element_size()
        if per_chunk < 1:
            raise ValueError(f"bucket is smaller than one element of {name}")
        flat = value.detach().contiguous().reshape(-1)
        for offset in range(0, flat.numel(), per_chunk):
            yield name, offset, flat[offset : offset + per_chunk].clone()


class StagedWeightTransfer:
    """Own receiver buffers and reject gaps, duplicates and mixed transfer IDs."""

    def __init__(self, transfer_id: str, policy_version: int, manifest: dict[str, Any]) -> None:
        if not transfer_id or not manifest:
            raise ValueError("staged weight transfer requires an ID and manifest")
        self.transfer_id = transfer_id
        self.policy_version = require_exact_int(policy_version, path="policy_version", minimum=0)
        self._specs = {}
        self._offsets = {}
        self._buffers: dict[str, torch.Tensor] = {}
        for name, spec in manifest.items():
            shape = spec["shape"]
            if (
                not isinstance(name, str)
                or not name
                or any(type(n) is not int or n < 0 for n in shape)
            ):
                raise ValueError("invalid weight transfer manifest")
            self._specs[name] = (
                tuple(shape),
                resolve_torch_dtype(spec["dtype"]),
                math.prod(shape),
            )
            self._offsets[name] = 0

    def require_id(self, transfer_id: str) -> None:
        if transfer_id != self.transfer_id:
            raise ValueError("weight chunk belongs to another transfer")

    def receive(self, transfer_id: str, chunk: tuple[str, int, torch.Tensor]) -> None:
        self.require_id(transfer_id)
        name, offset, value = chunk
        if name not in self._specs:
            raise ValueError(f"unknown weight transfer key: {name}")
        shape, dtype, count = self._specs[name]
        if (
            type(offset) is not int
            or offset != self._offsets[name]
            or not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.device.type != "cpu"
            or value.dtype != dtype
            or value.layout != torch.strided
            or value.is_quantized
            or hasattr(value, "placements")
            or value.numel() == 0
            or offset + value.numel() > count
        ):
            raise ValueError(f"invalid or out-of-order chunk for {name}")
        if name not in self._buffers:
            self._buffers[name] = torch.empty(shape, dtype=dtype, device="cpu")
        # Own the copy: keeping Ray's deserialized view would pin every bucket
        # in the object store and defeat the bounded wire-object lifetime.
        self._buffers[name].reshape(-1)[offset : offset + value.numel()].copy_(value.detach())
        self._offsets[name] += value.numel()

    def finish(self, transfer_id: str) -> dict[str, torch.Tensor]:
        self.require_id(transfer_id)
        missing = [
            name for name, (_, _, count) in self._specs.items() if self._offsets[name] != count
        ]
        if missing:
            raise ValueError(f"incomplete weight transfer: {missing[:5]}")
        for name, (shape, dtype, count) in self._specs.items():
            if count == 0:
                self._buffers[name] = torch.empty(shape, dtype=dtype, device="cpu")
        return self._buffers


def iter_weight_buckets(
    state: Mapping[str, torch.Tensor], bucket_bytes: int
) -> Iterator[list[tuple[str, int, torch.Tensor]]]:
    """Pack small tensor chunks together while retaining the tensor-byte ceiling."""

    bucket = []
    size = 0
    for chunk in iter_weight_chunks(state, bucket_bytes):
        chunk_size = chunk[2].numel() * chunk[2].element_size()
        if bucket and size + chunk_size > bucket_bytes:
            yield bucket
            bucket, size = [], 0
        bucket.append(chunk)
        size += chunk_size
    if bucket:
        yield bucket
