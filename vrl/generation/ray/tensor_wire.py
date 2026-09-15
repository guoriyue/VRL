"""Zero-copy tensor serialization for results leaving a generation actor.

Ray pickles a CPU ``torch.Tensor`` through its storage, which serializes the
bytes in-band: the actor copies the whole trajectory into a bytes object, Ray
copies that into the object store, and the driver copies it out again while
unpickling. For a multi-gigabyte trajectory that is seconds of actor time per
request during which the GPU sits idle. A NumPy array, by contrast, travels as a
pickle-5 out-of-band buffer: the actor hands Ray a view of the existing buffer,
and the driver reads it straight from the object store.

The serializer registered here ships every tensor as a ``uint8`` view of its
bytes plus dtype and shape, so bf16 and other dtypes NumPy lacks round-trip
unchanged. The receiving side makes one owned copy: object-store memory is
shared and read-only, and the trainer must be free to keep or mutate what it
received without pinning the object or corrupting other readers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TensorBytes:
    """A dense tensor as raw bytes; dtype and shape restore it on the far side."""

    data: np.ndarray
    dtype: str
    shape: tuple[int, ...]


def tensor_to_bytes(tensor: torch.Tensor) -> TensorBytes:
    """View a dense CPU tensor as bytes without copying when it is contiguous."""

    import torch

    if tensor.layout != torch.strided:
        raise TypeError(f"only dense tensors cross the actor boundary, got layout {tensor.layout}")
    dense = tensor.detach()
    if dense.is_cuda:
        dense = dense.cpu()
    if not dense.is_contiguous():
        dense = dense.contiguous()
    data = dense.reshape(-1).view(torch.uint8).numpy()
    return TensorBytes(
        data=data, dtype=str(dense.dtype).removeprefix("torch."), shape=tuple(dense.shape)
    )


def tensor_from_bytes(payload: TensorBytes) -> torch.Tensor:
    """Rebuild the tensor into memory the receiver owns."""

    import torch

    dtype = getattr(torch, payload.dtype)
    if payload.data.size == 0:
        return torch.empty(payload.shape, dtype=dtype)
    flat = torch.from_numpy(payload.data.copy())
    return flat.view(dtype).reshape(payload.shape)


def register_tensor_wire_serializer() -> None:
    """Route every tensor this process pickles for Ray through the byte view.

    Call once per generation actor process. The deserializer is a plain module
    function, so the driver needs no registration of its own.
    """

    import torch
    from ray.util.serialization import register_serializer

    register_serializer(torch.Tensor, serializer=tensor_to_bytes, deserializer=tensor_from_bytes)


__all__ = [
    "TensorBytes",
    "register_tensor_wire_serializer",
    "tensor_from_bytes",
    "tensor_to_bytes",
]
