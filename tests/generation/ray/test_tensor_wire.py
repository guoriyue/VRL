"""Tensors returned by a generation actor cross the wire as byte views."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ray import cloudpickle
from ray.util.serialization import StandaloneSerializationContext

from vrl.generation.ray.tensor_wire import (
    TensorBytes,
    tensor_from_bytes,
    tensor_to_bytes,
)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        torch.randn(3, 5).to(torch.bfloat16),
        torch.tensor([True, False, True]),
        torch.tensor(7, dtype=torch.int64),
        torch.empty(0, 4, dtype=torch.float16),
    ],
    ids=["fp32", "bf16", "bool", "scalar", "empty"],
)
def test_round_trip_preserves_dtype_shape_and_values(tensor: torch.Tensor) -> None:
    payload = tensor_to_bytes(tensor)

    assert isinstance(payload, TensorBytes)
    assert payload.data.dtype == np.uint8
    assert payload.data.nbytes == tensor.numel() * tensor.element_size()
    rebuilt = tensor_from_bytes(payload)
    assert rebuilt.dtype == tensor.dtype
    assert rebuilt.shape == tensor.shape
    assert torch.equal(rebuilt, tensor)


def test_contiguous_tensor_is_viewed_not_copied() -> None:
    tensor = torch.randn(4, 8)

    payload = tensor_to_bytes(tensor)

    assert payload.data.ctypes.data == tensor.data_ptr()


def test_non_contiguous_view_round_trips_by_value() -> None:
    base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    window = base[:, 1:]

    rebuilt = tensor_from_bytes(tensor_to_bytes(window))

    assert torch.equal(rebuilt, window)
    assert rebuilt.is_contiguous()


def test_rebuilt_tensor_owns_its_memory() -> None:
    tensor = torch.ones(6, dtype=torch.float32)
    payload = tensor_to_bytes(tensor)
    frozen = TensorBytes(data=payload.data.copy(), dtype=payload.dtype, shape=payload.shape)
    frozen.data.flags.writeable = False

    rebuilt = tensor_from_bytes(frozen)
    rebuilt.mul_(2)

    assert torch.equal(rebuilt, torch.full((6,), 2.0))
    assert torch.equal(tensor, torch.ones(6))


def test_sparse_tensor_is_refused() -> None:
    sparse = torch.eye(3).to_sparse()

    with pytest.raises(TypeError, match="dense"):
        tensor_to_bytes(sparse)


def test_registered_serializer_ships_bytes_out_of_band() -> None:
    context = StandaloneSerializationContext()
    tensor = torch.randn(64, 64).to(torch.bfloat16)
    try:
        context._register_cloudpickle_serializer(torch.Tensor, tensor_to_bytes, tensor_from_bytes)
        buffers: list = []
        wire = cloudpickle.dumps({"latents": tensor}, protocol=5, buffer_callback=buffers.append)
    finally:
        context._unregister_cloudpickle_reducer(torch.Tensor)

    assert len(buffers) == 1
    assert buffers[0].raw().nbytes == tensor.numel() * tensor.element_size()
    assert len(wire) < 1024
    rebuilt = cloudpickle.loads(wire, buffers=buffers)["latents"]
    assert torch.equal(rebuilt, tensor)
    assert rebuilt.data_ptr() != tensor.data_ptr()
