"""Runtime storage policy helpers for trajectory tensor payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, get_args

from PIL.Image import Image

from vrl.trajectory.device import map_tensor_tree
from vrl.trajectory.types import TrajectoryBatch
from vrl.utils.config import to_builtin_deep

TrajectoryStorageDevice = Literal["preserve", "cpu"]
TrajectoryStorageDType = Literal["preserve", "float32", "float16", "bfloat16"]

# Derived from the Literals above so validation tracks the single source of
# truth: adding a member to the Literal updates these (and the error messages
# that print ``sorted(...)`` of them) automatically.
_VALID_DEVICES = frozenset(get_args(TrajectoryStorageDevice))
_VALID_DTYPES = frozenset(get_args(TrajectoryStorageDType))


@dataclass(frozen=True, slots=True)
class TrajectoryStoragePolicy:
    """Runtime-only placement and dtype policy for trajectory tensor leaves."""

    device: TrajectoryStorageDevice = "preserve"
    dtype: TrajectoryStorageDType = "preserve"

    def __post_init__(self) -> None:
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                "trajectory storage device must be one of "
                f"{sorted(_VALID_DEVICES)}, got {self.device!r}",
            )
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                "trajectory storage dtype must be one of "
                f"{sorted(_VALID_DTYPES)}, got {self.dtype!r}",
            )

    @classmethod
    def from_config(cls, value: object) -> TrajectoryStoragePolicy:
        """Parse rollout.trajectory_storage config into a typed storage policy."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        value = to_builtin_deep(value)
        if isinstance(value, Mapping):
            return cls(
                device=str(value.get("device", "preserve")),
                dtype=str(value.get("dtype", "preserve")),
            )
        # A non-None, non-mapping value is a misconfiguration (e.g. a bare
        # ``trajectory_storage: cpu`` string surviving config resolution). Fail
        # loudly instead of silently degrading to a default no-op policy.
        raise TypeError(
            f"rollout.trajectory_storage must be a mapping with 'device'/'dtype' keys, got {value!r}",
        )

    def apply_to_trajectory_(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        """Replace trajectory tensor leaves in place and return the same batch."""
        if self == TrajectoryStoragePolicy():
            return batch
        for segment in batch.segments.values():
            for tensor in segment.tensors.values():
                tensor.value = self.apply_to_value(tensor.value)
        return batch

    def apply_to_value(self, value: Any) -> Any:
        """Return a converted tensor tree; preserve the input for a no-op policy.

        Generation applies this before worker-to-driver transfer to reduce wire
        bytes. Only floating tensors are cast; integer ids retain their dtype.
        """
        if self == TrajectoryStoragePolicy():
            return value

        def _place(tensor: Any) -> Any:
            kwargs: dict[str, Any] = {}
            if self.device == "cpu":
                kwargs["device"] = "cpu"
            dtype = _torch_dtype(self.dtype)
            if dtype is not None and tensor.is_floating_point():
                kwargs["dtype"] = dtype
            if not kwargs:
                return tensor
            return tensor.to(**kwargs)

        return map_tensor_tree(value, _place, is_leaf=_is_torch_tensor)


def trajectory_tensor_bytes(value: object) -> int:
    """Return an estimated byte count for tensor-like leaves in ``value``."""

    return _tensor_bytes(value, seen=set())


def _tensor_bytes(value: object, *, seen: set[int]) -> int:
    if value is None:
        return 0
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    if isinstance(value, TrajectoryBatch):
        total = 0
        for segment in value.segments.values():
            for tensor in segment.tensors.values():
                total += _tensor_bytes(tensor.value, seen=seen)
        total += _tensor_bytes(value.context, seen=seen)
        for segment in value.segments.values():
            total += _tensor_bytes(segment.metadata, seen=seen)
        for view in value.reward_views.values():
            total += _tensor_bytes(view.metadata, seen=seen)
        return total

    if _is_torch_tensor(value):
        return int(value.numel()) * int(value.element_size())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_tensor_bytes(getattr(value, item.name), seen=seen) for item in fields(value))
    if isinstance(value, Image):
        # Pillow commonly retains four-byte internal pixels even for RGB.
        # Count decoded storage without allocating an image.tobytes() copy.
        return value.width * value.height * max(4, len(value.getbands()))
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(inner, seen=seen) for inner in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(inner, seen=seen) for inner in value)
    return 0


def _torch_dtype(name: TrajectoryStorageDType) -> Any | None:
    if name == "preserve":
        return None

    from vrl.models.dtypes import resolve_torch_dtype

    return resolve_torch_dtype(name)


def _is_torch_tensor(value: object) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency.
        return False
    return isinstance(value, torch.Tensor)


__all__ = [
    "TrajectoryStoragePolicy",
    "trajectory_tensor_bytes",
]
