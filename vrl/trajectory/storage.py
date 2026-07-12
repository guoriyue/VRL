"""Runtime storage policy helpers for trajectory tensor payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from vrl.trajectory.types import TrajectoryBatch
from vrl.utils.config import cfg_get, to_builtin

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


def trajectory_storage_policy_from_cfg(value: object) -> TrajectoryStoragePolicy:
    """Parse rollout.trajectory_storage config into a typed storage policy."""

    if value is None:
        return TrajectoryStoragePolicy()
    value = to_builtin(value)
    if isinstance(value, TrajectoryStoragePolicy):
        return value
    if isinstance(value, Mapping):
        return TrajectoryStoragePolicy(
            device=str(value.get("device", "preserve")),
            dtype=str(value.get("dtype", "preserve")),
        )
    device = cfg_get(value, "device", "preserve")
    dtype = cfg_get(value, "dtype", "preserve")
    return TrajectoryStoragePolicy(device=str(device), dtype=str(dtype))


def apply_trajectory_storage_policy(
    batch: TrajectoryBatch,
    policy: TrajectoryStoragePolicy,
) -> TrajectoryBatch:
    """Apply placement/dtype policy to trajectory tensor leaves."""

    if policy == TrajectoryStoragePolicy():
        return batch

    batch.group_ids = _apply_value_policy(batch.group_ids, policy)
    for segment in batch.segments.values():
        for tensor in segment.tensors.values():
            tensor.value = _apply_value_policy(tensor.value, policy)
    batch.metrics.values = _apply_value_policy(batch.metrics.values, policy)
    return batch


def apply_value_storage_policy(value: Any, policy: TrajectoryStoragePolicy) -> Any:
    """Apply the placement/dtype policy to one tensor-like value tree.

    The leaf semantics are identical to ``apply_trajectory_storage_policy``;
    this entry point exists for callers that hold raw tensors/dicts (e.g. the
    generation worker applying the policy BEFORE tensors cross the
    worker->driver wire, where downcasting actually saves transfer bytes).
    Re-applying the same policy driver-side is a no-op.
    """

    if policy == TrajectoryStoragePolicy():
        return value
    return _apply_value_policy(value, policy)


def trajectory_tensor_bytes(value: object) -> int:
    """Return an estimated byte count for tensor-like leaves in ``value``."""

    return _tensor_bytes(value, seen=set())


def _apply_value_policy(value: Any, policy: TrajectoryStoragePolicy) -> Any:
    if _is_torch_tensor(value):
        kwargs: dict[str, Any] = {}
        if policy.device == "cpu":
            kwargs["device"] = "cpu"
        dtype = _torch_dtype(policy.dtype)
        if dtype is not None and value.is_floating_point():
            kwargs["dtype"] = dtype
        if not kwargs:
            return value
        return value.to(**kwargs)
    if isinstance(value, dict):
        return {key: _apply_value_policy(inner, policy) for key, inner in value.items()}
    if isinstance(value, list):
        return [_apply_value_policy(inner, policy) for inner in value]
    if isinstance(value, tuple):
        return tuple(_apply_value_policy(inner, policy) for inner in value)
    return value


def _tensor_bytes(value: object, *, seen: set[int]) -> int:
    if value is None:
        return 0
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    if isinstance(value, TrajectoryBatch):
        total = _tensor_bytes(value.group_ids, seen=seen)
        for segment in value.segments.values():
            for tensor in segment.tensors.values():
                total += _tensor_bytes(tensor.value, seen=seen)
        total += _tensor_bytes(value.metrics.values, seen=seen)
        return total

    if _is_torch_tensor(value):
        return int(value.numel()) * int(value.element_size())
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
    "apply_trajectory_storage_policy",
    "apply_value_storage_policy",
    "trajectory_storage_policy_from_cfg",
    "trajectory_tensor_bytes",
]
