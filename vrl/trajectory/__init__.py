"""Trajectory contract types for generation RL.

Lazy public boundary: ``types``/``storage``/``resolver`` are torch-free, but
``builders`` is not. ``vrl.config.schema`` reaches TrajectoryStoragePolicy while
validating every recipe, so an eager re-export here charged all config parsing
for the tensor builders. Deferring per symbol keeps both import paths honest.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.trajectory.builders import (
        build_ar_continuous_trajectory as build_ar_continuous_trajectory,
    )
    from vrl.trajectory.builders import (
        build_ar_discrete_trajectory as build_ar_discrete_trajectory,
    )
    from vrl.trajectory.builders import (
        build_ar_multisegment_trajectory as build_ar_multisegment_trajectory,
    )
    from vrl.trajectory.builders import (
        build_chunk_autoregressive_denoise_trajectory as build_chunk_autoregressive_denoise_trajectory,
    )
    from vrl.trajectory.builders import (
        build_chunk_autoregressive_generation_trajectory as build_chunk_autoregressive_generation_trajectory,
    )
    from vrl.trajectory.builders import build_diffusion_trajectory as build_diffusion_trajectory
    from vrl.trajectory.resolver import TrajectoryResolver as TrajectoryResolver
    from vrl.trajectory.resolver import TrajectoryResolverError as TrajectoryResolverError
    from vrl.trajectory.storage import TrajectoryStoragePolicy as TrajectoryStoragePolicy
    from vrl.trajectory.storage import (
        apply_trajectory_storage_policy as apply_trajectory_storage_policy,
    )
    from vrl.trajectory.storage import (
        trajectory_storage_policy_from_cfg as trajectory_storage_policy_from_cfg,
    )
    from vrl.trajectory.storage import trajectory_tensor_bytes as trajectory_tensor_bytes
    from vrl.trajectory.types import AxisKind as AxisKind
    from vrl.trajectory.types import DistributionKind as DistributionKind
    from vrl.trajectory.types import ReplayInput as ReplayInput
    from vrl.trajectory.types import SegmentModality as SegmentModality
    from vrl.trajectory.types import TensorRole as TensorRole
    from vrl.trajectory.types import TrajectoryAxis as TrajectoryAxis
    from vrl.trajectory.types import TrajectoryBatch as TrajectoryBatch
    from vrl.trajectory.types import TrajectoryMetrics as TrajectoryMetrics
    from vrl.trajectory.types import TrajectorySegment as TrajectorySegment
    from vrl.trajectory.types import TrajectoryTensor as TrajectoryTensor
    from vrl.trajectory.validation import TrajectoryValidationError as TrajectoryValidationError
    from vrl.trajectory.validation import TrajectoryValidator as TrajectoryValidator
    from vrl.trajectory.validation import tensor_ref as tensor_ref
    from vrl.trajectory.views import RewardView as RewardView
    from vrl.trajectory.views import named_tensor as named_tensor
    from vrl.trajectory.views import role_tensor as role_tensor

_PUBLIC_EXPORTS = {
    "AxisKind": ("vrl.trajectory.types", "AxisKind"),
    "DistributionKind": ("vrl.trajectory.types", "DistributionKind"),
    "ReplayInput": ("vrl.trajectory.types", "ReplayInput"),
    "RewardView": ("vrl.trajectory.views", "RewardView"),
    "SegmentModality": ("vrl.trajectory.types", "SegmentModality"),
    "TensorRole": ("vrl.trajectory.types", "TensorRole"),
    "TrajectoryAxis": ("vrl.trajectory.types", "TrajectoryAxis"),
    "TrajectoryBatch": ("vrl.trajectory.types", "TrajectoryBatch"),
    "TrajectoryMetrics": ("vrl.trajectory.types", "TrajectoryMetrics"),
    "TrajectoryResolver": ("vrl.trajectory.resolver", "TrajectoryResolver"),
    "TrajectoryResolverError": ("vrl.trajectory.resolver", "TrajectoryResolverError"),
    "TrajectorySegment": ("vrl.trajectory.types", "TrajectorySegment"),
    "TrajectoryStoragePolicy": ("vrl.trajectory.storage", "TrajectoryStoragePolicy"),
    "TrajectoryTensor": ("vrl.trajectory.types", "TrajectoryTensor"),
    "TrajectoryValidationError": ("vrl.trajectory.validation", "TrajectoryValidationError"),
    "TrajectoryValidator": ("vrl.trajectory.validation", "TrajectoryValidator"),
    "apply_trajectory_storage_policy": (
        "vrl.trajectory.storage",
        "apply_trajectory_storage_policy",
    ),
    "build_ar_continuous_trajectory": (
        "vrl.trajectory.builders",
        "build_ar_continuous_trajectory",
    ),
    "build_ar_discrete_trajectory": ("vrl.trajectory.builders", "build_ar_discrete_trajectory"),
    "build_ar_multisegment_trajectory": (
        "vrl.trajectory.builders",
        "build_ar_multisegment_trajectory",
    ),
    "build_chunk_autoregressive_denoise_trajectory": (
        "vrl.trajectory.builders",
        "build_chunk_autoregressive_denoise_trajectory",
    ),
    "build_chunk_autoregressive_generation_trajectory": (
        "vrl.trajectory.builders",
        "build_chunk_autoregressive_generation_trajectory",
    ),
    "build_diffusion_trajectory": ("vrl.trajectory.builders", "build_diffusion_trajectory"),
    "named_tensor": ("vrl.trajectory.views", "named_tensor"),
    "role_tensor": ("vrl.trajectory.views", "role_tensor"),
    "tensor_ref": ("vrl.trajectory.validation", "tensor_ref"),
    "trajectory_storage_policy_from_cfg": (
        "vrl.trajectory.storage",
        "trajectory_storage_policy_from_cfg",
    ),
    "trajectory_tensor_bytes": ("vrl.trajectory.storage", "trajectory_tensor_bytes"),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public trajectory symbol only when it is requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
