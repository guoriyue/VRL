"""Trajectory contract types for generation RL.

Lazy public boundary: ``types``/``storage``/``reader`` are torch-free, but
``builders`` is not. ``vrl.config.schema`` reaches TrajectoryStoragePolicy while
validating every recipe, so an eager re-export here charged all config parsing
for the tensor builders. Deferring per symbol keeps both import paths honest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrl.utils.config import install_lazy_exports

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
    from vrl.trajectory.reader import TrajectoryReader as TrajectoryReader
    from vrl.trajectory.reader import TrajectoryReaderError as TrajectoryReaderError
    from vrl.trajectory.storage import TrajectoryStoragePolicy as TrajectoryStoragePolicy
    from vrl.trajectory.storage import trajectory_tensor_bytes as trajectory_tensor_bytes
    from vrl.trajectory.types import AxisKind as AxisKind
    from vrl.trajectory.types import DistributionKind as DistributionKind
    from vrl.trajectory.types import ReplayInput as ReplayInput
    from vrl.trajectory.types import SegmentModality as SegmentModality
    from vrl.trajectory.types import TensorRole as TensorRole
    from vrl.trajectory.types import TrajectoryAxis as TrajectoryAxis
    from vrl.trajectory.types import TrajectoryBatch as TrajectoryBatch
    from vrl.trajectory.types import TrajectorySegment as TrajectorySegment
    from vrl.trajectory.types import TrajectoryTensor as TrajectoryTensor
    from vrl.trajectory.validation import TrajectoryValidationError as TrajectoryValidationError
    from vrl.trajectory.validation import TrajectoryValidator as TrajectoryValidator
    from vrl.trajectory.validation import tensor_ref as tensor_ref
    from vrl.trajectory.views import RewardInputSpec as RewardInputSpec

_PUBLIC_EXPORTS = {
    "AxisKind": "vrl.trajectory.types",
    "DistributionKind": "vrl.trajectory.types",
    "ReplayInput": "vrl.trajectory.types",
    "RewardInputSpec": "vrl.trajectory.views",
    "SegmentModality": "vrl.trajectory.types",
    "TensorRole": "vrl.trajectory.types",
    "TrajectoryAxis": "vrl.trajectory.types",
    "TrajectoryBatch": "vrl.trajectory.types",
    "TrajectoryReader": "vrl.trajectory.reader",
    "TrajectoryReaderError": "vrl.trajectory.reader",
    "TrajectorySegment": "vrl.trajectory.types",
    "TrajectoryStoragePolicy": "vrl.trajectory.storage",
    "TrajectoryTensor": "vrl.trajectory.types",
    "TrajectoryValidationError": "vrl.trajectory.validation",
    "TrajectoryValidator": "vrl.trajectory.validation",
    "build_ar_continuous_trajectory": "vrl.trajectory.builders",
    "build_ar_discrete_trajectory": "vrl.trajectory.builders",
    "build_ar_multisegment_trajectory": "vrl.trajectory.builders",
    "build_chunk_autoregressive_denoise_trajectory": "vrl.trajectory.builders",
    "build_chunk_autoregressive_generation_trajectory": "vrl.trajectory.builders",
    "build_diffusion_trajectory": "vrl.trajectory.builders",
    "tensor_ref": "vrl.trajectory.validation",
    "trajectory_tensor_bytes": "vrl.trajectory.storage",
}

install_lazy_exports(globals(), _PUBLIC_EXPORTS)
