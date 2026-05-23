"""Diffusion-specific generation metric helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vrl.trajectory import trajectory_tensor_bytes

if TYPE_CHECKING:
    from vrl.generation.types import GenerationOutput
    from vrl.trajectory import TrajectoryStoragePolicy


DIFFUSION_COUNTER_PREFIX = "diffusion_"


def diffusion_rollout_engine_counters(
    *,
    stage_durations: Mapping[str, float],
    num_denoise_steps: int,
    sample_batch_size: int,
    byte_values: Mapping[str, object],
) -> dict[str, Any]:
    """Build the diffusion rollout counter schema from final gathered tensors."""

    counters: dict[str, Any] = {
        "stage_durations_s": dict(stage_durations),
        "diffusion_num_denoise_steps": int(num_denoise_steps),
        "diffusion_sample_batch_size": int(sample_batch_size),
        "diffusion_storage_device": "preserve",
        "diffusion_storage_dtype": "preserve",
    }
    for name, value in byte_values.items():
        counters[f"diffusion_{name}_bytes"] = trajectory_tensor_bytes(value)
    return counters


def has_diffusion_counters(counters: Mapping[str, Any]) -> bool:
    """Return true when a metrics dict already carries diffusion counters."""

    return any(str(key).startswith(DIFFUSION_COUNTER_PREFIX) for key in counters)


def record_diffusion_storage_policy(
    output: GenerationOutput,
    policy: TrajectoryStoragePolicy,
) -> None:
    """Attach the consumed trajectory storage policy to diffusion counters."""

    metrics = output.metrics
    if metrics is None:
        return
    counters = metrics.engine_counters
    if not has_diffusion_counters(counters):
        return
    counters["diffusion_storage_device"] = policy.device
    counters["diffusion_storage_dtype"] = policy.dtype


__all__ = [
    "DIFFUSION_COUNTER_PREFIX",
    "diffusion_rollout_engine_counters",
    "has_diffusion_counters",
    "record_diffusion_storage_policy",
]
