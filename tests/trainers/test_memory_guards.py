"""Tests for host-memory guard helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.ray.config import (
    RayGenerationConfig,
    validate_colocated_replay_memory,
)
from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.memory import HostMemorySnapshot, format_host_memory


def test_format_host_memory_omits_unknown_fields() -> None:
    """Checks format host memory omits unknown fields."""
    snapshot = HostMemorySnapshot(rss_mb=10.0, available_mb=None, total_mb=None)

    assert format_host_memory(snapshot) == "rss=10.0MiB"


def test_cuda_oom_detection_prefers_the_typed_exception() -> None:
    """A typed CUDA OOM remains detectable even if its message format changes."""
    import torch

    assert is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("allocation failed"))


def test_colocated_full_generation_bundle_can_fail_strict_guard() -> None:
    """Checks colocated full generation bundle can fail strict guard."""
    bundle = SimpleNamespace(
        metadata={
            "loads_full_generation_modules": True,
        },
    )
    config = RayGenerationConfig(
        num_workers=1,
        gpus_per_worker=1.0,
        cpus_per_worker=1.0,
        allow_driver_gpu_overlap=True,
    )

    with pytest.raises(ValueError, match="loads_full_generation_modules=true"):
        validate_colocated_replay_memory(
            bundle=bundle,
            rollout_config=config,
            strict=True,
        )


def test_non_colocated_full_generation_bundle_passes_memory_guard() -> None:
    """Checks non colocated full generation bundle passes memory guard."""
    bundle = SimpleNamespace(metadata={"loads_full_generation_modules": True})
    config = RayGenerationConfig(
        num_workers=1,
        gpus_per_worker=1.0,
        cpus_per_worker=1.0,
        allow_driver_gpu_overlap=False,
    )

    validate_colocated_replay_memory(
        bundle=bundle,
        rollout_config=config,
        strict=True,
    )
