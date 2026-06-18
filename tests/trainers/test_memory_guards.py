"""Tests for host-memory guard helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.ray.config import RayGenerationConfig
from vrl.utils.cuda_memory import cap_cuda_memory_fraction
from vrl.utils.memory import (
    HostMemorySnapshot,
    format_host_memory,
    validate_colocated_replay_memory,
)


def test_format_host_memory_omits_unknown_fields() -> None:
    """Checks format host memory omits unknown fields."""
    snapshot = HostMemorySnapshot(rss_mb=10.0, available_mb=None, total_mb=None)

    assert format_host_memory(snapshot) == "rss=10.0MiB"


def test_cap_cuda_memory_fraction_validates_range() -> None:
    """Checks the allocator cap rejects fractions outside (0, 1]."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            cap_cuda_memory_fraction(bad)


def test_cap_cuda_memory_fraction_none_is_noop() -> None:
    """Checks an unset cap is a no-op (dedicated-GPU / CPU worker)."""
    cap_cuda_memory_fraction(None)


def test_colocated_full_generation_bundle_can_fail_strict_guard() -> None:
    """Checks colocated full generation bundle can fail strict guard."""
    bundle = SimpleNamespace(
        metadata={
            "runtime_role": "full_generation_model",
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
