"""Tests for host-memory guard helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.runtime.config import GenerationRuntimeConfig
from vrl.utils.memory import (
    HostMemorySnapshot,
    format_host_memory,
    validate_colocated_replay_memory,
)


def test_format_host_memory_omits_unknown_fields() -> None:
    snapshot = HostMemorySnapshot(rss_mb=10.0, available_mb=None, total_mb=None)

    assert format_host_memory(snapshot) == "rss=10.0MiB"


def test_colocated_full_generation_bundle_can_fail_strict_guard() -> None:
    bundle = SimpleNamespace(
        metadata={
            "runtime_role": "full_generation_model",
            "loads_full_generation_modules": True,
        },
    )
    config = GenerationRuntimeConfig(
        backend="ray",
        num_workers=1,
        gpus_per_worker=1.0,
        cpus_per_worker=1.0,
        allow_driver_gpu_overlap=True,
        release_after_collect=True,
    )

    with pytest.raises(ValueError, match="loads_full_generation_modules=true"):
        validate_colocated_replay_memory(
            bundle=bundle,
            rollout_config=config,
            strict=True,
        )


def test_non_colocated_full_generation_bundle_passes_memory_guard() -> None:
    bundle = SimpleNamespace(metadata={"loads_full_generation_modules": True})
    config = GenerationRuntimeConfig(
        backend="ray",
        num_workers=1,
        gpus_per_worker=1.0,
        cpus_per_worker=1.0,
        allow_driver_gpu_overlap=False,
        release_after_collect=False,
    )

    validate_colocated_replay_memory(
        bundle=bundle,
        rollout_config=config,
        strict=True,
    )
