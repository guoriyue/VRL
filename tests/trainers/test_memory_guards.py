"""Tests for host-memory guard helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrl.generation.ray.config import (
    RayGenerationConfig,
    validate_colocated_replay_memory,
)
from vrl.ray.resources import resolve_distributed_resources
from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.memory import HostMemorySnapshot, format_host_memory


def _ray_config(*, colocated: bool) -> RayGenerationConfig:
    rollout = (
        {
            "gpu_pool": "trainer",
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
        if colocated
        else {
            "devices": [1],
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
    )
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0] if colocated else [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": rollout,
                },
                "rollout": {},
            },
        },
    )
    return RayGenerationConfig.from_cfg(
        cfg,
        resources=resolve_distributed_resources(cfg),
    )


def test_format_host_memory_omits_unknown_fields() -> None:
    """Checks format host memory omits unknown fields."""
    snapshot = HostMemorySnapshot(rss_mb=10.0, available_mb=None, total_mb=None)

    assert format_host_memory(snapshot) == "rss=10.0MiB"


def test_cuda_oom_detection_prefers_the_typed_exception() -> None:
    """A typed CUDA OOM remains detectable even if its message format changes."""
    import torch

    assert is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("allocation failed"))


def test_colocated_full_generation_bundle_can_fail_strict_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks colocated full generation bundle can fail strict guard."""
    monkeypatch.setenv("VRL_STRICT_REPLAY_MEMORY_GUARD", "1")
    bundle = SimpleNamespace(
        metadata={
            "loads_full_generation_modules": True,
        },
    )
    config = _ray_config(colocated=True)

    with pytest.raises(ValueError, match="loads_full_generation_modules=true"):
        validate_colocated_replay_memory(
            bundle=bundle,
            rollout_config=config,
        )


def test_non_colocated_full_generation_bundle_passes_memory_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks non colocated full generation bundle passes memory guard."""
    monkeypatch.setenv("VRL_STRICT_REPLAY_MEMORY_GUARD", "1")
    bundle = SimpleNamespace(metadata={"loads_full_generation_modules": True})
    config = _ray_config(colocated=False)

    validate_colocated_replay_memory(
        bundle=bundle,
        rollout_config=config,
    )
