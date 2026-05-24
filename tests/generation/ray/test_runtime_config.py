"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from omegaconf import OmegaConf

from vrl.generation import GenerationOutput
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    RayGenerationConfig,
)
from vrl.generation.ray.launcher import RayGenerationLauncher


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


class _FakeParameter:
    def __init__(self, device: str) -> None:
        self.device = device


class _FakeModule:
    def __init__(self, device: str) -> None:
        self._parameters = [_FakeParameter(device)]

    def parameters(self) -> list[_FakeParameter]:
        return self._parameters


@dataclass
class _Bundle:
    model: Any
    trainable_modules: dict[str, Any]


class _FakeGatherer:
    def gather_chunks(self, request: Any, sample_rows: Any, chunks: Any) -> GenerationOutput:
        del sample_rows, chunks
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=[],
            output=None,
        )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        runtime_builder=(
            "tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"
        ),
        executor_cls="tests.generation.ray.test_rollout_launcher:_TinyChunkExecutor",
    )


def _cfg(
    *,
    backend: str | None = "ray",
    num_workers: int = 1,
    overlap: bool = False,
    release_after_collect: bool = False,
):
    rollout_devices = [0] if overlap else [1]
    visible_devices = [0] if overlap else [0, 1]
    distributed = {
        "resources": {
            "visible_devices": visible_devices,
            "trainer": {"devices": [0]},
            "rollout": {
                "devices": rollout_devices,
                "gpus_per_worker": 1,
                "num_workers": num_workers,
            },
            "allow_overlap": overlap,
        },
        "rollout": {
            "release_after_collect": release_after_collect,
        },
    }
    if backend is not None:
        distributed["backend"] = backend
    return OmegaConf.create(
        {
            "distributed": distributed,
        },
    )


def _resource_cfg(
    *,
    trainer_devices: list[int],
    rollout_devices: list[int],
    allow_overlap: bool = False,
    release_after_collect: bool = False,
):
    return OmegaConf.create(
        {
            "distributed": {
                "backend": "ray",
                "resources": {
                    "visible_devices": sorted(set(trainer_devices) | set(rollout_devices)),
                    "trainer": {"devices": trainer_devices},
                    "rollout": {
                        "devices": rollout_devices,
                        "gpus_per_worker": 1,
                        "num_workers": len(rollout_devices),
                    },
                    "allow_overlap": allow_overlap,
                },
                "rollout": {
                    "cpus_per_worker": 1,
                    "release_after_collect": release_after_collect,
                },
            },
        },
    )


def test_rollout_backend_config_from_cfg_uses_ray_by_default() -> None:
    config = RayGenerationConfig.from_cfg(_cfg(backend=None))

    assert config.resources is not None
    assert config.num_workers == 1


@pytest.mark.parametrize(
    ("launch_contract", "gatherer"),
    [
        pytest.param(None, _FakeGatherer(), id="missing-launch-contract"),
        pytest.param(_launch_contract(), None, id="missing-gatherer"),
        pytest.param(None, None, id="missing-both"),
    ],
)
def test_ray_backend_requires_launch_contract_and_gatherer(
    launch_contract: Any,
    gatherer: Any,
) -> None:
    with pytest.raises(ValueError, match="launch_contract plus gatherer"):
        RayGenerationLauncher().launch_from_cfg(
            _cfg(),
            launch_contract=launch_contract,
            gatherer=gatherer,
            driver_policy=_CpuPolicy(),
        )


def test_ray_backend_rejects_driver_cuda_policy_without_overlap() -> None:
    with pytest.raises(ValueError, match=r"resources\.allow_overlap=false"):
        RayGenerationConfig.from_cfg(
            _resource_cfg(
                trainer_devices=[0],
                rollout_devices=[0],
                allow_overlap=False,
            ),
        ).validate_driver_state(driver_policy=_CudaPolicy())

    assert "overlaps rollout devices" in DRIVER_CUDA_OWNERSHIP_ERROR


def test_ray_backend_detects_cuda_trainable_module_when_policy_has_no_device() -> None:
    bundle = _Bundle(
        model=object(),
        trainable_modules={"transformer": _FakeModule("cuda:0")},
    )

    with pytest.raises(ValueError, match=r"resources\.allow_overlap=false"):
        RayGenerationConfig.from_cfg(
            _resource_cfg(
                trainer_devices=[0],
                rollout_devices=[0],
                allow_overlap=False,
            ),
        ).validate_driver_state(driver_bundle=bundle)


def test_ray_backend_allows_driver_cuda_policy_with_explicit_overlap() -> None:
    config = RayGenerationConfig.from_cfg(
        _resource_cfg(
            trainer_devices=[0],
            rollout_devices=[0],
            allow_overlap=True,
            release_after_collect=True,
        ),
    ).validate_driver_state(driver_policy=_CudaPolicy())

    assert config.allow_driver_gpu_overlap is True


def test_ray_backend_allows_split_driver_cuda_when_devices_do_not_overlap() -> None:
    config = RayGenerationConfig.from_cfg(
        _resource_cfg(trainer_devices=[0], rollout_devices=[1]),
    ).validate_driver_state(driver_policy=_CudaPolicy())

    assert config.resources is not None
    assert config.resources.trainer_devices == (0,)
    assert config.resources.rollout_devices == (1,)
    assert config.allow_driver_gpu_overlap is False


def test_ray_backend_overlap_requires_release_after_collect() -> None:
    with pytest.raises(ValueError, match="release_after_collect=false"):
        RayGenerationConfig.from_cfg(
            _resource_cfg(
                trainer_devices=[0],
                rollout_devices=[0],
                allow_overlap=True,
                release_after_collect=False,
            ),
        ).validate_driver_state(driver_policy=_CudaPolicy())
