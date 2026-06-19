"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
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
from vrl.models.interfaces import RuntimeBuildSpec


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


class _FakeCapability:
    trajectory_kind = "diffusion"
    supports_torch_compile = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "sd3_5",
            "task": "t2i",
            "trajectory_kind": self.trajectory_kind,
            "supports_torch_compile": self.supports_torch_compile,
        }


class _FakeArCapability(_FakeCapability):
    trajectory_kind = "ar_discrete"
    supports_torch_compile = False


def extract_fake_runtime_spec(cfg: Any, device: str, weight_dtype: str) -> RuntimeBuildSpec:
    del cfg
    return RuntimeBuildSpec(
        model_name_or_path="unit-test",
        device=device,
        dtype=weight_dtype,
        model_config={
            "marker": "driver-config",
            "torch_compile": {
                "enable": False,
                "mode": "default",
            },
        },
        sampling_config={
            "num_steps": 1,
        },
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


def _build_inputs_entry(capability: Any | None = None) -> Any:
    return SimpleNamespace(
        family="sd3_5",
        task="t2i",
        runtime_builder="tests.generation.ray.test_runtime_config:build_fake_runtime",
        executor_cls="tests.generation.ray.test_runtime_config:FakeExecutor",
        runtime_spec_extractor=(
            "tests.generation.ray.test_runtime_config:extract_fake_runtime_spec"
        ),
        gatherer=SimpleNamespace(
            import_path="tests.generation.ray.test_runtime_config:_FakeGatherer",
            kwargs={},
        ),
        capability=capability or _FakeCapability(),
        executor_kwargs=SimpleNamespace(
            include_sample_batch_size=False,
            include_reference_image=False,
        ),
    )


def _cfg(
    *,
    num_workers: int = 1,
    overlap: bool = False,
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
        # Release scheduling is derived from topology; nothing to spell here.
        "rollout": {},
    }
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
    colocate: float | None = None,
):
    rollout_runtime: dict[str, Any] = {"cpus_per_worker": 1}
    if colocate is not None:
        rollout_runtime["colocate"] = {
            "memory_fraction": colocate,
        }
    return OmegaConf.create(
        {
            "distributed": {
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
                "rollout": rollout_runtime,
            },
        },
    )


def _build_inputs_cfg(*, denoise_compile: dict[str, Any] | None = None) -> Any:
    cfg: dict[str, Any] = {
        "distributed": {
            "resources": {
                "visible_devices": [],
                "trainer": {
                    "num_gpus": 0,
                    "devices": [],
                },
                "rollout": {
                    "num_gpus": 0,
                    "devices": [],
                    "gpus_per_worker": 0,
                    "num_workers": 1,
                },
                "allow_overlap": False,
            },
        },
        "rollout": {},
    }
    if denoise_compile is not None:
        cfg["rollout"]["denoise_compile"] = denoise_compile
    return OmegaConf.create(cfg)


def test_chunk_placement_strategy_switches_from_cfg() -> None:
    """Checks distributed.rollout.chunk_placement_strategy flips the policy."""
    assert RayGenerationConfig.from_cfg(_cfg()).chunk_placement_strategy == "round_robin"

    cfg = _cfg()
    cfg.distributed.rollout.chunk_placement_strategy = "dynamic"
    dynamic = RayGenerationConfig.from_cfg(cfg)
    assert dynamic.chunk_placement_strategy == "dynamic"
    # Invalid values are now rejected at the typed schema boundary
    # (RolloutWorkerSection Literal) at parse time, not in RayGenerationConfig —
    # see tests/config/test_schema.py::test_unknown_chunk_placement_strategy_raises.


def test_sync_trainable_state_defaults_on_for_from_cfg() -> None:
    """Online runs train the policy the rollout workers must resync, so an omitted
    sync_trainable_state defaults ON (True), not silently False (which would train
    the rollout on stale policy weights). Explicit values are kept."""
    assert RayGenerationConfig.from_cfg(_cfg()).sync_trainable_state is True

    cfg = _cfg()
    cfg.distributed.rollout.sync_trainable_state = False
    assert RayGenerationConfig.from_cfg(cfg).sync_trainable_state is False


def test_ray_build_inputs_leaves_model_compile_config_without_rollout_override() -> None:
    """Checks build inputs keep model compile config when rollout override is absent."""
    launch_inputs = RayGenerationLauncher.build_inputs(
        _build_inputs_cfg(),
        _build_inputs_entry(),
        weight_dtype="bfloat16",
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["device"] == "cpu"
    assert model_build["dtype"] == "bfloat16"
    assert model_build["model_config"]["marker"] == "driver-config"
    assert model_build["model_config"]["torch_compile"] == {
        "enable": False,
        "mode": "default",
    }


def test_ray_build_inputs_applies_rollout_only_compile_override() -> None:
    """Checks rollout denoise compile only mutates the worker launch payload."""
    launch_inputs = RayGenerationLauncher.build_inputs(
        _build_inputs_cfg(
            denoise_compile={
                "enable": True,
                "mode": "reduce-overhead",
            },
        ),
        _build_inputs_entry(),
        weight_dtype="bfloat16",
    )

    model_config = launch_inputs.launch_contract.model_build["model_config"]
    assert model_config["marker"] == "driver-config"
    assert model_config["torch_compile"] == {
        "enable": True,
        "mode": "reduce-overhead",
    }


def test_ray_build_inputs_carries_gpu_memory_fraction_to_worker_contract() -> None:
    """Checks the colocate GPU budget reaches the worker contract."""
    cfg = _resource_cfg(
        trainer_devices=[0],
        rollout_devices=[0],
        allow_overlap=True,
        colocate=0.4,
    )

    launch_inputs = RayGenerationLauncher.build_inputs(
        cfg,
        _build_inputs_entry(),
        weight_dtype="bfloat16",
    )

    assert launch_inputs.launch_contract.extra["gpu_memory_fraction"] == 0.4


def test_ray_build_inputs_omits_gpu_memory_fraction_when_unset() -> None:
    """Checks no budget key is sent when the cap is unset (dedicated-GPU worker)."""
    launch_inputs = RayGenerationLauncher.build_inputs(
        _build_inputs_cfg(),
        _build_inputs_entry(),
        weight_dtype="bfloat16",
    )

    assert "gpu_memory_fraction" not in launch_inputs.launch_contract.extra


def test_ray_build_inputs_rejects_compile_on_family_without_capability() -> None:
    """Checks enabling rollout compile on a family that can't compile fails fast."""
    with pytest.raises(ValueError, match="does not support torch compile"):
        RayGenerationLauncher.build_inputs(
            _build_inputs_cfg(
                denoise_compile={
                    "enable": True,
                    "mode": "reduce-overhead",
                },
            ),
            _build_inputs_entry(_FakeArCapability()),
            weight_dtype="bfloat16",
        )


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
    """Checks Ray backend requires launch contract and gatherer."""
    with pytest.raises(ValueError, match="launch_contract plus gatherer"):
        RayGenerationLauncher().launch_from_cfg(
            _cfg(),
            launch_contract=launch_contract,
            gatherer=gatherer,
            driver_policy=_CpuPolicy(),
        )


def test_ray_backend_rejects_driver_cuda_policy_without_overlap() -> None:
    """Checks Ray backend rejects driver cuda policy without overlap."""
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
    """Checks Ray backend detects cuda trainable module when policy has no device."""
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
    """Checks Ray backend allows driver cuda policy with explicit overlap.

    A colocated single-GPU topology derives release-after-collect, so the driver
    CUDA policy overlapping the rollout GPU is allowed.
    """
    config = RayGenerationConfig.from_cfg(
        _resource_cfg(
            trainer_devices=[0],
            rollout_devices=[0],
            allow_overlap=True,
        ),
    ).validate_driver_state(driver_policy=_CudaPolicy())

    assert config.allow_driver_gpu_overlap is True
    assert config.resources.lifecycle.rollout.mode == "on_demand"


def test_ray_backend_allows_split_driver_cuda_when_devices_do_not_overlap() -> None:
    """Checks Ray backend allows split driver cuda when devices do not overlap."""
    config = RayGenerationConfig.from_cfg(
        _resource_cfg(trainer_devices=[0], rollout_devices=[1]),
    ).validate_driver_state(driver_policy=_CudaPolicy())

    assert config.resources is not None
    assert config.resources.trainer_devices == (0,)
    assert config.resources.rollout_devices == (1,)
    assert config.allow_driver_gpu_overlap is False


def test_ray_backend_colocate_keeps_worker_resident() -> None:
    """Checks colocate keeps the colocated rollout worker resident."""
    config = RayGenerationConfig.from_cfg(
        _resource_cfg(
            trainer_devices=[0],
            rollout_devices=[0],
            allow_overlap=True,
            colocate=0.45,
        ),
    ).validate_driver_state(driver_policy=_CudaPolicy())

    assert config.allow_driver_gpu_overlap is True
    assert config.resources.lifecycle.rollout.mode == "resident"
    assert config.resources.rollout_gpu_memory_fraction is not None
    assert config.resources.rollout_gpu_memory_fraction == 0.45
