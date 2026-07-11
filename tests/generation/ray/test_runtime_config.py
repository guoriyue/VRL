"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tests.generation.ray.test_rollout_launcher import _Gatherer
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    RayGenerationConfig,
)
from vrl.generation.ray.launcher import (
    RayGenerationLauncher,
    _all_workers_support_versioned_slots,
)
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces import RuntimeBuildSpec


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


@dataclass
class _Bundle:
    model: Any
    trainable_modules: dict[str, Any]


def extract_fake_runtime_spec(cfg: Any, device: str, weight_dtype: str) -> RuntimeBuildSpec:
    model_node = OmegaConf.select(cfg, "model")
    model_config = (
        OmegaConf.to_container(model_node, resolve=True) if OmegaConf.is_config(model_node) else {}
    )
    return RuntimeBuildSpec(
        model_name_or_path="unit-test",
        device=device,
        dtype=weight_dtype,
        model_config=model_config,
        sampling_config={
            "num_steps": 1,
        },
    )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        runtime_builder=("tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"),
        executor_cls="tests.generation.ray.test_rollout_launcher:_TinyChunkExecutor",
    )


def _build_inputs_entry(capability: Any | None = None) -> Any:
    return SimpleNamespace(
        family="sd3_5",
        task="t2i",
        runtime_builder=("tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"),
        executor_cls="tests.generation.ray.test_rollout_launcher:_TinyChunkExecutor",
        runtime_spec_extractor=(
            "tests.generation.ray.test_runtime_config:extract_fake_runtime_spec"
        ),
        gatherer=SimpleNamespace(
            import_path="tests.generation.ray.test_rollout_launcher:_Gatherer",
            kwargs={},
        ),
        capability=capability or diffusion_family_capability("sd3_5", "t2i"),
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
    rollout_resource: dict[str, Any] = {
        "devices": rollout_devices,
        "gpus_per_worker": 1,
        "num_workers": len(rollout_devices),
    }
    # Resident colocation: the single authoritative grammar is gpu_pool=trainer +
    # memory_fraction on the resources rollout node (the legacy
    # distributed.rollout.colocate block was removed).
    if colocate is not None:
        rollout_resource["gpu_pool"] = "trainer"
        rollout_resource["memory_fraction"] = colocate
    return OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": sorted(set(trainer_devices) | set(rollout_devices)),
                    "trainer": {"devices": trainer_devices},
                    "rollout": rollout_resource,
                    "allow_overlap": allow_overlap,
                },
                "rollout": rollout_runtime,
            },
        },
    )


def _build_inputs_cfg(
    *,
    model_torch_compile: dict[str, Any] | None = None,
) -> Any:
    model_config = {
        "marker": "driver-config",
        "torch_compile": model_torch_compile
        or {
            "enable": False,
            "mode": "default",
        },
    }
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
        "model": model_config,
        "rollout": {},
    }
    return OmegaConf.create(cfg)


class _SlotWorker:
    """Real Ray actor exposing the versioned-slot capability query;
    ``supports=None`` raises like a dead/broken worker."""

    def __init__(self, supports: bool | None) -> None:
        self._supports = supports

    def supports_versioned_trainable_state(self) -> bool:
        if self._supports is None:
            raise RuntimeError("actor dead")
        return self._supports


def _slot_handles(ray: Any, *supports: bool | None) -> list[DistributedWorkerHandle]:
    actor_cls = ray.remote(num_cpus=0)(_SlotWorker)
    return [
        DistributedWorkerHandle(
            worker_id=f"w{index}",
            node_id="local",
            actor=actor_cls.remote(value),
        )
        for index, value in enumerate(supports)
    ]


@pytest.mark.slow_test
def test_runtime_capability_is_and_over_all_workers(local_ray) -> None:
    """supports_non_draining_weight_sync derives as the AND of every worker's
    supports_versioned_trainable_state(): all True -> True; any False -> False."""
    weight_sync = object()

    assert (
        _all_workers_support_versioned_slots(
            local_ray, _slot_handles(local_ray, True, True), weight_sync=weight_sync
        )
        is True
    )
    assert (
        _all_workers_support_versioned_slots(
            local_ray, _slot_handles(local_ray, True, False), weight_sync=weight_sync
        )
        is False
    )


@pytest.mark.slow_test
def test_runtime_capability_false_without_weight_sync_or_workers(local_ray) -> None:
    """No weight sync (sync_trainable_state off) or no workers -> safe draining
    barrier (False), never a silent True."""
    assert (
        _all_workers_support_versioned_slots(
            local_ray, _slot_handles(local_ray, True, True), weight_sync=None
        )
        is False
    )
    assert _all_workers_support_versioned_slots(local_ray, [], weight_sync=object()) is False


@pytest.mark.slow_test
def test_runtime_capability_false_when_a_worker_query_raises(local_ray) -> None:
    """A failed capability query (real ray.get raising RayTaskError) must fall
    back to the safe draining barrier, not crash the launch or optimistically
    assume support."""
    assert (
        _all_workers_support_versioned_slots(
            local_ray,
            _slot_handles(local_ray, True, None),
            weight_sync=object(),
        )
        is False
    )


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


def test_pipelined_switches_from_cfg() -> None:
    """Checks distributed.rollout.pipelined flips the per-request pipelined path."""
    assert RayGenerationConfig.from_cfg(_cfg()).pipelined is False

    cfg = _cfg()
    cfg.distributed.rollout.pipelined = True

    assert RayGenerationConfig.from_cfg(cfg).pipelined is True


def test_sync_trainable_state_defaults_on_for_from_cfg() -> None:
    """Online runs train the policy the rollout workers must resync, so an omitted
    sync_trainable_state defaults ON (True), not silently False (which would train
    the rollout on stale policy weights). Explicit values are kept."""
    assert RayGenerationConfig.from_cfg(_cfg()).sync_trainable_state is True

    cfg = _cfg()
    cfg.distributed.rollout.sync_trainable_state = False
    assert RayGenerationConfig.from_cfg(cfg).sync_trainable_state is False


def test_ray_build_inputs_uses_model_compile_config_as_single_source() -> None:
    """Checks build inputs carry model.torch_compile without rollout overrides."""
    launch_inputs = RayGenerationLauncher.build_inputs(
        _build_inputs_cfg(
            model_torch_compile={
                "enable": True,
                "mode": "default",
            },
        ),
        _build_inputs_entry(),
        weight_dtype="bfloat16",
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["device"] == "cpu"
    assert model_build["dtype"] == "bfloat16"
    assert model_build["model_config"]["marker"] == "driver-config"
    assert model_build["model_config"]["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_ray_build_inputs_preserves_disabled_model_compile_config() -> None:
    """Checks disabled model.torch_compile is preserved as ordinary model config."""
    launch_inputs = RayGenerationLauncher.build_inputs(
        _build_inputs_cfg(),
        _build_inputs_entry(),
        weight_dtype="bfloat16",
    )

    model_config = launch_inputs.launch_contract.model_build["model_config"]
    assert model_config["marker"] == "driver-config"
    assert model_config["torch_compile"] == {
        "enable": False,
        "mode": "default",
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


def test_ray_build_inputs_rejects_model_compile_on_family_without_capability() -> None:
    """Checks model.torch_compile fails fast on rollout families that cannot compile."""
    with pytest.raises(ValueError, match="does not support torch compile"):
        RayGenerationLauncher.build_inputs(
            _build_inputs_cfg(
                model_torch_compile={
                    "enable": True,
                    "mode": "default",
                },
            ),
            _build_inputs_entry(ar_discrete_family_capability("janus_pro", "ar_t2i")),
            weight_dtype="bfloat16",
        )


@pytest.mark.parametrize(
    ("launch_contract", "gatherer"),
    [
        pytest.param(None, _Gatherer(), id="missing-launch-contract"),
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


def test_launcher_default_ray_init_is_owned_local() -> None:
    assert RayGenerationLauncher().ray_init_kwargs == {"address": "local"}


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


@pytest.mark.gpu
def test_ray_backend_detects_cuda_trainable_module_when_policy_has_no_device() -> None:
    """Checks Ray backend detects cuda trainable module when policy has no device."""
    bundle = _Bundle(
        model=object(),
        trainable_modules={"transformer": torch.nn.Linear(1, 1).to("cuda:0")},
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

    A colocated single-GPU topology derives on-demand rollout activation, so the
    driver CUDA policy overlapping the rollout GPU is allowed.
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
