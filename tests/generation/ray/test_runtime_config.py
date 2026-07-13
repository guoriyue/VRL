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
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.utils import all_workers_support_versioned_slots
from vrl.models.interfaces import ModelBuild, RolloutBuildOptions


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


@dataclass
class _Bundle:
    model: Any
    trainable_modules: dict[str, Any]


def resolve_fake_model_build(cfg: Any, device: str) -> ModelBuild:
    model_node = OmegaConf.select(cfg, "model")
    model_config = (
        OmegaConf.to_container(model_node, resolve=True) if OmegaConf.is_config(model_node) else {}
    )
    return ModelBuild(
        model_name_or_path="unit-test",
        device=device,
        parameter_dtype=torch.bfloat16,
        family=str(model_config["family"]),
        model_config=model_config,
        sampling_config={
            "num_steps": 1,
        },
        # The fake extractor deliberately uses a different autocast dtype from
        # parameter storage so this fixture verifies both cross Ray intact.
        rollout=RolloutBuildOptions(
            autocast_dtype=torch.float32,
            prompt_encoder_dtype=torch.float16,
        ),
    )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        task="ar_t2i",
        generation_kind="ar",
        runtime_builder=("tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"),
        executor_cls="tests.generation.ray.test_rollout_launcher:_TinyChunkExecutor",
    )


def test_launch_contract_rejects_live_object_key_through_schema() -> None:
    payload = _launch_contract().to_dict()
    payload["executor"] = object()

    with pytest.raises(ValueError, match=r"unsupported.*executor"):
        GenerationRuntimeLaunchContract.from_dict(payload)


def test_launch_contract_accepts_primitive_config_leaves() -> None:
    contract = GenerationRuntimeLaunchContract(
        family="unit",
        task="test",
        generation_kind="ar",
        runtime_builder="tests.fake:build",
        executor_cls="tests.fake:Executor",
        model_config={
            "text": "value",
            "integer": 1,
            "number": 1.5,
            "enabled": True,
            "optional": None,
        },
    )

    assert contract.model_config["enabled"] is True
    assert contract.model_config["optional"] is None


def test_launch_contract_rejects_callable_config_leaf() -> None:
    with pytest.raises(TypeError, match="callable"):
        GenerationRuntimeLaunchContract(
            family="unit",
            task="test",
            generation_kind="ar",
            runtime_builder="tests.fake:build",
            executor_cls="tests.fake:Executor",
            executor_kwargs={"factory": lambda: None},
        )


def test_launch_contract_rejects_unknown_generation_kind() -> None:
    with pytest.raises(ValueError, match="unsupported generation_kind"):
        GenerationRuntimeLaunchContract(
            family="unit",
            task="test",
            generation_kind="video",  # type: ignore[arg-type]
            runtime_builder="tests.fake:build",
            executor_cls="tests.fake:Executor",
        )


def _build_inputs_entry(collector_kind: str = "diffusion") -> Any:
    return SimpleNamespace(
        family="sd3_5",
        task="t2i",
        runtime_builder=("tests.generation.ray.test_rollout_launcher:build_tiny_runtime_bundle"),
        executor_cls="tests.generation.ray.test_rollout_launcher:_TinyChunkExecutor",
        model_build_resolver=("tests.generation.ray.test_runtime_config:resolve_fake_model_build"),
        gatherer=SimpleNamespace(
            import_path="tests.generation.ray.test_rollout_launcher:_Gatherer",
            kwargs={},
        ),
        collector=SimpleNamespace(kind=collector_kind),
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
):
    rollout_runtime: dict[str, Any] = {"cpus_per_worker": 1}
    rollout_resource: dict[str, Any] = {
        "devices": rollout_devices,
        "gpus_per_worker": 1,
        "num_workers": len(rollout_devices),
    }
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
        "family": "sd3_5",
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
        all_workers_support_versioned_slots(
            local_ray, _slot_handles(local_ray, True, True), weight_sync=weight_sync
        )
        is True
    )
    assert (
        all_workers_support_versioned_slots(
            local_ray, _slot_handles(local_ray, True, False), weight_sync=weight_sync
        )
        is False
    )


@pytest.mark.slow_test
def test_runtime_capability_false_without_weight_sync_or_workers(local_ray) -> None:
    """No weight sync (sync_trainable_state off) or no workers -> safe draining
    barrier (False), never a silent True."""
    assert (
        all_workers_support_versioned_slots(
            local_ray, _slot_handles(local_ray, True, True), weight_sync=None
        )
        is False
    )
    assert all_workers_support_versioned_slots(local_ray, [], weight_sync=object()) is False


@pytest.mark.slow_test
def test_runtime_capability_false_when_a_worker_query_raises(local_ray) -> None:
    """A failed capability query (real ray.get raising RayTaskError) must fall
    back to the safe draining barrier, not crash the launch or optimistically
    assume support."""
    assert (
        all_workers_support_versioned_slots(
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
    )

    model_build = launch_inputs.launch_contract.model_build
    assert launch_inputs.launch_contract.generation_kind == "diffusion"
    assert model_build["device"] == "cpu"
    assert model_build["parameter_dtype"] == "bfloat16"
    assert model_build["rollout"]["autocast_dtype"] == "float32"
    assert model_build["rollout"]["prompt_encoder_dtype"] == "float16"
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
    )

    model_config = launch_inputs.launch_contract.model_build["model_config"]
    assert model_config["marker"] == "driver-config"
    assert model_config["torch_compile"] == {
        "enable": False,
        "mode": "default",
    }


def test_ray_build_inputs_rejects_config_and_registry_family_mismatch() -> None:
    entry = _build_inputs_entry()
    entry.family = "sana"

    with pytest.raises(ValueError, match=r"model_build='sd3_5'.*entry='sana'"):
        RayGenerationLauncher.build_inputs(_build_inputs_cfg(), entry)


def test_ray_build_inputs_threads_resolved_base_weight_sync() -> None:
    """The rollout lifecycle, not model YAML, owns master-weight retention."""
    cfg = _build_inputs_cfg()
    cfg.distributed.rollout = {"sync_trainable_state": False}

    launch_inputs = RayGenerationLauncher.build_inputs(
        cfg,
        _build_inputs_entry(),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_ray_build_inputs_marks_lora_as_adapter_only_sync() -> None:
    """LoRA sync never needs retained base-precision masters on the rollout."""
    cfg = _build_inputs_cfg()
    cfg.model.use_lora = True

    launch_inputs = RayGenerationLauncher.build_inputs(
        cfg,
        _build_inputs_entry(),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_ray_build_inputs_rejects_model_compile_for_ar_family() -> None:
    """Checks model.torch_compile fails fast on rollout families that cannot compile."""
    with pytest.raises(ValueError, match="does not support torch compile"):
        RayGenerationLauncher.build_inputs(
            _build_inputs_cfg(
                model_torch_compile={
                    "enable": True,
                    "mode": "default",
                },
            ),
            _build_inputs_entry("ar_discrete"),
        )


def test_ray_build_inputs_marks_ar_generation_kind() -> None:
    launch_inputs = RayGenerationLauncher.build_inputs(
        _build_inputs_cfg(),
        _build_inputs_entry("ar_discrete"),
    )

    assert launch_inputs.launch_contract.generation_kind == "ar"


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
