"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.loading import bundled_config_resource
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.families.registry import ModelFamilyEntry, get_model_family_entry
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.utils import all_workers_support_versioned_slots
from vrl.ray.placement import GlobalRayPlacementOwner, RolePlacement
from vrl.ray.resources import resolve_distributed_resources


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


@dataclass
class _Bundle:
    model: Any
    trainable_modules: dict[str, Any]


def test_launch_contract_rejects_unknown_fields_at_typed_boundary() -> None:
    payload = {
        "family": "unit",
        "model_build": {},
        "executor": object(),
    }

    with pytest.raises(TypeError, match=r"unexpected keyword argument 'executor'"):
        GenerationRuntimeLaunchContract(**payload)  # type: ignore[arg-type]


def test_launch_contract_accepts_primitive_config_leaves() -> None:
    contract = GenerationRuntimeLaunchContract(
        family="unit",
        model_build={
            "model_name_or_path": "unit-test",
            "model_config": {
                "text": "value",
                "integer": 1,
                "number": 1.5,
                "enabled": True,
                "optional": None,
            },
        },
    )

    model_config = contract.model_build["model_config"]
    assert model_config["enabled"] is True
    assert model_config["optional"] is None


def test_launch_contract_rejects_callable_config_leaf() -> None:
    with pytest.raises(TypeError, match="callable"):
        GenerationRuntimeLaunchContract(
            family="unit",
            model_build={},
            executor_kwargs={"factory": lambda: None},
        )


def test_launch_contract_rejects_empty_registry_identity() -> None:
    with pytest.raises(ValueError, match=r"family must be non-empty"):
        GenerationRuntimeLaunchContract(
            family="",
            model_build={},
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


def _ray_config(cfg: Any) -> RayGenerationConfig:
    return RayGenerationConfig.from_cfg(
        cfg,
        resources=resolve_distributed_resources(cfg),
    )


def _launch_cfg(
    *,
    model_torch_compile: dict[str, Any] | None = None,
) -> Any:
    model_config = {
        "family": "sd3_5",
        "path": "unit-test",
        "revision": "driver-config",
        "use_lora": False,
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
        "precision": {
            "float32_precision": "tf32",
            "training": {"dtype": "bf16", "outer_autocast": True},
            # Deliberately differs from training and prompt encoding so this
            # fixture proves role-specific values survive the Ray projection.
            "rollout": {
                "dtype": "fp32",
                "outer_autocast": True,
                "prompt_encoders": {"dtype": "fp16"},
            },
        },
        "rollout": {},
    }
    return OmegaConf.create(cfg)


def _capture_launch_inputs(
    cfg: Any,
    entry: ModelFamilyEntry,
) -> RayGenerationLaunchInputs:
    """Intercept the public launch boundary without starting Ray actors."""

    captured: list[RayGenerationLaunchInputs] = []
    config = _ray_config(cfg)
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)

    def capture_launch(
        _launcher: RayGenerationLauncher,
        resolved_config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationLaunchInputs:
        assert isinstance(placement, RolePlacement)
        assert resolved_config is config
        captured.append(launch_inputs)
        return launch_inputs

    with patch.object(RayGenerationLauncher, "launch", new=capture_launch):
        result = RayGenerationLauncher(init_ray=False).launch_from_cfg(
            root,
            precision=precision,
            config=config,
            entry=entry,
            driver_bundle=_Bundle(model=_CpuPolicy(), trainable_modules={}),
            placement=RolePlacement(
                placement_group=object(),
                bundle_indices=(),
                expected_gpu_ids=(),
            ),
        )

    assert captured == [result]
    return captured[0]


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
            local_ray,
            _slot_handles(local_ray, True, True),
            weight_sync=weight_sync,
        )
        is True
    )
    assert (
        all_workers_support_versioned_slots(
            local_ray,
            _slot_handles(local_ray, True, False),
            weight_sync=weight_sync,
        )
        is False
    )


@pytest.mark.slow_test
def test_runtime_capability_false_without_weight_sync_or_workers(local_ray) -> None:
    """No weight sync (sync_trainable_state off) or no workers -> safe draining
    barrier (False), never a silent True."""
    assert (
        all_workers_support_versioned_slots(
            local_ray,
            _slot_handles(local_ray, True, True),
            weight_sync=None,
        )
        is False
    )
    assert (
        all_workers_support_versioned_slots(
            local_ray,
            [],
            weight_sync=object(),
        )
        is False
    )


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


def test_launcher_capability_failure_kills_candidate_actor_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.launcher as launcher_module

    query_error = RuntimeError("versioned-slot capability query failed")

    class _ActorGroup:
        def __init__(self) -> None:
            self.handles = [
                SimpleNamespace(
                    worker_id="rollout-0",
                    node_ip="node",
                    gpu_ids=(),
                    actor=object(),
                ),
            ]
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    actor_group = _ActorGroup()
    monkeypatch.setattr(launcher_module, "require_ray", lambda: object())
    monkeypatch.setattr(
        launcher_module.RayActorGroup,
        "launch",
        staticmethod(lambda **_kwargs: actor_group),
    )

    def _failing_query(*_args: Any, **_kwargs: Any) -> bool:
        raise query_error

    monkeypatch.setattr(
        launcher_module,
        "all_workers_support_versioned_slots",
        _failing_query,
    )
    cfg = _launch_cfg()
    config = _ray_config(cfg)
    entry = get_model_family_entry("sd3_5")
    inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
        ),
        gatherer=entry.new_gatherer(),
    )

    with pytest.raises(RuntimeError, match="capability query failed") as caught:
        RayGenerationLauncher(init_ray=False).launch(
            config,
            inputs,
            placement=RolePlacement(
                placement_group=object(),
                bundle_indices=(0,),
                expected_gpu_ids=(),
            ),
        )

    assert caught.value is query_error
    assert actor_group.shutdown_calls == 1


def test_chunk_placement_strategy_switches_from_cfg() -> None:
    """Checks distributed.rollout.chunk_placement_strategy flips the policy."""
    assert _ray_config(_cfg()).worker.chunk_placement_strategy == "round_robin"

    cfg = _cfg()
    cfg.distributed.rollout.chunk_placement_strategy = "dynamic"
    dynamic = _ray_config(cfg)
    assert dynamic.worker.chunk_placement_strategy == "dynamic"
    # Invalid values are now rejected at the typed schema boundary
    # (RolloutWorkerSection Literal) at parse time, not in RayGenerationConfig —
    # see tests/config/test_schema.py::test_unknown_chunk_placement_strategy_raises.


def test_worker_defaults_and_explicit_override_project_from_public_schema() -> None:
    default = _ray_config(_cfg()).worker
    assert default.cpus_per_worker == 1.0
    assert default.max_inflight_chunks_per_worker == 1
    assert default.pipelined is False
    assert default.sync_trainable_state is True

    cfg = _cfg()
    cfg.distributed.rollout.cpus_per_worker = 2.5
    cfg.distributed.rollout.max_inflight_chunks_per_worker = 3
    cfg.distributed.rollout.sync_trainable_state = False
    override = _ray_config(cfg).worker

    assert override.cpus_per_worker == 2.5
    assert override.max_inflight_chunks_per_worker == 3
    assert override.sync_trainable_state is False


@pytest.mark.parametrize(
    "preset_name",
    [
        "base/distributed/ray_rollout",
        "base/distributed/ray_rollout_colocated_single_gpu",
        "base/distributed/ray_rollout_cross_node",
    ],
)
def test_base_rollout_presets_pin_only_the_cpu_override(preset_name: str) -> None:
    with bundled_config_resource(preset_name).open("r", encoding="utf-8") as stream:
        cfg = OmegaConf.load(stream)

    assert dict(cfg.distributed.rollout) == {"cpus_per_worker": 4.0}
    config = RayGenerationConfig.from_cfg(
        cfg,
        resources=SimpleNamespace(
            rollout_num_workers=1,
            rollout_gpus_per_worker=1.0,
        ),
    )
    assert config.worker.cpus_per_worker == 4.0
    assert config.worker.max_inflight_chunks_per_worker == 1
    assert config.worker.chunk_placement_strategy == "round_robin"
    assert config.worker.sync_trainable_state is True


def test_ray_generation_config_requires_an_explicit_worker_snapshot() -> None:
    cfg = _cfg()
    resources = resolve_distributed_resources(cfg)

    with pytest.raises(TypeError, match="worker"):
        RayGenerationConfig(resources=resources)  # type: ignore[call-arg]


def test_rollout_worker_snapshot_is_frozen() -> None:
    worker = _ray_config(_cfg()).worker

    with pytest.raises(FrozenInstanceError):
        worker.cpus_per_worker = 2.0  # type: ignore[misc]


def test_placement_and_launcher_consume_the_same_worker_snapshot(monkeypatch) -> None:
    import vrl.generation.ray.launcher as launcher_module

    cfg = _launch_cfg()
    cfg.distributed.rollout = {
        "cpus_per_worker": 2.5,
        "health_check_interval_s": 0.0,
        "sync_trainable_state": False,
    }
    config = _ray_config(cfg)
    owner = GlobalRayPlacementOwner(config.resources, config.worker)
    assert owner.rollout_worker is config.worker
    assert owner._bundle_requirements() == [{"CPU": 2.5}]

    launch_kwargs: dict[str, Any] = {}

    class _ActorGroup:
        def __init__(self) -> None:
            self.handles = [
                SimpleNamespace(
                    worker_id="rollout-0",
                    node_ip="node",
                    gpu_ids=(),
                    actor=object(),
                ),
            ]

        @staticmethod
        def shutdown() -> None:
            return None

    def capture_actor_launch(**kwargs: Any) -> _ActorGroup:
        launch_kwargs.update(kwargs)
        return _ActorGroup()

    monkeypatch.setattr(launcher_module, "require_ray", lambda: object())
    monkeypatch.setattr(
        launcher_module.RayActorGroup,
        "launch",
        staticmethod(capture_actor_launch),
    )
    monkeypatch.setattr(
        launcher_module,
        "all_workers_support_versioned_slots",
        lambda *_args, **_kwargs: False,
    )
    entry = get_model_family_entry("sd3_5")
    runtime = RayGenerationLauncher(init_ray=False).launch(
        config,
        RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                model_build={},
            ),
            gatherer=entry.new_gatherer(),
        ),
        placement=RolePlacement(
            placement_group=object(),
            bundle_indices=(0,),
            expected_gpu_ids=(),
        ),
    )

    assert launch_kwargs["num_cpus"] == owner.rollout_worker.cpus_per_worker == 2.5
    assert runtime.executor.max_inflight_chunks_per_worker == 1


def test_health_check_settings_default_and_project_overrides() -> None:
    default = _ray_config(_cfg())
    assert default.worker.health_check_interval_s == 30.0
    assert default.worker.health_check_timeout_s == 30.0
    assert default.worker.health_check_first_wait_s == 0.0

    cfg = _cfg()
    cfg.distributed.rollout.health_check_interval_s = 5.0
    cfg.distributed.rollout.health_check_timeout_s = 37.5
    cfg.distributed.rollout.health_check_first_wait_s = 12.0
    override = _ray_config(cfg)
    assert override.worker.health_check_interval_s == 5.0
    assert override.worker.health_check_timeout_s == 37.5
    assert override.worker.health_check_first_wait_s == 12.0


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_ray_generation_config_rejects_invalid_health_check_timeout(
    timeout_s: float,
) -> None:
    cfg = _cfg()
    cfg.distributed.rollout.health_check_timeout_s = timeout_s

    with pytest.raises(ValueError, match="health_check_timeout_s must be finite and > 0"):
        _ray_config(cfg)


def test_ray_generation_config_rejects_negative_health_check_first_wait() -> None:
    cfg = _cfg()
    cfg.distributed.rollout.health_check_first_wait_s = -1.0

    with pytest.raises(ValueError, match="health_check_first_wait_s must be finite and >= 0"):
        _ray_config(cfg)


def test_pipelined_switches_from_cfg() -> None:
    """Checks distributed.rollout.pipelined flips the per-request pipelined path."""
    assert _ray_config(_cfg()).worker.pipelined is False

    cfg = _cfg()
    cfg.distributed.rollout.pipelined = True

    assert _ray_config(cfg).worker.pipelined is True


def test_pipelined_rejects_multiple_resolved_workers() -> None:
    cfg = _resource_cfg(trainer_devices=[0], rollout_devices=[1, 2])
    cfg.distributed.rollout.pipelined = True

    with pytest.raises(ValueError, match="requires exactly one rollout worker"):
        RayGenerationConfig.from_cfg(
            cfg,
            resources=resolve_distributed_resources(cfg),
        )


def test_pipelined_rejects_multiple_placement_bundles_before_ray_start(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vrl.generation.ray.launcher.require_ray",
        lambda: pytest.fail("Ray must not start before pipeline placement validation"),
    )
    cfg = _launch_cfg()
    cfg.distributed.rollout = {
        "pipelined": True,
        "sync_trainable_state": False,
    }
    config = RayGenerationConfig.from_cfg(
        cfg,
        resources=resolve_distributed_resources(cfg),
    )
    entry = get_model_family_entry("sd3_5")
    launch_inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
        ),
        gatherer=entry.new_gatherer(),
    )
    placement = RolePlacement(
        placement_group=object(),
        bundle_indices=(0, 1),
        expected_gpu_ids=(),
    )

    with pytest.raises(ValueError, match="exactly one rollout placement bundle"):
        RayGenerationLauncher(init_ray=False).launch(
            config,
            launch_inputs,
            placement=placement,
        )


def test_sync_trainable_state_defaults_on_for_from_cfg() -> None:
    """Online runs train the policy the rollout workers must resync, so an omitted
    sync_trainable_state defaults ON (True), not silently False (which would train
    the rollout on stale policy weights). Explicit values are kept."""
    assert _ray_config(_cfg()).worker.sync_trainable_state is True

    cfg = _cfg()
    cfg.distributed.rollout.sync_trainable_state = False
    assert _ray_config(cfg).worker.sync_trainable_state is False


def test_launch_from_cfg_projects_model_compile_and_precision() -> None:
    """The public launch path projects model config and dtype wire values once."""
    launch_inputs = _capture_launch_inputs(
        _launch_cfg(
            model_torch_compile={
                "enable": True,
                "mode": "default",
            },
        ),
        get_model_family_entry("sd3_5"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert launch_inputs.launch_contract.family == "sd3_5"
    assert "family" not in model_build
    assert model_build["device"] == "cpu"
    assert model_build["parameter_dtype"] == "float32"
    assert model_build["precision"] == {
        "dtype": "fp32",
        "float32_precision": "tf32",
        "quantization": None,
        "outer_autocast": True,
    }
    assert model_build["rollout"]["prompt_encoder_dtype"] == "float16"
    assert model_build["revision"] == "driver-config"
    assert "revision" not in model_build["model_config"]
    assert model_build["model_config"]["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_launch_from_cfg_preserves_disabled_model_compile_config() -> None:
    """Checks disabled model.torch_compile is preserved as ordinary model config."""
    launch_inputs = _capture_launch_inputs(
        _launch_cfg(),
        get_model_family_entry("sd3_5"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["revision"] == "driver-config"
    model_config = model_build["model_config"]
    assert "revision" not in model_config
    assert model_config["torch_compile"] == {
        "enable": False,
        "mode": "default",
    }


def test_launch_from_cfg_derives_versioned_sync_from_schedule() -> None:
    strict = _capture_launch_inputs(
        _launch_cfg(),
        get_model_family_entry("sd3_5"),
    )
    assert strict.launch_contract.versioned_weight_sync is False

    continuous_fullparam_cfg = _launch_cfg()
    continuous_fullparam_cfg.trainer = {
        "rollout_orchestration": {"schedule_mode": "continuous"},
    }
    continuous_fullparam = _capture_launch_inputs(
        continuous_fullparam_cfg,
        get_model_family_entry("sd3_5"),
    )
    assert continuous_fullparam.launch_contract.versioned_weight_sync is False

    continuous_lora_cfg = _launch_cfg()
    continuous_lora_cfg.model.use_lora = True
    continuous_lora_cfg.trainer = {
        "rollout_orchestration": {"schedule_mode": "continuous"},
    }
    continuous_lora = _capture_launch_inputs(
        continuous_lora_cfg,
        get_model_family_entry("sd3_5"),
    )
    assert continuous_lora.launch_contract.versioned_weight_sync is True


def test_launch_from_cfg_threads_resolved_base_weight_sync() -> None:
    """The rollout lifecycle, not model YAML, owns master-weight retention."""
    cfg = _launch_cfg()
    cfg.distributed.rollout = {"sync_trainable_state": False}

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_launch_from_cfg_marks_lora_as_adapter_only_sync() -> None:
    """LoRA sync never needs retained base-precision masters on the rollout."""
    cfg = _launch_cfg()
    cfg.model.use_lora = True

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_launch_from_cfg_rejects_model_compile_for_ar_family() -> None:
    """Checks model.torch_compile fails fast on rollout families that cannot compile."""
    cfg = _launch_cfg(
        model_torch_compile={
            "enable": True,
            "mode": "default",
        },
    )
    cfg.model.family = "janus_pro"
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)

    with pytest.raises(ValueError, match="does not support torch compile"):
        RayGenerationLauncher(init_ray=False).launch_from_cfg(
            root,
            precision=precision,
            config=_ray_config(cfg),
            entry=get_model_family_entry("janus_pro"),
            driver_bundle=_Bundle(model=_CpuPolicy(), trainable_modules={}),
            placement=RolePlacement(
                placement_group=object(),
                bundle_indices=(),
                expected_gpu_ids=(),
            ),
        )


def test_launcher_default_ray_init_is_owned_local() -> None:
    assert RayGenerationLauncher().ray_init_kwargs == {"address": "local"}


def test_ray_backend_rejects_unapproved_driver_cuda_overlap() -> None:
    """The runtime backstop reports the concrete conflicting devices and policy."""
    config = _ray_config(
        _resource_cfg(
            trainer_devices=[1],
            rollout_devices=[0],
        ),
    )
    # The resolved trainer owns GPU 1, but the actual driver model reports GPU
    # 0. The launch boundary must reject that real topology mismatch.

    with pytest.raises(
        ValueError,
        match=(
            r"Trainer device cuda:0 overlaps rollout devices \[0\], "
            r"but resources\.allow_overlap=false"
        ),
    ):
        config.validate_driver_state(
            driver_bundle=_Bundle(model=_CudaPolicy(), trainable_modules={}),
        )


@pytest.mark.gpu
def test_ray_backend_detects_cuda_trainable_module_when_policy_has_no_device() -> None:
    """Checks Ray backend detects cuda trainable module when policy has no device."""
    bundle = _Bundle(
        model=object(),
        trainable_modules={"transformer": torch.nn.Linear(1, 1).to("cuda:0")},
    )

    config = _ray_config(
        _resource_cfg(
            trainer_devices=[1],
            rollout_devices=[0],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"Trainer device cuda:0 overlaps rollout devices \[0\]",
    ):
        config.validate_driver_state(driver_bundle=bundle)


def test_ray_backend_allows_driver_cuda_policy_with_explicit_overlap() -> None:
    """Checks Ray backend allows driver cuda policy with explicit overlap.

    A colocated single-GPU topology derives on-demand rollout activation, so the
    driver CUDA policy overlapping the rollout GPU is allowed.
    """
    config = _ray_config(
        _resource_cfg(
            trainer_devices=[0],
            rollout_devices=[0],
            allow_overlap=True,
        ),
    ).validate_driver_state(
        driver_bundle=_Bundle(model=_CudaPolicy(), trainable_modules={}),
    )

    assert config.resources.colocated is True
    assert config.resources.lifecycle.rollout.mode == "on_demand"


def test_ray_backend_allows_split_driver_cuda_when_devices_do_not_overlap() -> None:
    """Checks Ray backend allows split driver cuda when devices do not overlap."""
    config = _ray_config(
        _resource_cfg(trainer_devices=[0], rollout_devices=[1]),
    ).validate_driver_state(
        driver_bundle=_Bundle(model=_CudaPolicy(), trainable_modules={}),
    )

    assert config.resources.trainer_devices == (0,)
    assert config.resources.rollout_devices == (1,)
    assert config.resources.colocated is False
