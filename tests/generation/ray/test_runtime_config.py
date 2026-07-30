"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.builders import BuiltConfigs
from vrl.config.loading import bundled_config_resource
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.families.registry import ModelFamilyEntry, get_model_family_entry
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import (
    RayGenerationLauncher,
    _all_workers_support_versioned_slots,
)
from vrl.generation.ray.lifecycle_fsm import RuntimePhase
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationRequest
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.ray.placement import GlobalRayPlacementOwner, RolePlacement
from vrl.ray.resources import resolve_distributed_resources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.run import (
    OnlineRunConfig,
    ResolvedModel,
    ResolvedOnlineRun,
    resolve_ray_generation_launch_inputs,
)
from vrl.trainers.checkpointing import TrainingResumeConfig


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


@dataclass
class _Bundle:
    """Driver-bundle stand-in for the CUDA-ownership checks these tests cover.

    ``loads_full_generation_modules`` defaults to the replay answer so the
    colocated-RAM guard stays out of the way; the guard has its own tests in
    ``tests/trainers/test_memory_guards.py``.
    """

    model: Any
    trainable_modules: dict[str, Any]
    loads_full_generation_modules: bool = False


_TEST_MODEL_IDENTITY = {"schema": "test"}
_TEST_RPC_TIMEOUT_S = 30.0


def test_launch_contract_rejects_unknown_fields_at_typed_boundary() -> None:
    payload = {
        "family": "unit",
        "model_build": {},
        "expected_model_identity": _TEST_MODEL_IDENTITY,
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
        expected_model_identity=_TEST_MODEL_IDENTITY,
    )

    model_config = contract.model_build["model_config"]
    assert model_config["enabled"] is True
    assert model_config["optional"] is None
    assert contract.expected_model_identity == _TEST_MODEL_IDENTITY


def test_launch_contract_rejects_callable_config_leaf() -> None:
    with pytest.raises(TypeError, match="callable"):
        GenerationRuntimeLaunchContract(
            family="unit",
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
            executor_kwargs={"factory": lambda: None},
        )


def test_launch_contract_rejects_empty_registry_identity() -> None:
    with pytest.raises(ValueError, match=r"family must be non-empty"):
        GenerationRuntimeLaunchContract(
            family="",
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
        )


def test_launch_contract_rejects_empty_model_identity() -> None:
    with pytest.raises(ValueError, match=r"expected_model_identity must be non-empty"):
        GenerationRuntimeLaunchContract(
            family="unit",
            model_build={},
            expected_model_identity={},
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
    *,
    rollout_model_identity: dict[str, Any] | None = None,
) -> RayGenerationLaunchInputs:
    """Resolve the public Ray worker boundary without starting actors."""

    config = _ray_config(cfg)
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    schedule_mode = str(
        OmegaConf.select(
            cfg,
            "trainer.rollout_orchestration.schedule_mode",
            default="strict_on_policy",
        ),
    )
    run = ResolvedOnlineRun(
        built=BuiltConfigs(
            root=root,
            algorithm=None,
            precision=precision,
            trainer=SimpleNamespace(
                rollout_orchestration=SimpleNamespace(schedule_mode=schedule_mode),
            ),
            reward=None,
            resume=TrainingResumeConfig(),
        ),
        family=entry,
        resources=config.resources,
        device=torch.device("cpu"),
        run=OnlineRunConfig(total_epochs=0),
        generation=config,
        collector=RolloutCollectorConfig(),
    )
    replay_model = ResolvedModel(
        entry=entry,
        build=entry.resolve_model_build(
            root,
            torch.device("cpu"),
            precision=precision,
            for_rollout=False,
        ),
        identity=_TEST_MODEL_IDENTITY,
    )

    with (
        patch(
            "vrl.models.checkpoint_identity.resolve_checkpoint_model_identity",
            return_value=rollout_model_identity or _TEST_MODEL_IDENTITY,
        ) as resolve_identity,
    ):
        result = resolve_ray_generation_launch_inputs(
            run,
            replay_model=replay_model,
        )

    resolve_identity.assert_called_once()
    return result


class _SlotWorker:
    """Real Ray actor exposing the versioned-slot capability query;
    ``supports=None`` raises like a dead/broken worker."""

    def __init__(self, supports: bool | None) -> None:
        self._supports = supports

    def supports_versioned_trainable_state(self) -> bool:
        if self._supports is None:
            raise RuntimeError("actor dead")
        return self._supports


@contextlib.contextmanager
def _slot_handles(ray: Any, *supports: bool | None) -> Iterator[list[DistributedWorkerHandle]]:
    """Real ``_SlotWorker`` actors, killed on exit.

    The cluster is shared across this package, so a fleet left running would keep
    holding actors for every later test. See ``real_local_ray``'s docstring.
    """

    actor_cls = ray.remote(num_cpus=0)(_SlotWorker)
    handles = [
        DistributedWorkerHandle(
            worker_id=f"w{index}",
            actor=actor_cls.remote(value),
        )
        for index, value in enumerate(supports)
    ]
    try:
        yield handles
    finally:
        for handle in handles:
            ray.kill(handle.actor, no_restart=True)


@pytest.mark.slow_test
def test_runtime_capability_is_and_over_all_workers(local_ray) -> None:
    """supports_non_draining_weight_sync derives as the AND of every worker's
    supports_versioned_trainable_state(): all True -> True; any False -> False."""
    weight_sync = object()

    with _slot_handles(local_ray, True, True) as handles:
        assert (
            _all_workers_support_versioned_slots(
                local_ray,
                handles,
                weight_sync=weight_sync,
                worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
            )
            is True
        )
    with _slot_handles(local_ray, True, False) as handles:
        assert (
            _all_workers_support_versioned_slots(
                local_ray,
                handles,
                weight_sync=weight_sync,
                worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
            )
            is False
        )


@pytest.mark.slow_test
def test_runtime_capability_false_without_weight_sync_or_workers(local_ray) -> None:
    """No weight sync (sync_trainable_state off) or no workers -> safe draining
    barrier (False), never a silent True."""
    with _slot_handles(local_ray, True, True) as handles:
        assert (
            _all_workers_support_versioned_slots(
                local_ray,
                handles,
                weight_sync=None,
                worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
            )
            is False
        )
    assert (
        _all_workers_support_versioned_slots(
            local_ray,
            [],
            weight_sync=object(),
            worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
        )
        is False
    )


@pytest.mark.slow_test
def test_runtime_capability_worker_query_failure_propagates(local_ray) -> None:
    """A failed capability RPC means the candidate worker is broken."""
    with (
        _slot_handles(local_ray, True, None) as handles,
        pytest.raises(local_ray.exceptions.RayTaskError, match="actor dead"),
    ):
        _all_workers_support_versioned_slots(
            local_ray,
            handles,
            weight_sync=object(),
            worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
        )


def test_launcher_capability_failure_kills_candidate_actor_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.launcher as launcher_module

    query_error = RuntimeError("versioned-slot capability query failed")
    capability_ref = object()

    class _CapabilityMethod:
        @staticmethod
        def remote() -> object:
            return capability_ref

    class _GetTimeoutError(TimeoutError):
        pass

    class _RayApi:
        exceptions = SimpleNamespace(GetTimeoutError=_GetTimeoutError)

        @staticmethod
        def get(refs: list[object], *, timeout: float) -> None:
            assert refs == [capability_ref]
            assert timeout > 0
            raise query_error

    class _ActorGroup:
        def __init__(self) -> None:
            self.handles = [
                SimpleNamespace(
                    worker_id="rollout-0",
                    node_ip="node",
                    gpu_ids=(),
                    actor=SimpleNamespace(
                        supports_versioned_trainable_state=_CapabilityMethod(),
                    ),
                ),
            ]
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    actor_group = _ActorGroup()
    monkeypatch.setattr(launcher_module, "require_ray", lambda: _RayApi)
    monkeypatch.setattr(
        launcher_module.RayActorGroup,
        "launch",
        staticmethod(lambda **_kwargs: actor_group),
    )

    cfg = _launch_cfg()
    config = _ray_config(cfg)
    entry = get_model_family_entry("sd3_5")
    inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
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
    assert default.worker_rpc_timeout_s == 600.0
    assert default.generation_stall_timeout_s == 3600.0
    assert default.pipelined is False
    assert default.sync_trainable_state is True

    cfg = _cfg()
    cfg.distributed.rollout.cpus_per_worker = 2.5
    cfg.distributed.rollout.worker_rpc_timeout_s = 3600.0
    cfg.distributed.rollout.generation_stall_timeout_s = 1200.0
    cfg.distributed.rollout.sync_trainable_state = False
    override = _ray_config(cfg).worker

    assert override.cpus_per_worker == 2.5
    assert override.worker_rpc_timeout_s == 3600.0
    assert override.generation_stall_timeout_s == 1200.0
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
    assert config.worker.worker_rpc_timeout_s == 600.0
    assert config.worker.generation_stall_timeout_s == 3600.0
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
        "sync_trainable_state": True,
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
        "_all_workers_support_versioned_slots",
        lambda *_args, **_kwargs: False,
    )
    entry = get_model_family_entry("sd3_5")
    runtime = RayGenerationLauncher(init_ray=False).launch(
        config,
        RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                model_build={},
                expected_model_identity=_TEST_MODEL_IDENTITY,
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
    assert launch_kwargs["rpc_timeout_s"] == config.worker.worker_rpc_timeout_s
    assert launch_kwargs["operation_prefix"] == "rollout"
    assert runtime.executor.generation_stall_timeout_s == config.worker.generation_stall_timeout_s
    assert runtime.weight_sync is not None
    assert runtime.executor.actor_dispatcher is runtime.weight_sync.actor_dispatcher


def test_runtime_rejects_split_actor_admission_owners() -> None:
    with pytest.raises(ValueError, match="share one actor dispatcher"):
        RayGenerationRuntime(
            SimpleNamespace(actor_dispatcher=RayActorDispatcher(("rollout-0",))),
            weight_sync=SimpleNamespace(
                actor_dispatcher=RayActorDispatcher(("rollout-0",)),
            ),
        )


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


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_ray_generation_config_rejects_invalid_worker_rpc_timeout(
    timeout_s: float,
) -> None:
    cfg = _cfg()
    cfg.distributed.rollout.worker_rpc_timeout_s = timeout_s

    with pytest.raises(ValueError, match="worker_rpc_timeout_s must be finite and > 0"):
        _ray_config(cfg)


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_ray_generation_config_rejects_invalid_generation_stall_timeout(
    timeout_s: float,
) -> None:
    cfg = _cfg()
    cfg.distributed.rollout.generation_stall_timeout_s = timeout_s

    with pytest.raises(ValueError, match="generation_stall_timeout_s must be finite and > 0"):
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
            expected_model_identity=_TEST_MODEL_IDENTITY,
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


def test_generation_launch_inputs_project_model_compile_and_precision() -> None:
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
    assert launch_inputs.launch_contract.expected_model_identity == _TEST_MODEL_IDENTITY
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


def test_generation_launch_inputs_project_resolved_generation_memory() -> None:
    """The launch contract carries typed memory values, not raw model config."""
    cfg = _launch_cfg()
    cfg.model.memory = {
        "vae_decode": {
            "tiling": True,
            "slicing": False,
        },
    }

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["generation_memory"] == {
        "vae_decode": {
            "tiling": True,
            "slicing": False,
        },
    }
    assert "memory" not in model_build["model_config"]


def test_generation_launch_inputs_project_wan_offload_to_rollout_contract(monkeypatch) -> None:
    from diffusers import DiffusionPipeline

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(lambda *a, **k: {"boundary_ratio": None}),
    )
    cfg = _launch_cfg()
    cfg.model.family = "wan_2_1_i2v"
    cfg.model.revision = "a" * 40
    cfg.model.offload_mode = "sequential"

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("wan_2_1_i2v"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["rollout"]["pipeline_offload_mode"] == "sequential"
    assert "offload_mode" not in model_build["model_config"]


def test_generation_launch_inputs_reject_rollout_identity_mismatch() -> None:
    cfg = _launch_cfg()

    with pytest.raises(
        ValueError,
        match="rollout model identity does not match the driver replay model identity",
    ):
        _capture_launch_inputs(
            cfg,
            get_model_family_entry("sd3_5"),
            rollout_model_identity={"schema": "different"},
        )


def test_generation_launch_inputs_preserve_disabled_model_compile_config() -> None:
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


def test_generation_launch_inputs_derive_versioned_sync_from_schedule() -> None:
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


def test_generation_launch_inputs_thread_resolved_base_weight_sync() -> None:
    """The rollout lifecycle, not model YAML, owns master-weight retention."""
    cfg = _launch_cfg()
    cfg.distributed.rollout = {"sync_trainable_state": False}

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_generation_launch_inputs_mark_lora_as_adapter_only_sync() -> None:
    """LoRA sync never needs retained base-precision masters on the rollout."""
    cfg = _launch_cfg()
    cfg.model.use_lora = True

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_generation_launch_inputs_reject_model_compile_for_ar_family() -> None:
    """Checks model.torch_compile fails fast on rollout families that cannot compile."""
    cfg = _launch_cfg(
        model_torch_compile={
            "enable": True,
            "mode": "default",
        },
    )
    cfg.model.family = "janus_pro"

    with pytest.raises(ValueError, match="does not support torch compile"):
        _capture_launch_inputs(cfg, get_model_family_entry("janus_pro"))


def test_create_runtime_launches_resident_topology() -> None:
    config = _ray_config(_launch_cfg())
    entry = get_model_family_entry("sd3_5")
    launch_inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
        ),
        gatherer=entry.new_gatherer(),
    )
    placement = RolePlacement(
        placement_group=object(),
        bundle_indices=(),
        expected_gpu_ids=(),
    )
    launcher = RayGenerationLauncher(init_ray=False)
    expected_runtime = object()

    with (
        patch.object(
            RayGenerationLauncher,
            "launch",
            autospec=True,
            return_value=expected_runtime,
        ) as launch,
        patch(
            "vrl.generation.ray.launcher._OnDemandRayGenerationRuntime",
            autospec=True,
        ) as on_demand,
    ):
        runtime = launcher.create_runtime(
            config,
            launch_inputs,
            placement=placement,
        )

    assert runtime is expected_runtime
    launch.assert_called_once_with(
        launcher,
        config,
        launch_inputs,
        placement=placement,
    )
    on_demand.assert_not_called()


def test_create_runtime_defers_on_demand_topology_launch() -> None:
    resident_config = _ray_config(_launch_cfg())
    resources = replace(
        resident_config.resources,
        lifecycle=replace(
            resident_config.resources.lifecycle,
            rollout=replace(
                resident_config.resources.lifecycle.rollout,
                mode="on_demand",
            ),
        ),
    )
    config = RayGenerationConfig(
        resources=resources,
        worker=resident_config.worker,
    )
    entry = get_model_family_entry("sd3_5")
    launch_inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
        ),
        gatherer=entry.new_gatherer(),
    )
    placement = RolePlacement(
        placement_group=object(),
        bundle_indices=(),
        expected_gpu_ids=(),
    )
    launcher = RayGenerationLauncher(init_ray=False)
    expected_runtime = object()

    with (
        patch.object(
            RayGenerationLauncher,
            "launch",
            autospec=True,
        ) as launch,
        patch(
            "vrl.generation.ray.launcher._OnDemandRayGenerationRuntime",
            autospec=True,
            return_value=expected_runtime,
        ) as on_demand,
    ):
        runtime = launcher.create_runtime(
            config,
            launch_inputs,
            placement=placement,
        )

    assert runtime is expected_runtime
    launch.assert_not_called()
    on_demand.assert_called_once_with(
        config,
        launch_inputs,
        placement=placement,
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


# ------------------------------------- real chunk-size probe fan-out (real Ray)


def _auto_chunk_request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req-probe",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=10,
        sampling={"num_steps": 20, "samples_per_chunk": "auto"},
        policy_version=1,
    )


@pytest.mark.asyncio
async def test_remote_chunk_size_probe_timeout_is_terminal_and_cancels_refs(
    monkeypatch,
) -> None:
    import vrl.ray.operation_deadline as deadline_module

    class _NeverRef:
        def __await__(self):
            async def wait_forever() -> None:
                await asyncio.Event().wait()

            return wait_forever().__await__()

    ref = _NeverRef()

    class _RemoteProbe:
        @staticmethod
        def remote(_request: Any, *, max_samples: int) -> _NeverRef:
            assert max_samples == 10
            return ref

    worker = DistributedWorkerHandle(
        worker_id="w0",
        actor=SimpleNamespace(probe_chunk_size=_RemoteProbe()),
    )
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [worker],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=0.01,
    )

    async def execute(_request: Any) -> None:
        raise AssertionError("timed-out probe must not enter generation")

    executor.execute = execute

    class _Ray:
        cancelled: ClassVar[list[tuple[Any, bool]]] = []

        @classmethod
        def cancel(cls, value: Any, *, force: bool) -> None:
            cls.cancelled.append((value, force))

    runtime = RayGenerationRuntime(executor)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    with pytest.raises(
        RayOperationTimeout,
        match=r"rollout\.generation\.chunk_size_probe",
    ):
        await runtime.generate(_auto_chunk_request())

    assert _Ray.cancelled == [(ref, False)]
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_concurrent_auto_chunk_requests_share_one_probe_before_submission() -> None:
    gate = asyncio.Event()
    probe_requests: list[str] = []
    executed_requests: list[GenerationRequest] = []
    probe_result = {
        "samples_per_chunk": 3,
        "budget_bytes": 1,
        "trials": [],
    }

    class _ProbeRef:
        def __await__(self):
            async def wait() -> dict[str, Any]:
                await gate.wait()
                return probe_result

            return wait().__await__()

    class _RemoteProbe:
        @staticmethod
        def remote(request: GenerationRequest, *, max_samples: int) -> _ProbeRef:
            assert max_samples == 10
            probe_requests.append(request.request_id)
            return _ProbeRef()

    worker = DistributedWorkerHandle(
        worker_id="w0",
        actor=SimpleNamespace(probe_chunk_size=_RemoteProbe()),
    )
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [worker],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=30.0,
    )

    async def execute(request: GenerationRequest) -> GenerationRequest:
        executed_requests.append(request)
        return request

    executor.execute = execute
    runtime = RayGenerationRuntime(executor)
    first_request = _auto_chunk_request()
    second_request = replace(first_request, request_id="req-probe-second")

    first = asyncio.create_task(runtime.generate(first_request))
    await asyncio.sleep(0)
    second = asyncio.create_task(runtime.generate(second_request))
    await asyncio.sleep(0)

    assert probe_requests == ["req-probe"]

    gate.set()
    assert await first == executed_requests[0]
    assert await second == executed_requests[1]
    assert probe_requests == ["req-probe"]
    assert [request.sampling["samples_per_chunk"] for request in executed_requests] == [3, 3]


class _Arrivals:
    """Counts probe arrivals across actor processes so concurrency is observable."""

    def __init__(self) -> None:
        self._count = 0

    def arrived(self) -> int:
        self._count += 1
        return self._count

    def count(self) -> int:
        return self._count


class _ProbeWorker:
    """Real Ray actor exposing ``probe_chunk_size`` as a remote method.

    It blocks until the whole fleet has arrived, so "probed concurrently" becomes
    a fact the test can fail on rather than a word in a name.
    """

    def __init__(self, answer: int, arrivals: Any, fleet_size: int) -> None:
        self._answer = int(answer)
        self._arrivals = arrivals
        self._fleet_size = int(fleet_size)
        self._calls = 0

    def probe_chunk_size(self, request: Any, *, max_samples: int) -> dict[str, Any]:
        import time

        import ray

        # Asserted inside the actor process: the request really survived Ray
        # serialization with its type and fields intact. Nothing else checks this.
        assert isinstance(request, GenerationRequest), type(request).__name__
        assert request.sampling["samples_per_chunk"] == "auto"
        assert request.inputs[0].prompt == "p"
        assert max_samples == 10
        self._calls += 1

        ray.get(self._arrivals.arrived.remote())
        deadline = time.monotonic() + 20.0
        while ray.get(self._arrivals.count.remote()) < self._fleet_size:
            if time.monotonic() > deadline:
                raise TimeoutError("probes were dispatched sequentially, not concurrently")
            time.sleep(0.01)
        return {"samples_per_chunk": self._answer, "budget_bytes": 32 * 1024**3, "trials": []}

    def calls(self) -> int:
        return self._calls


@pytest.mark.slow_test
def test_real_ray_probe_fan_out_resolves_auto_once_across_the_fleet(local_ray) -> None:
    """The executor-owned chunk-size probe path on a live cluster.

    The executor sends N remote probes through its shared actor dispatcher, so
    generation cannot enter the same synchronous actor mailbox concurrently.
    The barrier inside ``_ProbeWorker`` makes fleet fan-out checkable: an
    implementation that waited on each worker inside the submission loop would
    deadlock instead of passing.
    """

    arrivals = local_ray.remote(num_cpus=0)(_Arrivals).remote()
    actor_cls = local_ray.remote(num_cpus=0)(_ProbeWorker)
    actors = [actor_cls.remote(answer, arrivals, 2) for answer in (6, 4)]
    executed: list[Any] = []

    workers = [
        DistributedWorkerHandle(worker_id=f"w{index}", actor=actor)
        for index, actor in enumerate(actors)
    ]
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        workers,
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(tuple(worker.worker_id for worker in workers)),
        generation_stall_timeout_s=30.0,
    )

    async def execute(request: Any) -> Any:
        executed.append(request)
        return SimpleNamespace(request_id=request.request_id)

    executor.execute = execute
    runtime = RayGenerationRuntime(executor)

    async def go() -> None:
        await runtime.generate(_auto_chunk_request())
        await runtime.generate(_auto_chunk_request())

    try:
        asyncio.run(go())

        # Fleet answer is the min, and it is probed once: the second request is
        # rewritten from the cached verdict, so no actor sees a second probe.
        assert [request.sampling["samples_per_chunk"] for request in executed] == [4, 4]
        assert local_ray.get([actor.calls.remote() for actor in actors]) == [1, 1]
    finally:
        for actor in (*actors, arrivals):
            local_ray.kill(actor, no_restart=True)
