"""Tests for the Ray generation runtime's configuration, launch contract and launch.

Launch-contract projections resolve a real tiny run -- the SANA online recipe
on its tiny snapshot, or the tiny Wan snapshot for Wan-only knobs -- and read
the contract the recipe would ship to its workers. Launch-shape and topology
tests launch a real fleet on the package cluster (``ray_sana_runtime``).
Contract-validation and config-projection tests construct the production
values directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from tests.generation.ray._helpers import ray_sana_runtime
from tests.scripts.eval.fixtures import tiny_sana_online_config, write_tiny_wan_snapshot
from vrl import run
from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.config.validation import compile_conflicts
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.placement import RolePlacement
from vrl.ray.resources import ResolvedDistributedResources
from vrl.utils.lifecycle import RuntimePhase

_TEST_MODEL_IDENTITY = {"schema": "test"}
_CONTINUOUS = (
    "/base/rollout/orchestration=continuous",
    "trainer.rollout_orchestration.continuous.max_stale_policy_versions=1",
)


# ------------------------------------------------------- launch contract values


@pytest.mark.parametrize("family", ["", None, True, 123, ["unit"], {"name": "unit"}])
def test_launch_contract_rejects_invalid_registry_identity(family) -> None:
    with pytest.raises(ValueError, match=r"family must be a non-empty string"):
        GenerationRuntimeLaunchContract(
            family=family,
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


@pytest.mark.parametrize("policy_version", [True, 1.9, "1", -1])
def test_launch_contract_rejects_invalid_policy_version(policy_version) -> None:
    with pytest.raises(ValueError, match="policy_version must be"):
        GenerationRuntimeLaunchContract(
            family="unit",
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
            policy_version=policy_version,
        )


# ------------------------------------------------------- worker config projection


def _cfg() -> Any:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "num_engines": 1},
                },
                # Release scheduling is derived from topology; nothing to spell here.
                "rollout": {},
            },
        },
    )


def _ray_config(cfg: Any) -> RayGenerationConfig:
    return RayGenerationConfig.from_root(
        parse_config(cfg),
        resources=ResolvedDistributedResources.from_root(parse_config(cfg)),
    )


def test_worker_override_projects_from_public_schema() -> None:
    cfg = _cfg()
    cfg.distributed.rollout.cpus_per_worker = 2.5
    cfg.distributed.rollout.worker_rpc_timeout_s = 3600.0
    cfg.distributed.rollout.generation_stall_timeout_s = 1200.0
    cfg.distributed.rollout.pipelined = True
    override = _ray_config(cfg).worker

    assert override.cpus_per_worker == 2.5
    assert override.worker_rpc_timeout_s == 3600.0
    assert override.generation_stall_timeout_s == 1200.0
    assert override.pipelined is True


def test_pipelined_accepts_multiple_resolved_engines() -> None:
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1, 2], "num_engines": 2},
                },
                "rollout": {"cpus_per_worker": 1, "pipelined": True},
            },
        },
    )

    config = _ray_config(cfg)

    assert config.worker.pipelined is True
    assert config.resources.rollout_num_engines == 2


# ------------------------------------------------------- real launch shape


def _capture_launches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the keyword arguments of every real actor-group launch."""

    launches: list[dict[str, Any]] = []
    real_launch = RayActorGroup.launch

    def launch(**kwargs: Any) -> RayActorGroup:
        launches.append(kwargs)
        return real_launch(**kwargs)

    monkeypatch.setattr(RayActorGroup, "launch", staticmethod(launch))
    return launches


@pytest.mark.asyncio
async def test_placement_and_launcher_consume_the_same_worker_snapshot(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    launches = _capture_launches(monkeypatch)

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=("distributed.rollout.cpus_per_worker=1.5",),
    ) as ray_run:
        worker = ray_run.resolved.generation.worker
        owner = ray_run.placement_owner
        session = ray_run.runtime._session

        assert owner.rollout_worker is worker
        assert owner._bundle_requirements() == [{"CPU": 1.5}]
        (launch,) = launches
        assert launch["num_cpus"] == worker.cpus_per_worker == 1.5
        assert launch["rpc_timeout_s"] == worker.worker_rpc_timeout_s
        assert launch["operation_prefix"] == "rollout"
        assert session.executor.generation_stall_timeout_s == worker.generation_stall_timeout_s
        assert session.executor.actor_dispatcher is session.weight_sync.actor_dispatcher


@pytest.mark.asyncio
async def test_pipelined_launch_adds_one_finalizer_per_engine(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """The per-request path launches a CPU finalizer beside every engine,
    pinned to the engine's primary bundle and reserving no CPU or GPU."""

    launches = _capture_launches(monkeypatch)

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=(
            "distributed.rollout.pipelined=true",
            "distributed.resources.rollout.num_engines=2",
        ),
    ) as ray_run:
        session = ray_run.runtime._session

        ranks, finalize = launches
        assert ranks["worker_cls"].__name__ == "RayGenerationWorker"
        assert finalize["worker_cls"].__name__ == "RayGenerationFinalizer"
        assert finalize["worker_ids"] == ["rollout-0.finalize", "rollout-1.finalize"]
        assert finalize["num_cpus"] == 0
        assert finalize["num_gpus"] == 0
        # One rank per engine here, so each engine's primary bundle is its only one.
        assert finalize["bundle_indices"] == list(ranks["bundle_indices"])
        assert finalize["operation_prefix"] == "rollout.finalize"
        assert "startup_method" not in finalize
        assert [handle.worker_id for handle in session.finalizer_handles] == finalize["worker_ids"]
        assert session.executor.finalizers == tuple(session.finalizer_handles)
        # The finalizers really merge: a request runs end to end on the path.
        output = await ray_run.runtime.generate(ray_run.request(["a cat"]))
        assert output.output.shape[0] == 2


# ------------------------------------------------------- launch contract projection


def _sana_launch(tmp_path: Path, *overrides: str) -> tuple[RayGenerationLaunchInputs, Any]:
    """The launch inputs the recipe would ship for a tiny SANA run, and its replay model."""

    resolved = run.resolve_online_run(tiny_sana_online_config(tmp_path, overrides=overrides))
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    return resolved.ray_launch_inputs(replay), replay


def test_generation_launch_inputs_project_model_compile_and_precision(tmp_path) -> None:
    """The public launch path projects model config and dtype wire values once."""

    launch_inputs, replay = _sana_launch(
        tmp_path,
        "model.revision=driver-config",
        "model.torch_compile.enable=true",
        "actor.gradient_checkpointing=false",
        "precision.float32_precision=tf32",
        # Training differs from rollout and prompt encoding, so the contract
        # must carry the rollout role's own values.
        "precision.training.dtype=bf16",
        "precision.rollout.dtype=fp32",
        "precision.rollout.outer_autocast=true",
        "precision.rollout.prompt_encoders.dtype=fp16",
    )

    contract = launch_inputs.launch_contract
    model_build = contract.model_build
    assert contract.family == "sana"
    assert contract.expected_model_identity == replay.identity
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
    assert model_build["model_config"]["torch_compile"] == {"enable": True, "mode": "default"}


def test_generation_launch_inputs_project_resolved_generation_memory(tmp_path) -> None:
    """The launch contract carries typed memory values, not raw model config."""

    launch_inputs, _ = _sana_launch(
        tmp_path,
        "model.memory.vae_decode.tiling=true",
        "model.memory.vae_decode.slicing=false",
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["generation_memory"] == {
        "vae_decode": {"tiling": True, "slicing": False},
        "cpu_resident": (),
    }
    assert "memory" not in model_build["model_config"]


def test_generation_launch_inputs_project_wan_offload_to_rollout_contract(tmp_path) -> None:
    """A Wan run's public ``model.offload_mode`` becomes the rollout build's
    pipeline residency, resolved from a real Wan snapshot on disk."""

    snapshot = write_tiny_wan_snapshot(tmp_path / "wan-snapshot")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(json.dumps({"prompt": "a cat"}) + "\n", encoding="utf-8")
    cfg = load_config(
        "experiment/wan_2_1/online_grpo_hpsv3_4x_l40s",
        overrides=[
            f"model.path={snapshot}",
            "model.revision=null",
            "model.offload_mode=sequential",
            f"data.manifest={manifest}",
            f"trainer.output_dir={tmp_path / 'run'}",
            "distributed.training.strategy=single_process",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
        ],
    )
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )

    model_build = resolved.ray_launch_inputs(replay).launch_contract.model_build

    assert model_build["rollout"]["pipeline_offload_mode"] == "sequential"
    assert "offload_mode" not in model_build["model_config"]


def test_generation_launch_inputs_reject_rollout_identity_mismatch(tmp_path) -> None:
    """The checkpoint changing between the driver's replay resolution and the
    worker launch is refused before any actor starts."""

    resolved = run.resolve_online_run(tiny_sana_online_config(tmp_path))
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    (tmp_path / "sana-snapshot" / "extra-weights.bin").write_bytes(b"drift")

    with pytest.raises(
        ValueError,
        match="rollout model identity does not match the driver replay model identity",
    ):
        resolved.ray_launch_inputs(replay)


def test_generation_launch_inputs_preserve_disabled_model_compile_config(tmp_path) -> None:
    """Checks disabled model.torch_compile is preserved as ordinary model config."""

    launch_inputs, _ = _sana_launch(tmp_path, "model.revision=driver-config")

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["revision"] == "driver-config"
    model_config = model_build["model_config"]
    assert "revision" not in model_config
    assert model_config["torch_compile"] == {"enable": False, "mode": "default"}


def test_generation_launch_inputs_derive_versioned_sync_from_schedule(tmp_path) -> None:
    strict, _ = _sana_launch(tmp_path / "strict")
    continuous_fullparam, _ = _sana_launch(tmp_path / "fullparam", *_CONTINUOUS)
    continuous_lora, _ = _sana_launch(tmp_path / "lora", "model.use_lora=true", *_CONTINUOUS)

    assert strict.launch_contract.versioned_weight_sync is False
    assert continuous_fullparam.launch_contract.versioned_weight_sync is False
    assert continuous_lora.launch_contract.versioned_weight_sync is True


def test_generation_launch_inputs_thread_resolved_base_weight_sync(tmp_path) -> None:
    """Master-weight retention follows the sync payload: a full-parameter sync
    replaces base weights, so the worker retains masters."""

    launch_inputs, _ = _sana_launch(tmp_path)

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["model_config"].get("use_lora", False) is False
    assert model_build["rollout"]["base_weight_sync"] is True


def test_generation_launch_inputs_mark_lora_as_adapter_only_sync(tmp_path) -> None:
    """LoRA sync never needs retained base-precision masters on the rollout."""

    launch_inputs, _ = _sana_launch(tmp_path, "model.use_lora=true")

    assert launch_inputs.launch_contract.model_build["rollout"]["base_weight_sync"] is False


def test_compile_on_an_uncompilable_family_is_a_config_conflict() -> None:
    """model.torch_compile on a family whose runtime cannot compile fails at
    config load, with the rest of the compile compatibility matrix.

    MAGI-1 is the only family without compile support.
    """

    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [],
                    "trainer": {"num_gpus": 0, "devices": []},
                    "rollout": {"num_gpus": 0, "devices": [], "num_engines": 1},
                },
            },
            "model": {
                "family": "magi_1",
                "path": "unit-test",
                "use_lora": False,
                "torch_compile": {"enable": True, "mode": "default"},
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"dtype": "fp32"},
            },
            "rollout": {},
        },
    )
    conflicts = compile_conflicts(parse_config(cfg))

    assert "family" in [conflict.feature for conflict in conflicts]


# ------------------------------------------------------- runtime topology


@pytest.mark.asyncio
async def test_create_runtime_launches_resident_topology(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime

        assert ray_run.resolved.resources.lifecycle.rollout_mode == "resident"
        assert runtime._session is not None
        assert runtime._session_factory is None
        assert [rank.worker_id for rank in runtime._session.rank_handles] == ["rollout-0"]


def _on_demand_launch(tmp_path: Path) -> tuple[Any, RayGenerationLaunchInputs, RolePlacement]:
    """A tiny SANA run whose rollout shares the trainer's one GPU, resolved for real.

    The rollout bundle is a plain placement value: creating the real GPU
    placement group would probe the card from a Ray actor, which a test
    outside the gpu lane must not do, and a deferred runtime binds the
    placement without using it until activation.
    """

    resolved = run.resolve_online_run(
        tiny_sana_online_config(tmp_path, overrides=("distributed.resources.rollout.num_gpus=1",))
    )
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    placement = RolePlacement(
        placement_group=None,
        bundle_indices=(0,),
        expected_gpu_ids=tuple(resolved.resources.rollout_devices),
    )
    return resolved, resolved.ray_launch_inputs(replay), placement


@pytest.mark.asyncio
async def test_create_runtime_defers_on_demand_topology_launch(
    monkeypatch, tmp_path, cuda_devices
) -> None:
    """A rollout sharing the trainer's GPU gets a deferred session: nothing
    launches until the schedule activates it."""

    cuda_devices(1)
    resolved, launch_inputs, placement = _on_demand_launch(tmp_path)
    launched: list[Any] = []
    real_launch = RayGenerationLauncher._launch_session

    def launch(self: RayGenerationLauncher, *args: Any, **kwargs: Any) -> Any:
        launched.append(args)
        return real_launch(self, *args, **kwargs)

    monkeypatch.setattr(RayGenerationLauncher, "_launch_session", launch)

    runtime = RayGenerationLauncher().create_runtime(
        resolved.generation, launch_inputs, placement=placement
    )

    assert resolved.resources.lifecycle.rollout_mode == "on_demand"
    assert runtime._session is None
    assert runtime._session_factory is not None
    assert launched == []
    await runtime.shutdown()
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_deferred_activation_reuses_factory_launcher(
    monkeypatch, tmp_path, cuda_devices
) -> None:
    """Activation launches through the launcher's own async path, passing on
    unchanged the shared-GPU sleep-offload contract the resolved on-demand
    topology already carries.

    The launch itself is stopped at that boundary: it would place a GPU actor,
    which a test outside the gpu lane must not do.
    """

    cuda_devices(1)
    resolved, launch_inputs, placement = _on_demand_launch(tmp_path)
    calls: list[tuple[Any, RayGenerationLaunchInputs, Any]] = []

    async def launch_async(
        self: RayGenerationLauncher,
        config: Any,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: Any,
    ) -> Any:
        calls.append((config, launch_inputs, placement))
        raise RuntimeError("deferred launch stopped at the GPU boundary")

    monkeypatch.setattr(RayGenerationLauncher, "_launch_session_async", launch_async)
    runtime = RayGenerationLauncher().create_runtime(
        resolved.generation, launch_inputs, placement=placement
    )

    with pytest.raises(RuntimeError, match="deferred launch stopped"):
        await runtime.activate()

    ((config, launched_inputs, launched_placement),) = calls
    assert config is resolved.generation
    assert launch_inputs.launch_contract.sleep_offload is True
    assert launched_inputs is launch_inputs
    assert launched_placement is placement
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
