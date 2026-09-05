"""Ray generation launcher integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from vrl.config.schema import parse_config
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import BatchPayload
from vrl.generation.ray.config import RayGenerationConfig, RolloutWorkerConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.trajectory import TrajectoryBatch

# These build real Ray workers on the package cluster — slow by nature, nightly.
pytestmark = pytest.mark.slow_test


class _Gatherer:
    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        return GenerationOutput(
            output=list(batches),
            trajectory=TrajectoryBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                sample_rows=list(sample_rows),
                axes={},
                segments={},
            ),
        )


def _launch_inputs() -> RayGenerationLaunchInputs:
    return RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family="janus_pro",
            model_build={
                "model_name_or_path": "unit-test",
                "revision": None,
                "device": "cpu",
                "parameter_dtype": "float32",
                "precision": {
                    "dtype": "fp32",
                    "float32_precision": "tf32",
                    "quantization": None,
                    "outer_autocast": False,
                },
            },
            expected_model_identity={"schema": "test"},
            policy_version=7,
        ),
        gatherer=_Gatherer(),
    )


def _worker_config(**overrides: Any) -> RolloutWorkerConfig:
    values = {
        "cpus_per_worker": 0.5,
        "health_check_interval_s": 30.0,
        "health_check_timeout_s": 30.0,
        "health_check_first_wait_s": 0.0,
        "worker_rpc_timeout_s": 30.0,
        "generation_stall_timeout_s": 30.0,
        "pipelined": False,
        "batch_placement_strategy": "round_robin",
        "sync_trainable_state": False,
    }
    values.update(overrides)
    return RolloutWorkerConfig(**values)


def test_ray_generation_launcher_builds_worker_runtime_with_embedded_ray(local_ray) -> None:
    """Launcher builds real workers into the owner's placement group."""
    ray = local_ray
    import vrl.generation.ray.launcher as launcher_mod

    worker = _worker_config(batch_placement_strategy="dynamic")
    owner = _cpu_rollout_owner(ray, worker=worker)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).create_runtime(
            RayGenerationConfig(
                resources=owner.resources,
                worker=worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        assert isinstance(runtime, RayGenerationRuntime)
        assert runtime.current_policy_version == 7
        session = runtime._session
        assert session is not None
        assert session.weight_sync is None
        # Config-selected placement strategy must reach the live planner.
        assert session.executor.planner.strategy == "dynamic"

        engines = session.executor.engines
        assert [engine.engine_id for engine in engines] == ["rollout-0"]
        assert engines[0].primary.actor is not None
        metadata = ray.get(engines[0].primary.actor.worker_metadata.remote())
        assert metadata["worker_id"] == "rollout-0"
        assert "policy_version" not in metadata
    finally:
        # The cluster is shared with the rest of this package: release the
        # workers and the placement group, never the cluster.
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()


def test_create_runtime_rejects_missing_rollout_placement() -> None:
    """placement=None fails fast with the config knob, not an AttributeError."""
    from omegaconf import OmegaConf

    import vrl.generation.ray.launcher as launcher_mod
    from vrl.ray.resources import ResolvedDistributedResources

    resolved = ResolvedDistributedResources.resolve(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "resources": {
                            "visible_devices": [],
                            "trainer": {"num_gpus": 0},
                            "rollout": {"num_gpus": 0, "num_engines": 1},
                        },
                        "rollout": {},
                    },
                },
            )
        ),
    )
    with pytest.raises(ValueError, match="rollout placement"):
        launcher_mod.RayGenerationLauncher(init_ray=False).create_runtime(
            RayGenerationConfig(resources=resolved, worker=_worker_config()),
            _launch_inputs(),
            placement=None,
        )


def _cpu_rollout_owner(
    ray: Any,
    *,
    worker: RolloutWorkerConfig | None = None,
) -> Any:
    """Build a GlobalRayPlacementOwner with a single CPU rollout bundle."""
    from omegaconf import OmegaConf

    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import ResolvedDistributedResources

    resolved = ResolvedDistributedResources.resolve(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "resources": {
                            "visible_devices": [],
                            "trainer": {"num_gpus": 0},
                            "rollout": {"num_gpus": 0, "num_engines": 1},
                        },
                        "rollout": {},
                    },
                },
            )
        ),
    )
    owner = GlobalRayPlacementOwner(resolved, worker or _worker_config())
    owner.create()
    return owner


def test_owner_placement_runtime_does_not_own_placement_group(local_ray) -> None:
    """Persistent runtime built on owner placement must not own/remove the PG."""
    import vrl.generation.ray.launcher as launcher_mod

    owner = _cpu_rollout_owner(local_ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).create_runtime(
            RayGenerationConfig(
                resources=owner.resources,
                worker=owner.rollout_worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        session = runtime._session
        assert session is not None
        assert [e.engine_id for e in session.executor.engines] == ["rollout-0"]

        # Tearing down the runtime kills workers but leaves the owner's PG alive.
        asyncio.run(runtime.shutdown())
        runtime = None
        assert owner._placement_group is not None
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()


def test_launcher_uses_resolved_colocation_protocol_signal(local_ray) -> None:
    """Launcher derives the runtime colocation signal from resolved topology."""
    import vrl.generation.ray.launcher as launcher_mod

    owner = _cpu_rollout_owner(local_ray)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).create_runtime(
            RayGenerationConfig(
                resources=owner.resources,
                worker=owner.rollout_worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        assert runtime.requires_driver_model_offload is False
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()


def test_phase_handoff_keeps_actor_and_owner_placement(local_ray) -> None:
    """A shared-GPU handoff parks its actor without dropping the owner PG."""
    owner = _cpu_rollout_owner(local_ray)
    on_demand_resources = replace(
        owner.resources,
        lifecycle=replace(
            owner.resources.lifecycle,
            trainer_and_rollout_share_gpu=True,
        ),
    )
    runtime = RayGenerationLauncher(init_ray=False).create_runtime(
        RayGenerationConfig(
            resources=on_demand_resources,
            worker=owner.rollout_worker,
        ),
        _launch_inputs(),
        placement=owner.rollout_placement,
    )
    try:
        # Explicit activation launches workers; offload parks them in place.
        asyncio.run(runtime.activate())
        session = runtime._session
        assert session is not None
        first_actor = session.executor.engines[0].primary.actor
        asyncio.run(runtime.offload())
        assert runtime._session is session
        assert runtime._session_parked is True
        # The owner's placement group is untouched and activation wakes in place.
        assert owner._placement_group is not None
        asyncio.run(runtime.activate())
        reacquired = runtime._session
        assert reacquired is not None
        assert [e.engine_id for e in reacquired.executor.engines] == ["rollout-0"]
        assert reacquired is session
        assert reacquired.executor.engines[0].primary.actor is first_actor
    finally:
        asyncio.run(runtime.shutdown())
        owner.shutdown()
