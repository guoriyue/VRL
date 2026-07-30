"""Ray generation launcher integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkResult
from vrl.generation.ray.config import RayGenerationConfig, RolloutWorkerConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.on_demand_runtime import _OnDemandRayGenerationRuntime
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow

# These build real Ray workers on the package cluster — slow by nature, nightly.
pytestmark = pytest.mark.slow_test


class _Gatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=list(chunks),
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
        "chunk_placement_strategy": "round_robin",
        "sync_trainable_state": False,
    }
    values.update(overrides)
    return RolloutWorkerConfig(**values)


def test_ray_generation_launcher_builds_worker_runtime_with_embedded_ray(local_ray) -> None:
    """Launcher builds real workers into the owner's placement group."""
    ray = local_ray
    import vrl.generation.ray.launcher as launcher_mod

    worker = _worker_config(chunk_placement_strategy="dynamic")
    owner = _cpu_rollout_owner(ray, worker=worker)
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                resources=owner.resources,
                worker=worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        assert isinstance(runtime, RayGenerationRuntime)
        assert runtime.current_policy_version == 7
        assert runtime.weight_sync is None
        # Launcher uses the owner's group; it does not own/remove it.
        assert not hasattr(runtime, "_placement_group")
        # Config-selected placement strategy must reach the live planner.
        assert runtime.executor.planner.policy.strategy == "dynamic"

        workers = runtime.executor.workers
        assert [worker.worker_id for worker in workers] == ["rollout-0"]
        assert workers[0].actor is not None
        metadata = ray.get(workers[0].actor.worker_metadata.remote())
        assert metadata["worker_id"] == "rollout-0"
        assert metadata["policy_version"] == 7
    finally:
        # The cluster is shared with the rest of this package: release the
        # workers and the placement group, never the cluster.
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()


def _cpu_rollout_owner(
    ray: Any,
    *,
    worker: RolloutWorkerConfig | None = None,
) -> Any:
    """Build a GlobalRayPlacementOwner with a single CPU rollout bundle."""
    from omegaconf import OmegaConf

    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import resolve_distributed_resources

    resolved = resolve_distributed_resources(
        OmegaConf.create(
            {
                "distributed": {
                    "resources": {
                        "visible_devices": [],
                        "trainer": {"num_gpus": 0},
                        "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 1},
                    },
                    "rollout": {},
                    "reward": {},
                },
            },
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
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                resources=owner.resources,
                worker=owner.rollout_worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        # Runtime owns its workers but not the owner-managed placement group.
        assert not hasattr(runtime, "_placement_group")
        assert not hasattr(runtime, "_owned_actors")
        assert [w.worker_id for w in runtime.executor.workers] == ["rollout-0"]

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
        runtime = launcher_mod.RayGenerationLauncher(init_ray=False).launch(
            RayGenerationConfig(
                resources=owner.resources,
                worker=owner.rollout_worker,
            ),
            _launch_inputs(),
            placement=owner.rollout_placement,
        )

        assert runtime.is_colocated() is False
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
            rollout=replace(
                owner.resources.lifecycle.rollout,
                mode="on_demand",
            ),
        ),
    )
    runtime = _OnDemandRayGenerationRuntime(
        RayGenerationConfig(
            resources=on_demand_resources,
            worker=owner.rollout_worker,
        ),
        _launch_inputs(),
        launcher=RayGenerationLauncher(init_ray=False),
        placement=owner.rollout_placement,
    )
    try:
        # Explicit activation launches workers; offload parks them in place.
        asyncio.run(runtime.activate())
        inner = runtime._inner_runtime
        assert inner is not None
        assert not hasattr(inner, "_placement_group")  # inner never owns the PG
        first_actor = inner.executor.workers[0].actor
        asyncio.run(runtime.offload())
        assert runtime._inner_runtime is inner
        assert runtime._workers_offloaded is True
        # The owner's placement group is untouched and activation wakes in place.
        assert owner._placement_group is not None
        asyncio.run(runtime.activate())
        reacquired = runtime._inner_runtime
        assert reacquired is not None
        assert [w.worker_id for w in reacquired.executor.workers] == ["rollout-0"]
        assert reacquired is inner
        assert reacquired.executor.workers[0].actor is first_actor
    finally:
        asyncio.run(runtime.shutdown())
        owner.shutdown()
