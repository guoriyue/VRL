"""Ray generation launcher integration tests.

Every launch is the online recipe's own: the tiny SANA run resolved from its
config, the run-level placement group, ``RayGenerationLauncher.create_runtime``
starting real ``RayGenerationWorker`` actors over the package's tiny snapshot.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

from tests.generation.ray._helpers import ray_sana_runtime
from vrl.config.builders import BuiltConfigs
from vrl.generation.ray.config import RolloutWorkerConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.resources import ResourcePlan

# These build real Ray workers on the package cluster — slow by nature, nightly.
pytestmark = pytest.mark.slow_test


@contextlib.asynccontextmanager
async def _launched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    snapshot: Any,
    *,
    plan: Callable[[Any], Any] = lambda resources: resources,
    overrides: tuple[str, ...] = (),
) -> AsyncIterator[tuple[RayGenerationRuntime, Any, Any]]:
    """Launch exactly as the recipe does, with ``plan`` applied to the resource
    plan the launcher receives; yields (runtime, placement owner, resources)."""

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl.ray.placement import GlobalRayPlacementOwner

    cfg = tiny_sana_online_config(
        tmp_path,
        snapshot=snapshot,
        overrides=("distributed.rollout.cpus_per_worker=0.5", *overrides),
    )
    TinySanaPipeline().install(monkeypatch, snapshot)
    built = BuiltConfigs.from_cfg(cfg)
    resources = ResourcePlan.from_root(built.root)
    replay_build = built.family.resolve_model_build(
        built.root,
        torch.device(resources.trainer_torch_device),
        precision=built.precision,
        for_rollout=False,
    )
    replay_identity = replay_build.checkpoint_identity()
    owner = GlobalRayPlacementOwner(resources, RolloutWorkerConfig.from_root(built.root))
    owner.create()
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = RayGenerationLauncher().create_runtime(
            plan(resources),
            RolloutWorkerConfig.from_root(built.root),
            RayGenerationLaunchInputs.from_run(built, resources, replay_build, replay_identity),
            placement=owner.rollout_placement,
        )
        yield runtime, owner, resources
    finally:
        # The cluster is shared with the rest of this package: release the
        # workers and the placement group, never the cluster.
        if runtime is not None:
            await runtime.shutdown()
        owner.shutdown()


@pytest.mark.asyncio
async def test_ray_generation_launcher_builds_worker_runtime_with_embedded_ray(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Launcher builds real workers into the owner's placement group."""

    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime

        assert isinstance(runtime, RayGenerationRuntime)
        # The launch config's version: the checkpoint the run starts from.
        assert runtime.current_policy_version == 0
        session = runtime._session
        assert session is not None
        engines = session.executor.engines
        assert [engine.engine_id for engine in engines] == ["rollout-0"]
        metadata = local_ray.get(engines[0].ranks[0].actor.worker_metadata.remote())
        assert metadata["worker_id"] == "rollout-0"
        assert metadata["gpu_ids"] == []
        assert "policy_version" not in metadata


def test_create_runtime_rejects_missing_rollout_placement(monkeypatch, tmp_path) -> None:
    """placement=None fails fast with the config knob, not an AttributeError."""

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config

    cfg = tiny_sana_online_config(tmp_path)
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    built = BuiltConfigs.from_cfg(cfg)
    resources = ResourcePlan.from_root(built.root)
    replay_build = built.family.resolve_model_build(
        built.root,
        torch.device(resources.trainer_torch_device),
        precision=built.precision,
        for_rollout=False,
    )
    replay_identity = replay_build.checkpoint_identity()
    with pytest.raises(ValueError, match="rollout placement"):
        RayGenerationLauncher().create_runtime(
            resources,
            RolloutWorkerConfig.from_root(built.root),
            RayGenerationLaunchInputs.from_run(built, resources, replay_build, replay_identity),
            placement=None,
        )


@pytest.mark.asyncio
async def test_owner_placement_runtime_does_not_own_placement_group(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Persistent runtime built on owner placement must not own/remove the PG."""

    async with _launched(monkeypatch, tmp_path, ray_sana_snapshot) as (runtime, owner, _):
        session = runtime._session
        assert session is not None
        assert [e.engine_id for e in session.executor.engines] == ["rollout-0"]

        # Tearing down the runtime kills workers but leaves the owner's PG alive.
        await runtime.shutdown()
        assert owner._placement_group is not None


@pytest.mark.asyncio
async def test_phase_handoff_keeps_actor_and_owner_placement(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A shared-GPU handoff parks its actor without dropping the owner PG."""

    def shared_gpu(resources: Any) -> Any:
        # A CPU host has no trainer GPU; pin both roles to one card to force the lease.
        return replace(
            resources,
            lifecycle=replace(resources.lifecycle, trainer=(0,), rollout=(0,)),
        )

    async with _launched(monkeypatch, tmp_path, ray_sana_snapshot, plan=shared_gpu) as (
        runtime,
        owner,
        _,
    ):
        # Explicit activation launches workers; offload parks them in place.
        await runtime.activate()
        session = runtime._session
        assert session is not None
        first_actor = session.executor.engines[0].ranks[0].actor
        await runtime.offload()
        assert runtime._session is session
        assert runtime._session_parked is True
        # The owner's placement group is untouched and activation wakes in place.
        assert owner._placement_group is not None
        await runtime.activate()
        reacquired = runtime._session
        assert reacquired is session
        assert [e.engine_id for e in reacquired.executor.engines] == ["rollout-0"]
        assert reacquired.executor.engines[0].ranks[0].actor is first_actor
