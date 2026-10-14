"""Ray generation launcher integration tests.

Every launch is the online recipe's own: the tiny SANA run resolved from its
config, the run-level placement group, ``RayGenerationLauncher.create_runtime``
starting real ``RayGenerationWorker`` actors over the package's tiny snapshot.
"""

from __future__ import annotations

import pytest

from tests.generation.ray._helpers import ray_sana_runtime, run_launch_inputs
from vrl.config.builders import BuiltConfigs
from vrl.generation.ray.config import RolloutWorkerConfig
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.resources import ResourcePlan

# These build real Ray workers on the package cluster — slow by nature, nightly.
pytestmark = pytest.mark.slow_test


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
    with pytest.raises(ValueError, match="rollout placement"):
        RayGenerationLauncher().create_runtime(
            resources,
            RolloutWorkerConfig.from_root(built.root),
            run_launch_inputs(built, resources),
            placement=None,
        )


@pytest.mark.asyncio
async def test_owner_placement_runtime_does_not_own_placement_group(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Persistent runtime built on owner placement must not own/remove the PG."""

    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        assert session is not None
        assert [e.engine_id for e in session.executor.engines] == ["rollout-0"]

        # Tearing down the runtime kills workers but leaves the owner's PG alive.
        await runtime.shutdown()
        assert ray_run.placement_owner._placement_group is not None
