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

from tests.generation.ray._helpers import ray_sana_runtime
from vrl.generation.ray.config import RolloutWorkerConfig
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.types import GenerationOutput

# These build real Ray workers on the package cluster — slow by nature, nightly.
pytestmark = pytest.mark.slow_test


@pytest.mark.parametrize("section", [False, 0, "", []])
def test_worker_section_does_not_treat_invalid_values_as_absent(section) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RolloutWorkerConfig.from_public_section(section)


def test_worker_section_defaults_only_for_absent_or_empty_mapping() -> None:
    assert RolloutWorkerConfig.from_public_section(
        None
    ) == RolloutWorkerConfig.from_public_section({})


@contextlib.asynccontextmanager
async def _launched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    snapshot: Any,
    *,
    generation: Callable[[Any], Any] = lambda config: config,
    overrides: tuple[str, ...] = (),
) -> AsyncIterator[tuple[RayGenerationRuntime, Any, Any]]:
    """Launch exactly as the recipe does, with ``generation`` applied to the
    resolved generation config first; yields (runtime, placement owner, resolved)."""

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl import run
    from vrl.ray.placement import GlobalRayPlacementOwner

    cfg = tiny_sana_online_config(
        tmp_path,
        snapshot=snapshot,
        overrides=("distributed.rollout.cpus_per_worker=0.5", *overrides),
    )
    TinySanaPipeline().install(monkeypatch, snapshot)
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    owner = GlobalRayPlacementOwner(resolved.resources, resolved.generation.worker)
    owner.create()
    runtime: RayGenerationRuntime | None = None
    try:
        runtime = RayGenerationLauncher().create_runtime(
            generation(resolved.generation),
            resolved.ray_launch_inputs(replay),
            placement=owner.rollout_placement,
        )
        yield runtime, owner, resolved
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
        # The launch contract's version: the checkpoint the run starts from.
        assert runtime.current_policy_version == 0
        session = runtime._session
        assert session is not None
        assert session.weight_sync is not None
        engines = session.executor.engines
        assert [engine.engine_id for engine in engines] == ["rollout-0"]
        metadata = local_ray.get(engines[0].primary.actor.worker_metadata.remote())
        assert metadata["worker_id"] == "rollout-0"
        assert metadata["gpu_ids"] == []
        assert "policy_version" not in metadata


@pytest.mark.asyncio
async def test_pipelined_runtime_stages_batches_and_merges_on_the_finalizer(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Per-request path end to end on real actors: the rank stages each batch in
    the object store, the finalizer merges the references, and the driver gets
    the gathered output in plan order."""

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=("distributed.rollout.pipelined=true", "rollout.samples_per_generation_batch=2"),
    ) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        assert session is not None
        assert [handle.worker_id for handle in session.finalizer_handles] == [
            "rollout-0.finalize",
        ]
        assert session.executor.finalizers == tuple(session.finalizer_handles)
        request = ray_run.request(["p"], group_size=4)

        await runtime.activate()
        output = await runtime.generate(request)

        assert isinstance(output, GenerationOutput)
        assert output.request_id == request.request_id
        assert output.output.shape[0] == 4
        assert [row.sample_index for row in output.sample_rows] == [0, 1, 2, 3]
        assert output.trajectory.request_id == request.request_id


def test_create_runtime_rejects_missing_rollout_placement(monkeypatch, tmp_path) -> None:
    """placement=None fails fast with the config knob, not an AttributeError."""

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl import run

    cfg = tiny_sana_online_config(tmp_path)
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    with pytest.raises(ValueError, match="rollout placement"):
        RayGenerationLauncher().create_runtime(
            resolved.generation,
            resolved.ray_launch_inputs(replay),
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

    def shared_gpu(generation: Any) -> Any:
        # A CPU host has no trainer GPU; pin both roles to one card to force the lease.
        resources = generation.resources
        return replace(
            generation,
            resources=replace(
                resources,
                lifecycle=replace(resources.lifecycle, trainer=(0,), rollout=(0,)),
            ),
        )

    async with _launched(monkeypatch, tmp_path, ray_sana_snapshot, generation=shared_gpu) as (
        runtime,
        owner,
        _,
    ):
        # Explicit activation launches workers; offload parks them in place.
        await runtime.activate()
        session = runtime._session
        assert session is not None
        first_actor = session.executor.engines[0].primary.actor
        await runtime.offload()
        assert runtime._session is session
        assert runtime._session_parked is True
        # The owner's placement group is untouched and activation wakes in place.
        assert owner._placement_group is not None
        await runtime.activate()
        reacquired = runtime._session
        assert reacquired is session
        assert [e.engine_id for e in reacquired.executor.engines] == ["rollout-0"]
        assert reacquired.executor.engines[0].primary.actor is first_actor


@pytest.mark.parametrize("query", ["current_node_ip", "current_gpu_ids"])
def test_worker_metadata_preserves_placement_query_failure(monkeypatch, tmp_path, query) -> None:
    """A failed placement query surfaces as itself from the real worker."""

    from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
    from vrl import run
    from vrl.generation.ray import worker

    cfg = tiny_sana_online_config(tmp_path)
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    actor = worker.RayGenerationWorker("rollout-0", resolved.ray_launch_inputs(replay))
    failure = RuntimeError("placement query failed")

    def fail():
        raise failure

    monkeypatch.setattr(worker, query, fail)
    with pytest.raises(RuntimeError, match="placement query failed") as caught:
        actor.worker_metadata()
    assert caught.value is failure


def test_cross_node_validation_preserves_driver_node_query_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    from vrl.generation.ray import launcher

    failure = RuntimeError("driver node query failed")

    def fail():
        raise failure

    monkeypatch.setattr(launcher, "current_node_ip", fail)
    config = SimpleNamespace(resources=SimpleNamespace(rollout_devices=(0,), cross_node=True))
    with pytest.raises(RuntimeError, match="driver node query failed") as caught:
        launcher.RayGenerationLauncher._validate_rank_gpu_ids(config, [], expected_gpu_ids=())
    assert caught.value is failure
