"""The driver-side engine vouches for its rank's parking report.

The engine is assembled from a real ``RayGenerationWorker`` the real launcher
started over the tiny SANA snapshot; ``remote()`` is the rank's own ``.remote``
and is exercised by every executor test, so only ``sleep``'s identity check
lives here.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

import vrl.generation.ray.launcher as launcher_module
from tests.generation.ray._helpers import RaySanaRuntime, ray_sana_runtime
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import GenerationBatchEnvelope, WorkerMemoryParkingSnapshot
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_group import RayActorHandle

pytestmark = pytest.mark.slow_test


class _MisreportingWorker(RayGenerationWorker):
    """The real worker, except that its parking report names another worker."""

    def sleep(self) -> WorkerMemoryParkingSnapshot:
        return replace(super().sleep(), worker_id="someone-else")


@contextlib.asynccontextmanager
async def _rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    snapshot: Any,
    *,
    worker: type[RayGenerationWorker] = RayGenerationWorker,
) -> AsyncIterator[tuple[RaySanaRuntime, RayActorHandle]]:
    """Launch one real rank actor; yields the run and its handle."""

    monkeypatch.setattr(launcher_module, "RayGenerationWorker", worker)
    async with ray_sana_runtime(monkeypatch, tmp_path, snapshot) as run:
        await run.runtime.activate()
        (engine,) = run.runtime._session.executor.engines
        yield run, engine.rank


def _envelope(run: RaySanaRuntime) -> GenerationBatchEnvelope:
    generation_request = run.request(["p"], group_size=2)
    if generation_request.policy_version is None:
        generation_request = replace(generation_request, policy_version=0)
    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=2)
    return GenerationBatchEnvelope(request=generation_request, batch=batch)


@pytest.mark.asyncio
async def test_sleep_validates_the_rank_snapshot(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _rank(monkeypatch, tmp_path, ray_sana_snapshot) as (_run, handle):
        snapshot = await RayGenerationEngine("engine-0", handle).sleep()
        assert snapshot.worker_id == "rollout-0"

    async with _rank(
        monkeypatch, tmp_path / "mismatched", ray_sana_snapshot, worker=_MisreportingWorker
    ) as (_run, handle):
        with pytest.raises(RuntimeError, match="mismatched rank memory-parking report"):
            await RayGenerationEngine("engine-0", handle).sleep()
