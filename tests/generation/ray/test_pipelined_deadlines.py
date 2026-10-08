"""Pipelined generation deadlines: admission first, then a budget per batch.

Every fleet is real: the real launcher starts ``RayGenerationWorker`` and
``RayGenerationFinalizer`` actors over the tiny SANA snapshot, and requests go
through ``RayGenerationRuntime.generate``. To make admission observable the
launcher starts subclasses of the real actors that take longer on a request
whose prompt is ``"slow"``; every other call runs unchanged. A spy around the
real ``RayCallDeadline`` records when each deadline starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

import vrl.generation.ray.launcher as launcher_module
import vrl.ray.actor_pool as actor_pool_module
from tests.generation.ray._helpers import (
    RaySanaRuntime,
    SlowGenerationWorker,
    install_slow_workers,
    ray_sana_runtime,
)
from vrl.generation.ray.finalizer import RayGenerationFinalizer

pytestmark = pytest.mark.slow_test

_SLOW_S = SlowGenerationWorker.HOLD_S


class _SlowMergeFinalizer(RayGenerationFinalizer):
    """The real finalizer; a ``"slow"`` request holds the finalizer for ``_SLOW_S``."""

    def merge_request(self, request: Any, batch_refs: Any) -> Any:
        if request.prompts == ["slow"]:
            time.sleep(_SLOW_S)
        return super().merge_request(request, batch_refs)


def _record_deadlines(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float, float]]:
    """(operation, budget, start time) of every real call deadline, in start order."""

    started: list[tuple[str, float, float]] = []
    real_deadline = actor_pool_module.RayCallDeadline

    def recording_deadline(operation: str, timeout_s: float, **kwargs: Any) -> Any:
        started.append((operation, float(timeout_s), time.perf_counter()))
        return real_deadline(operation, timeout_s, **kwargs)

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    return started


@contextlib.asynccontextmanager
async def _pipelined_fleet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, snapshot: Any, *, stall_timeout_s: float
) -> AsyncIterator[RaySanaRuntime]:
    install_slow_workers(monkeypatch)
    monkeypatch.setattr(launcher_module, "RayGenerationFinalizer", _SlowMergeFinalizer)
    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        snapshot,
        overrides=(
            "distributed.rollout.pipelined=true",
            f"distributed.rollout.generation_stall_timeout_s={stall_timeout_s}",
            "rollout.samples_per_generation_batch=2",
        ),
    ) as run:
        await run.runtime.activate()
        yield run


def _operations(started: list[tuple[str, float, float]]) -> list[str]:
    return [operation for operation, _, _ in started]


async def _wait_for(condition, timeout_s: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while not condition():
        assert time.perf_counter() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_pipelined_call_budget_is_the_stall_timeout_per_batch(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """One RPC produces every batch of the engine's share, so its budget covers each."""

    async with _pipelined_fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, stall_timeout_s=2.0
    ) as run:
        started = _record_deadlines(monkeypatch)

        output = await run.runtime.generate(run.request(["p"], group_size=6))

        assert output.output.shape[0] == 6
        pipelined = [(op, budget) for op, budget, _ in started if op.endswith(".pipelined")]
        assert pipelined == [("rollout.generation.pipelined", 6.0)]


@pytest.mark.asyncio
async def test_pipelined_submission_gets_deadline_only_after_fleet_admission(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _pipelined_fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, stall_timeout_s=_SLOW_S * 3
    ) as run:
        started = _record_deadlines(monkeypatch)
        # A one-batch request takes the per-batch path and holds the one engine.
        first = asyncio.create_task(run.runtime.generate(run.request(["slow"], group_size=2)))
        await _wait_for(lambda: "rollout.generation.batch" in _operations(started))

        pipelined = asyncio.create_task(run.runtime.generate(run.request(["p"], group_size=4)))
        await asyncio.sleep(_SLOW_S / 2)

        # Waiting for the engine is not on the clock: its deadline starts at admission.
        assert not pipelined.done()
        assert "rollout.generation.pipelined" not in _operations(started)

        assert (await first).output.shape[0] == 2
        assert (await pipelined).output.shape[0] == 4
        operations = _operations(started)
        assert operations.index("rollout.generation.batch") < operations.index(
            "rollout.generation.pipelined"
        )


@pytest.mark.asyncio
async def test_finalize_gets_its_deadline_only_after_a_finalizer_slot(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A request whose merge queues behind a slow one does not burn its budget
    waiting: the finalizer has one slot, FIFO waiters, and the deadline starts
    at admission, exactly like the engines."""

    async with _pipelined_fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, stall_timeout_s=5.0
    ) as run:
        started = _record_deadlines(monkeypatch)

        slow = asyncio.create_task(run.runtime.generate(run.request(["slow"], group_size=4)))
        queued = asyncio.create_task(run.runtime.generate(run.request(["queued"], group_size=4)))
        slow_output, queued_output = await asyncio.gather(slow, queued)

        assert slow_output.output.shape[0] == queued_output.output.shape[0] == 4
        finalize_starts = [at for op, _, at in started if op == "rollout.generation.finalize"]
        assert len(finalize_starts) == 2
        # The queued merge went on the clock only once the slow merge released
        # the finalizer's one slot.
        assert finalize_starts[1] - finalize_starts[0] >= _SLOW_S * 0.9
