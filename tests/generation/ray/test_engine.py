"""Multi-rank fan-out/aggregate semantics of the driver-side engine.

Each engine here is assembled from real ``RayGenerationWorker`` actors that the
real launcher started over the tiny SANA snapshot: a CPU fleet cannot group
ranks (rank groups need GPUs), so each launched single-rank engine's actor
stands for one rank of the N=2/3 engine under test. The launcher starts a
subclass of the real worker (``_rank_worker``) that turns one named rank into
the failure a theorem needs -- a real executor error, torch's CUDA OOM error,
or a reply whose identity disagrees -- and otherwise runs unchanged.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vrl.generation.ray.launcher as launcher_module
from tests.generation.ray._helpers import RaySanaRuntime, ray_sana_runtime
from tests.rollouts.collector._helpers import Trace
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import (
    GenerationBatchEnvelope,
    GenerationBatchResult,
    RequestBatchOutOfMemory,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.ray.engine import EngineCallRef, RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_group import RayActorHandle

pytestmark = pytest.mark.slow_test

_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 4.00 GiB"
_VERSIONED = (
    "model.use_lora=true",
    "trainer.rollout_orchestration.schedule_mode=continuous",
    "trainer.rollout_orchestration.continuous.max_stale_policy_versions=1",
)


def _rank_worker(
    *, faulty: str = "rollout-1", fault: str | None = None
) -> type[RayGenerationWorker]:
    """The real worker; the rank named ``faulty`` turns ``fault`` on.

    ``decode`` / ``oom``: its executor's forward raises a real error / torch's
    CUDA OOM error; ``request_oom``: only inside the per-request loop;
    ``request_id`` / ``policy_version``: its batch reply disagrees on that
    field; ``snapshot``: its parking report names another worker.
    """

    class _RankWorker(RayGenerationWorker):
        def load_policy(self) -> None:
            super().load_policy()
            executor = self.core.executor
            if self.core.worker_id != faulty or "forward_batch" in vars(executor):
                return
            real_forward = executor.forward_batch

            def forward_batch(request: Any, batch: Any) -> Any:
                if fault == "decode":
                    raise ValueError("decode failed")
                if fault == "oom" or (fault == "request_oom" and self._in_request):
                    raise torch.cuda.OutOfMemoryError(_OOM_MESSAGE)
                return real_forward(request, batch)

            executor.forward_batch = forward_batch

        _in_request = False

        def execute_batch(self, envelope: Any) -> Any:
            result = super().execute_batch(envelope)
            if self.core.worker_id == faulty and fault == "request_id":
                result = replace(result, request_id="wrong")
            if self.core.worker_id == faulty and fault == "policy_version":
                result = replace(result, policy_version=8)
            return result

        def execute_request_batches(self, request: Any, engine_plan: Any) -> Any:
            self._in_request = True
            try:
                return super().execute_request_batches(request, engine_plan)
            finally:
                self._in_request = False

        def sleep(self) -> WorkerMemoryParkingSnapshot:
            snapshot = super().sleep()
            if self.core.worker_id == faulty and fault == "snapshot":
                snapshot = replace(snapshot, worker_id="someone-else")
            return snapshot

    return _RankWorker


@contextlib.asynccontextmanager
async def _ranks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    snapshot: Any,
    count: int,
    *,
    fault: str | None = None,
    overrides: tuple[str, ...] = (),
) -> AsyncIterator[tuple[RaySanaRuntime, list[RayActorHandle]]]:
    """Launch ``count`` real rank actors; yields the run and their handles."""

    monkeypatch.setattr(launcher_module, "RayGenerationWorker", _rank_worker(fault=fault))
    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        snapshot,
        overrides=(f"distributed.resources.rollout.num_engines={count}", *overrides),
    ) as run:
        await run.runtime.activate()
        yield run, [engine.primary for engine in run.runtime._session.executor.engines]


def _envelope(run: RaySanaRuntime, *, samples: int = 2, **request: Any) -> GenerationBatchEnvelope:
    generation_request = run.request(["p"], group_size=samples, **request)
    if generation_request.policy_version is None:
        generation_request = replace(generation_request, policy_version=0)
    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=samples)
    return GenerationBatchEnvelope(request=generation_request, batch=batch)


class _FailingSubmission:
    """The real actor handle, except that submitting ``method`` raises ``error``.

    A Ray handle builds a fresh method object per attribute read, so the fault
    sits on a thin proxy; every other call reaches the real actor.
    """

    def __init__(self, actor: Any, method: str, error: BaseException) -> None:
        self._actor = actor
        self._method = method
        self._error = error

    def __getattr__(self, name: str) -> Any:
        if name == self._method:

            def remote(*args: Any, **kwargs: Any) -> Any:
                raise self._error

            return SimpleNamespace(remote=remote)
        return getattr(self._actor, name)


@pytest.mark.asyncio
async def test_broadcast_submits_to_every_rank_and_returns_rank0(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 3) as (_run, handles):
        engine = RayGenerationEngine("engine-0", handles)

        ref = engine.remote("worker_metadata")()
        assert isinstance(ref, EngineCallRef)
        result = await ref

        assert result["worker_id"] == "rollout-0"


@pytest.mark.asyncio
async def test_single_rank_returns_the_raw_rank_ref(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 1) as (run, handles):
        engine = RayGenerationEngine("engine-0", handles)

        ref = engine.execute_batch()(_envelope(run))

        # No aggregate wrapper for the degenerate case: the rank's own ObjectRef,
        # so completion and cancellation timing are the rank's.
        assert isinstance(ref, local_ray.ObjectRef)
        result = await ref
        assert result.error is None
        assert result.worker_id == "rollout-0"


@pytest.mark.asyncio
# execute_batch submits through remote(); sleep (like wake) gathers directly.
@pytest.mark.parametrize("method", ["execute_batch", "sleep"])
@pytest.mark.parametrize("failed_rank", [0, 1])
async def test_partial_submission_cancels_owned_refs_and_reports_terminal_failure(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, method, failed_rank
) -> None:
    import vrl.generation.ray.engine as engine_module
    from vrl.runtime_errors import TerminalRuntimeError

    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 3) as (run, handles):
        failure = RuntimeError("rank submission failed")
        handles[failed_rank] = replace(
            handles[failed_rank],
            actor=_FailingSubmission(handles[failed_rank].actor, method, failure),
        )
        engine = RayGenerationEngine("engine-0", handles)
        cancels = Trace(monkeypatch)
        cancels.watch(engine_module, "cancel_ray_refs", "cancel")

        with pytest.raises(RuntimeError) as caught:
            if method == "execute_batch":
                engine.remote(method)(_envelope(run))
            else:
                await getattr(engine, method)()

        if failed_rank:
            assert isinstance(caught.value, TerminalRuntimeError)
            assert caught.value.__cause__ is failure
            ((_ray, submitted), *_rest) = [args for _, args in cancels.calls]
            assert len(submitted) == 1
        else:
            assert caught.value is failure
            assert cancels.events == []


@pytest.mark.asyncio
async def test_any_rank_failure_fails_the_engine_call_and_cancels_siblings(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    import vrl.generation.ray.engine as engine_module

    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 3) as (run, handles):
        cancels = Trace(monkeypatch)
        cancels.watch(engine_module, "cancel_ray_refs", "cancel")
        local_ray.kill(handles[1].actor, no_restart=True)
        engine = RayGenerationEngine("engine-0", handles)

        with pytest.raises(local_ray.exceptions.RayActorError):
            await engine.execute_batch()(_envelope(run))

        # Waiting only on rank 0 would HANG once ranks run collectives; the
        # aggregate surfaces the dead rank and cancels every sibling ref.
        ((_ray, refs),) = [args for _, args in cancels.calls]
        assert len(refs) == 3


@pytest.mark.asyncio
async def test_sleep_validates_and_aggregates_per_rank_snapshots(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 2) as (_run, handles):
        snapshots = await RayGenerationEngine("engine-0", handles).sleep()
        assert [snap.worker_id for snap in snapshots] == ["rollout-0", "rollout-1"]

    async with _ranks(
        monkeypatch, tmp_path / "mismatched", ray_sana_snapshot, 2, fault="snapshot"
    ) as (_run, handles):
        with pytest.raises(RuntimeError, match="mismatched rank memory-parking report"):
            await RayGenerationEngine("engine-0", handles).sleep()


def test_engine_requires_at_least_one_rank() -> None:
    with pytest.raises(ValueError, match="at least one rank"):
        RayGenerationEngine("engine-0", [])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["decode", "oom", "stale"])
async def test_generation_combiner_retains_nonprimary_error_payload(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, failure
):
    """A failing non-primary rank's result wins over the primary's success."""

    fault = None if failure == "stale" else failure
    overrides = _VERSIONED if failure == "stale" else ()
    async with _ranks(
        monkeypatch, tmp_path, ray_sana_snapshot, 2, fault=fault, overrides=overrides
    ) as (run, handles):
        if failure == "stale":
            # Only the primary installs version 1: the other rank holds no slot for it.
            await handles[0].actor.update_weights.remote(run.trainable_state(), 1)
            await handles[1].actor.update_weights.remote(run.trainable_state(), 2)
        envelope = _envelope(run, policy_version=1 if failure == "stale" else None)

        result = await RayGenerationEngine("engine-0", handles).execute_batch()(envelope)

        assert result.worker_id == "rollout-1"
        assert result.output is None
        if failure == "decode":
            assert "decode failed" in result.error
        elif failure == "oom":
            assert "out of memory" in result.error
        else:
            assert result.stale_slot is True


def test_generation_combiner_prioritizes_terminal_errors_over_retry_and_discard():
    batch = GenerationSampleBatch(0, 0, 2)
    oom = GenerationBatchResult("r", "r0", batch, None, error="CUDA out of memory")
    stale = GenerationBatchResult("r", "r1", batch, None, error="evicted", stale_slot=True)
    terminal = GenerationBatchResult("r", "r2", batch, None, error="decode failed")
    assert GenerationBatchResult.from_rank_results([oom, stale, terminal]) is terminal
    assert GenerationBatchResult.from_rank_results([oom, stale]) is stale


@pytest.mark.asyncio
async def test_pipeline_combiner_retains_nonprimary_oom_payload(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
):
    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 2, fault="request_oom") as (
        run,
        handles,
    ):
        engine = RayGenerationEngine("engine-0", handles)
        request = replace(run.request(["p"], group_size=4), policy_version=0)
        plan = EnginePlan(
            sample_batches=(GenerationSampleBatch(0, 0, 2), GenerationSampleBatch(0, 2, 2))
        )

        result = await engine.remote(
            "execute_request_batches",
            combine=RayGenerationExecutor._select_request_rank_result,
        )(request, plan)

        assert isinstance(result, RequestBatchOutOfMemory)
        assert result.worker_id == "rollout-1"


@pytest.mark.parametrize("field", ["request_id", "batch", "policy_version"])
def test_generation_combiner_rejects_rank_identity_disagreement(field):
    good = GenerationBatchResult("r", "r0", GenerationSampleBatch(0, 0, 1), {}, policy_version=1)
    values = {"request_id": "other", "batch": GenerationSampleBatch(0, 1, 1), "policy_version": 2}
    other = replace(good, worker_id="r1", **{field: values[field]})
    with pytest.raises(RuntimeError, match="engine ranks returned different"):
        GenerationBatchResult.from_rank_results([good, other])
    good = replace(good, rank_metrics={"r0": {"peak_memory_mb": 10}})
    other = replace(good, worker_id="r1", rank_metrics={"r1": {"peak_memory_mb": 20}})
    result = GenerationBatchResult.from_rank_results([good, other])
    assert result.worker_id == good.worker_id
    assert result.output is good.output
    assert result.rank_metrics == {"r0": {"peak_memory_mb": 10}, "r1": {"peak_memory_mb": 20}}
    assert good.rank_metrics == {"r0": {"peak_memory_mb": 10}}


@pytest.mark.asyncio
async def test_batch_combines_metrics_from_every_rank_without_mutating_primary(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 2) as (run, handles):
        primary_ref = handles[0].actor.execute_batch.remote(_envelope(run, runtime_debug=True))
        primary = await primary_ref

        result = await RayGenerationEngine("engine-0", handles).execute_batch()(
            _envelope(run, runtime_debug=True)
        )

        assert set(result.rank_metrics) == {"rollout-0", "rollout-1"}
        assert result.worker_id == "rollout-0"
        assert set(primary.rank_metrics) == {"rollout-0"}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["request_id", "policy_version"])
async def test_batch_rejects_inconsistent_rank_identity(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, field: str
) -> None:
    async with _ranks(monkeypatch, tmp_path, ray_sana_snapshot, 2, fault=field) as (run, handles):
        with pytest.raises(RuntimeError, match="engine ranks returned different"):
            await RayGenerationEngine("engine-0", handles).execute_batch()(_envelope(run))
