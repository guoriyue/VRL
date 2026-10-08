"""Commit-ACK tests for generation worker weight synchronization.

Every fleet here is real: ``RayGenerationWorker`` actors serving tiny SANA on
the package cluster, launched by the real launcher (``ray_sana_runtime``),
with the real executor, dispatcher and weight sync. Payloads are the trainer's
real trainable-state export. An actor is kept busy with a ``"slow"`` request,
which ``SlowGenerationWorker`` holds for a fixed second before generating, so
its slot stays taken for a bound that does not depend on the host's speed;
submissions, deadlines, object-store puts and ref cancellations are observed
by recording the real calls.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

import vrl.ray.actor_pool as actor_pool_module
from tests.generation.ray._helpers import RaySanaRuntime, install_slow_workers, ray_sana_runtime
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import GenerationBatchEnvelope
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.ray.actor_pool import RayActorJob
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout
from vrl.utils.lifecycle import RuntimePhase

_TWO_ENGINES = ("distributed.resources.rollout.num_engines=2",)
# A continuous LoRA run keeps versioned slots, so a request may keep running
# against its own version while a newer one installs.
_VERSIONED = (
    "model.use_lora=true",
    "/base/rollout/orchestration=continuous",
    "trainer.rollout_orchestration.continuous.max_stale_policy_versions=1",
)


def _record_submissions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, tuple[Any, ...], Any]]:
    """Record every engine call (engine, method, args, ref) as it reaches the actor."""

    log: list[tuple[str, str, tuple[Any, ...], Any]] = []
    real_remote = RayGenerationEngine.remote

    def remote(self: RayGenerationEngine, method_name: str, *, combine: Any = None) -> Any:
        submit = real_remote(self, method_name, combine=combine)

        def submitted(*args: Any, **kwargs: Any) -> Any:
            ref = submit(*args, **kwargs)
            log.append((self.engine_id, method_name, args, ref))
            return ref

        return submitted

    monkeypatch.setattr(RayGenerationEngine, "remote", remote)
    return log


def _record(monkeypatch: pytest.MonkeyPatch, target: Any, name: str, log: list[Any]) -> None:
    """Record every call of ``target.name`` (args and kwargs), then run it."""

    real = getattr(target, name)

    def recorded(*args: Any, **kwargs: Any) -> Any:
        log.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, recorded)


def _methods(log: list[tuple[str, str, tuple[Any, ...], Any]]) -> list[str]:
    return [method for _, method, _, _ in log]


@pytest.fixture(autouse=True)
def _slow_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    install_slow_workers(monkeypatch)


def _busy_request(ray_run: RaySanaRuntime, **kwargs: Any) -> Any:
    return ray_run.request(["slow"], group_size=1, **kwargs)


def _envelope(request: Any) -> GenerationBatchEnvelope:
    (batch,) = EnginePlan.from_request(request).sample_batches
    return GenerationBatchEnvelope(request=request, batch=batch)


@pytest.mark.asyncio
async def test_invalid_policy_version_does_not_terminalize_resident_runtime(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        payload = ray_run.trainable_state()
        await runtime.update_weights(payload, 3)
        submissions = _record_submissions(monkeypatch)

        for version in (object(), 3.9, "3", True, -1):
            with pytest.raises(ValueError, match="policy_version"):
                await runtime.update_weights(payload, policy_version=version)

        assert submissions == []
        assert runtime.current_policy_version == 3
        assert runtime.lifecycle.failure is None
        assert runtime.lifecycle.phase is RuntimePhase.RUNNING
        # The fleet still serves the version it last installed.
        output = await runtime.generate(ray_run.request(["a cat"]))
        assert output.output.shape[0] == 2


@pytest.mark.asyncio
async def test_remote_update_puts_the_state_once_for_every_engine(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """One object-store put reaches both worker processes, which install it."""

    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        session = ray_run.runtime._session
        payload = ray_run.trainable_state()
        puts: list[Any] = []
        _record(monkeypatch, local_ray, "put", puts)
        submissions = _record_submissions(monkeypatch)

        await session.weight_sync.push_to_rollout_engines(payload, policy_version=4)

        ((put_args, _),) = puts
        assert put_args[0] is payload
        # Both workers were handed the one put's ObjectRef, not a copy each.
        (first_args, second_args) = [args for _, _, args, _ in submissions]
        assert [engine for engine, _, _, _ in submissions] == ["rollout-0", "rollout-1"]
        assert isinstance(first_args[0], local_ray.ObjectRef)
        assert first_args[0] is second_args[0]
        # Each worker dereferenced the shared ref into the real state and now
        # serves version 4: a batch stamped 4 runs on both.
        envelope = _envelope(ray_run.request(["a cat"], group_size=1, policy_version=4))
        results = local_ray.get(
            [engine.primary.actor.execute_batch.remote(envelope) for engine in session.engines],
            timeout=60,
        )
        assert [result.error for result in results] == [None, None]


@pytest.mark.asyncio
async def test_remote_update_timeout_rejects_partial_ack_and_cancels_every_ref(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        session = ray_run.runtime._session
        stalled = session.engines[1]
        # Another caller holds rollout-1's actor with a long generation, so its
        # install queues behind it while rollout-0 acknowledges.
        busy = stalled.primary.actor.execute_batch.remote(_envelope(_busy_request(ray_run)))
        sync = RayGenerationWeightSync(
            session.engines,
            actor_dispatcher=session.executor.actor_dispatcher,
            worker_rpc_timeout_s=0.5,
        )
        submissions = _record_submissions(monkeypatch)
        cancelled: list[Any] = []
        _record(monkeypatch, local_ray, "cancel", cancelled)

        with pytest.raises(RayOperationTimeout, match=r"rollout\.weight_sync"):
            await sync.push_to_rollout_engines(ray_run.trainable_state(), policy_version=7)

        refs = {engine: ref for engine, _, _, ref in submissions}
        assert local_ray.get(refs["rollout-0"], timeout=30) == 7
        assert ((refs["rollout-1"],), {"force": False}) in cancelled
        assert all(kwargs == {"force": False} for _, kwargs in cancelled)
        local_ray.get(busy, timeout=60)


@pytest.mark.asyncio
async def test_weight_sync_gets_a_full_deadline_after_shared_worker_admission(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A healthy generation call may outlive the weight ACK timeout: the ACK
    deadline starts only once the sync owns the worker's slot."""

    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        session = ray_run.runtime._session
        deadlines: list[tuple[str, float]] = []
        real_deadline = actor_pool_module.RayCallDeadline

        def recording_deadline(operation: str, timeout_s: float, **kwargs: Any) -> Any:
            deadlines.append((operation, timeout_s))
            return real_deadline(operation, timeout_s, **kwargs)

        monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
        submissions = _record_submissions(monkeypatch)
        generation = asyncio.create_task(ray_run.runtime.generate(_busy_request(ray_run)))
        while "execute_batch" not in _methods(submissions):
            await asyncio.sleep(0.01)

        sync = RayGenerationWeightSync(
            session.engines,
            actor_dispatcher=session.executor.actor_dispatcher,
            worker_rpc_timeout_s=0.3,
        )
        started = time.monotonic()
        update = asyncio.create_task(
            sync.push_to_rollout_engines(ray_run.trainable_state(), policy_version=3)
        )
        await asyncio.sleep(0.5)

        assert not update.done()
        assert _methods(submissions) == ["execute_batch"]

        await generation
        await update
        assert time.monotonic() - started > 0.3
        assert _methods(submissions) == ["execute_batch", "update_weights"]
        stall_timeout = ray_run.resolved.generation.worker.generation_stall_timeout_s
        assert deadlines == [
            ("rollout.generation.batch", stall_timeout),
            ("rollout.weight_sync", 0.3),
        ]


@pytest.mark.asyncio
async def test_waiting_weight_sync_gets_fair_handoff_before_pending_chunks(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=(*_VERSIONED, "rollout.samples_per_generation_batch=1"),
    ) as ray_run:
        runtime = ray_run.runtime
        payload = ray_run.trainable_state()
        await runtime.update_weights(payload, 1)
        submissions = _record_submissions(monkeypatch)
        generation = asyncio.create_task(runtime.generate(ray_run.request(["slow"], group_size=3)))
        while "execute_batch" not in _methods(submissions):
            await asyncio.sleep(0.01)

        await runtime.update_weights(payload, 2)
        output = await generation

        # The waiting sync takes the actor's next slot, ahead of the request's
        # remaining batches, which finish on their own (version 1) slot.
        assert _methods(submissions) == [
            "execute_batch",
            "update_weights",
            "execute_batch",
            "execute_batch",
        ]
        assert output.output.shape[0] == 3
        assert runtime.current_policy_version == 2


@pytest.mark.asyncio
async def test_cancelling_weight_sync_before_submission_keeps_runtime_running(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        payload = ray_run.trainable_state()
        submissions = _record_submissions(monkeypatch)
        generation = asyncio.create_task(runtime.generate(_busy_request(ray_run)))
        while "execute_batch" not in _methods(submissions):
            await asyncio.sleep(0.01)

        waiting = asyncio.create_task(runtime.update_weights(payload, policy_version=4))
        await asyncio.sleep(0.2)
        waiting.cancel()

        with pytest.raises(asyncio.CancelledError) as caught:
            await waiting
        assert caught.value.__cause__ is None
        assert runtime.lifecycle.phase is RuntimePhase.RUNNING
        assert _methods(submissions) == ["execute_batch"]

        await generation
        await runtime.update_weights(payload, policy_version=4)
        assert runtime.current_policy_version == 4


@pytest.mark.asyncio
async def test_completed_weight_sync_wins_cancellation_and_publishes_version(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        runtime = ray_run.runtime
        await runtime.update_weights(ray_run.trainable_state(), 1)
        submissions = _record_submissions(monkeypatch)
        update = asyncio.create_task(runtime.update_weights(ray_run.trainable_state(), 2))
        while "update_weights" not in _methods(submissions):
            await asyncio.sleep(0)
        ((_, _, _, ack),) = submissions
        # The worker has acknowledged before the caller's cancellation arrives.
        await asyncio.to_thread(local_ray.wait, [ack], timeout=30)

        update.cancel()
        await update

        assert runtime.current_policy_version == 2
        assert runtime.lifecycle.phase is RuntimePhase.RUNNING


@pytest.mark.asyncio
async def test_cancelling_partially_completed_weight_sync_terminalizes_runtime(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        runtime = ray_run.runtime
        session = runtime._session
        dispatcher = session.executor.actor_dispatcher
        payload = ray_run.trainable_state()
        submissions = _record_submissions(monkeypatch)
        cancelled: list[Any] = []
        _record(monkeypatch, local_ray, "cancel", cancelled)
        busy_engine = session.engines[1]
        generation = asyncio.create_task(
            dispatcher.run(
                [
                    RayActorJob(
                        0,
                        busy_engine.engine_id,
                        busy_engine.execute_batch(),
                        _envelope(_busy_request(ray_run)),
                    )
                ],
                operation="rollout.generation.batch",
                call_timeout_s=60.0,
            )
        )
        while "execute_batch" not in _methods(submissions):
            await asyncio.sleep(0.01)
        ((_, _, _, busy_ref),) = submissions

        update = asyncio.create_task(runtime.update_weights(payload, 2))
        while "update_weights" not in _methods(submissions):
            await asyncio.sleep(0)
        completed = next(ref for _, method, _, ref in submissions if method == "update_weights")
        await asyncio.to_thread(local_ray.wait, [completed], timeout=30)
        while completed in dispatcher._active_refs:
            await asyncio.sleep(0)
        # rollout-0 acknowledged; rollout-1's install still waits for its slot.
        assert [engine for engine, method, _, _ in submissions if method == "update_weights"] == [
            "rollout-0"
        ]

        update.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await update

        assert isinstance(caught.value.__cause__, RayOperationCancelled)
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime.current_policy_version == 0
        cancelled_refs = [args[0] for args, _ in cancelled]
        assert busy_ref in cancelled_refs
        assert completed not in cancelled_refs
        generation.cancel()
        await asyncio.gather(generation, return_exceptions=True)


@pytest.mark.asyncio
async def test_session_does_not_coerce_invalid_policy_version(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        session = ray_run.runtime._session
        payload = ray_run.trainable_state()
        submissions = _record_submissions(monkeypatch)

        for policy_version in (True, 1.9, "1", -1):
            with pytest.raises(ValueError, match="policy_version must be"):
                await session.update_weights(payload, policy_version)

        assert submissions == []
