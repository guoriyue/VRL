"""Explicit activation/offload of an on-demand Ray rollout runtime.

Every runtime here is real: the tiny SANA run resolved on the package cluster,
its run-level placement group, and ``RayGenerationWorker`` actors started by
the real launcher. The on-demand runtime is built the way the launcher's
on-demand branch builds it -- no session at construction, the launcher's own
``_launch_session_async`` as the cold-start factory. ``create_runtime`` picks
that branch only for a rollout that leases the trainer's GPU, which a CPU
cluster cannot resolve, and the branch's ``sleep_offload`` contract needs a
CuMem pool a CPU worker cannot have; the runtime's lease state machine reads
neither, so CPU workers park through their real move ledger.

Faults are real or injected onto real replies: a worker subclass the real
launcher starts (keeping the real actor body) returns a doctored parking
report, raises from ``sleep``/``wake``, or blocks past a tightened real park
deadline; an install of a mis-shaped real payload fails on the worker;
``Trace`` queues one-shot failures on real driver-side methods; awaited gates
around real methods order concurrent operations. Graceful release is observed
as the session's ``ray.get`` on the release refs, force-kill as dead actors.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

import vrl.generation.ray.launcher as launcher_module
import vrl.generation.ray.session as session_module
from tests.rollouts.collector._helpers import Trace
from tests.scripts.eval.fixtures import TinySanaPipeline, tiny_sana_online_config
from vrl import run
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.ray.placement import GlobalRayPlacementOwner
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.utils.deadline import OperationTimeout
from vrl.utils.lifecycle import RuntimeLifecycleError, RuntimePhase

_TWO_ENGINES = ("distributed.resources.rollout.num_engines=2",)


@dataclass
class _OnDemand:
    runtime: RayGenerationRuntime
    launcher: RayGenerationLauncher
    resolved: Any
    replay: Any

    def request(self, prompts: list[str], **kwargs: Any) -> Any:
        builder = GenerationRequestBuilder(
            entry=self.resolved.family, config=self.resolved.collector
        )
        return builder.build(prompts, 2, **kwargs).request

    def payload(self) -> dict[str, Any]:
        """A fresh copy of the trainer's real weight-sync payload."""

        bundle = self.replay.materialize(context="lease test weight payload")
        return SingleProcessStrategy().export_rollout_state(bundle)

    def bad_payload(self) -> dict[str, Any]:
        """The real payload with one tensor of the wrong shape: the install fails."""

        payload = self.payload()
        key = next(iter(payload))
        payload[key] = torch.zeros(1)
        return payload

    def actors(self) -> list[Any]:
        session = self.runtime._session
        assert session is not None
        return [rank.actor for rank in session.rank_handles]


@contextlib.asynccontextmanager
async def _on_demand(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot: Path,
    *,
    worker: type[RayGenerationWorker] | None = None,
    overrides: tuple[str, ...] = (),
) -> AsyncIterator[_OnDemand]:
    """An on-demand runtime over the real launcher; no session until ``activate``."""

    if worker is not None:
        monkeypatch.setattr(launcher_module, "RayGenerationWorker", worker)
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
    launcher = RayGenerationLauncher()
    launch_inputs = resolved.ray_launch_inputs(replay)

    async def launch_session() -> RayGenerationSession:
        # Looked up at call time, so a test may gate or fail the real launch.
        return await launcher._launch_session_async(
            resolved.generation,
            launch_inputs,
            placement=owner.rollout_placement,
        )

    runtime = RayGenerationRuntime(
        session=None,
        session_factory=launch_session,
        initial_policy_version=launch_inputs.launch_contract.policy_version,
    )
    try:
        yield _OnDemand(runtime=runtime, launcher=launcher, resolved=resolved, replay=replay)
    finally:
        with contextlib.suppress(BaseException):
            await runtime.shutdown()
        owner.shutdown()


def _parking_worker(
    *,
    sleep_reply: Callable[[Any], Any] | None = None,
    sleep_error: BaseException | None = None,
    sleep_error_on: frozenset[str] = frozenset(),
    wake_error: BaseException | None = None,
    block: frozenset[str] = frozenset(),
) -> type[RayGenerationWorker]:
    """The real Ray generation worker with a fault on its parking RPCs.

    ``sleep_reply`` rewrites the real parking report; ``sleep_error`` is raised
    by the workers named in ``sleep_error_on`` (all when empty); ``block``
    names the RPCs (``sleep``/``wake``) that never return in time.
    """

    class _ParkingWorker(RayGenerationWorker):
        def __init__(self, worker_id: str, launch_inputs: Any) -> None:
            super().__init__(worker_id, launch_inputs)
            self.sleep_calls = 0

        def sleep(self) -> Any:
            self.sleep_calls += 1
            if "sleep" in block:
                time.sleep(600)
            if sleep_error is not None and (
                not sleep_error_on or self.core.worker_id in sleep_error_on
            ):
                raise sleep_error
            snapshot = super().sleep()
            return snapshot if sleep_reply is None else sleep_reply(snapshot)

        def wake(self) -> None:
            if "wake" in block:
                time.sleep(600)
            if wake_error is not None:
                raise wake_error
            super().wake()

        def parked_count(self) -> int:
            return self.sleep_calls

    return _ParkingWorker


def _is_dead(ray: Any, actor: Any, *, settle_s: float = 10.0) -> bool:
    """Whether the actor stops answering; ``ray.kill`` takes effect asynchronously."""

    deadline = time.monotonic() + settle_s
    while True:
        try:
            ray.get(actor.worker_metadata.remote(), timeout=30)
        except ray.exceptions.RayActorError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _watch_release(monkeypatch: pytest.MonkeyPatch, ray: Any) -> Trace:
    """The graceful release waits on the release refs with ``ray.get``; a
    force-close never submits them."""

    release = Trace(monkeypatch)
    release.watch(ray, "get", "ray.get")
    return release


def _gate_class(
    monkeypatch: pytest.MonkeyPatch,
    cls: type,
    method: str,
    *,
    before: Callable[..., Any],
) -> None:
    """Await ``before(self, *args)`` ahead of the real coroutine method of ``cls``."""

    real = getattr(cls, method)

    async def gated(self: Any, *args: Any, **kwargs: Any) -> Any:
        await before(self, *args, **kwargs)
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(cls, method, gated)


# -------------------------------------------------- worker parking evidence


@pytest.mark.asyncio
async def test_offload_accepts_complete_worker_parking_evidence(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        worker=_parking_worker(),
        overrides=_TWO_ENGINES,
    ) as lease:
        runtime = lease.runtime
        await runtime.activate()
        actors = lease.actors()

        await runtime.offload()

        assert runtime._session_parked is True
        assert [local_ray.get(actor.parked_count.remote()) for actor in actors] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "error_type", "message"),
    [
        (
            lambda snapshot: replace(snapshot, worker_id="another-worker"),
            RuntimeError,
            "mismatched rank memory-parking report",
        ),
        # A CPU worker parks without GPU memory; the residual a CUDA rank
        # could leave behind is put onto its real report.
        (
            lambda snapshot: replace(snapshot, backend="cpu_offload", residual_gpu_used_bytes=1),
            RuntimeError,
            "incomplete cpu_offload memory parking",
        ),
        (
            lambda snapshot: {"worker_id": snapshot.worker_id},
            TypeError,
            "invalid memory-parking report",
        ),
    ],
    ids=["mismatched_worker", "gpu_residual", "invalid_type"],
)
async def test_offload_rejects_untrustworthy_parking_evidence(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, reply, error_type, message
) -> None:
    async with _on_demand(
        monkeypatch, tmp_path, ray_sana_snapshot, worker=_parking_worker(sleep_reply=reply)
    ) as lease:
        runtime = lease.runtime
        await runtime.activate()
        session = runtime._session
        actors = lease.actors()
        release = _watch_release(monkeypatch, local_ray)

        with pytest.raises(error_type, match=message) as caught:
            await runtime.offload()

        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._force_close is True
        assert release.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_offload_requires_every_worker_parking_rpc_to_succeed(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _parking_worker(
        sleep_error=RuntimeError("worker sleep failed"),
        sleep_error_on=frozenset({"rollout-1"}),
    )
    async with _on_demand(
        monkeypatch, tmp_path, ray_sana_snapshot, worker=worker, overrides=_TWO_ENGINES
    ) as lease:
        runtime = lease.runtime
        await runtime.activate()
        session = runtime._session
        actors = lease.actors()
        release = _watch_release(monkeypatch, local_ray)

        with pytest.raises(RuntimeError, match="worker sleep failed") as caught:
            await runtime.offload()

        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime._session is None
        assert session._force_close is True
        assert release.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["offload", "activate"])
async def test_worker_parking_remote_error_force_kills_without_graceful_release(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, transition
) -> None:
    timeout = TimeoutError(f"worker {transition} timed out")
    worker = (
        _parking_worker(sleep_error=timeout)
        if transition == "offload"
        else _parking_worker(wake_error=timeout)
    )
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot, worker=worker) as lease:
        runtime = lease.runtime
        await runtime.activate()
        if transition == "activate":
            await runtime.offload()
        session = runtime._session
        actors = lease.actors()
        release = _watch_release(monkeypatch, local_ray)

        with pytest.raises(TimeoutError, match=f"worker {transition} timed out") as caught:
            await getattr(runtime, transition)()

        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._force_close is True
        assert release.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["offload", "activate"])
async def test_worker_parking_deadline_force_kills_without_graceful_release(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, transition
) -> None:
    blocked = "sleep" if transition == "offload" else "wake"
    async with _on_demand(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        worker=_parking_worker(block=frozenset({blocked})),
    ) as lease:
        runtime = lease.runtime
        await runtime.activate()
        if transition == "activate":
            await runtime.offload()
        session = runtime._session
        actors = lease.actors()
        # The fleet park/wake barrier's real deadline, shortened so a wedged
        # rank exhausts it in this test.
        monkeypatch.setattr(session_module, "_RANK_PARK_TIMEOUT_S", 0.5)
        release = _watch_release(monkeypatch, local_ray)

        with pytest.raises(OperationTimeout) as caught:
            await getattr(runtime, transition)()

        assert f"generation.engine_{blocked}" in str(caught.value)
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert session._force_close is True
        assert release.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["offload", "activate"])
async def test_sibling_failure_before_lease_transition_preserves_root_and_force_kills(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, transition
) -> None:
    """A failure recorded before either lease transition is the error that
    surfaces, the runtime terminates, and the fleet is force-killed."""

    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.activate()
        session = runtime._session
        actors = lease.actors()
        release = _watch_release(monkeypatch, local_ray)
        sibling_failure = RayOperationTimeout("rollout.generation.batch", 0.5)
        runtime.lifecycle.fail(sibling_failure)

        with pytest.raises(RayOperationTimeout) as caught:
            await getattr(runtime, transition)()

        assert caught.value is sibling_failure
        assert runtime.lifecycle.failure is sibling_failure
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime._session is None
        assert session._force_close is True
        assert release.events == []
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_offload_cleanup_failure_retries_without_replacing_root(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _parking_worker(sleep_error=RuntimeError("sleep failed"))
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot, worker=worker) as lease:
        runtime = lease.runtime
        await runtime.activate()
        session = runtime._session
        closes = Trace(monkeypatch)
        closes.watch(session, "close", "session.close")
        closes.fail("session.close", "worker cleanup failed")

        with pytest.raises(RuntimeError, match="sleep failed") as caught:
            await runtime.offload()

        assert any("worker cleanup failed" in note for note in caught.value.__notes__)
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        assert runtime._session is session

        await runtime.shutdown()

        assert closes.events == ["session.close", "session.close"]
        assert runtime._session is None
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


# ------------------------------------------------------- activation and lease


@pytest.mark.asyncio
async def test_generate_requires_explicit_activation(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        with pytest.raises(RuntimeError, match=r"await activate\(\) first"):
            await lease.runtime.generate(lease.request(["p"]))

        assert lease.runtime._session is None


@pytest.mark.asyncio
async def test_cold_weights_are_staged_then_applied_during_activation(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        installs = Trace(monkeypatch)
        installs.watch(RayGenerationSession, "update_weights", "session.update_weights")

        await runtime.update_weights(lease.payload(), 2)

        assert runtime._session is None
        assert runtime._pending_install.policy_version == 2
        assert runtime.current_policy_version == 2

        await runtime.activate()

        assert runtime._session is not None
        assert runtime._pending_install is None
        (install,) = installs.calls
        assert install[1][0] is runtime._session
        assert install[1][2] == 2
        # The cold fleet serves the staged version: a request stamped 2 runs.
        output = await runtime.generate(lease.request(["a cat"]))
        assert output.output.shape[0] == 2


@pytest.mark.asyncio
async def test_offload_keeps_workers_and_activation_wakes_latest_policy(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.activate()
        await runtime.update_weights(lease.payload(), 1)
        session = runtime._session
        actors = lease.actors()
        lease_ops = Trace(monkeypatch)
        lease_ops.watch(session, "sleep_engines", "sleep")
        lease_ops.watch(session, "wake_engines", "wake")
        lease_ops.watch(session, "update_weights", "update")

        await runtime.offload()
        await runtime.update_weights(lease.payload(), 2)

        assert runtime._session is session
        assert runtime._session_parked is True
        assert runtime._pending_install.policy_version == 2
        assert lease_ops.events == ["sleep"]

        await runtime.activate()

        assert runtime._session is session
        assert runtime._session_parked is False
        assert runtime._pending_install is None
        assert lease_ops.events == ["sleep", "wake", "update"]
        assert lease_ops.calls[-1][1][1] == 2
        # The same workers, woken, serve the latest policy.
        assert lease.actors() == actors
        output = await runtime.generate(lease.request(["a cat"]))
        assert output.output.shape[0] == 2


@pytest.mark.asyncio
async def test_active_update_publishes_only_after_session_ack(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.activate()
        await runtime.update_weights(lease.payload(), 1)
        started = asyncio.Event()
        finish = asyncio.Event()

        async def hold(_self: Any, *_args: Any, **_kwargs: Any) -> None:
            started.set()
            await finish.wait()

        _gate_class(monkeypatch, RayGenerationSession, "update_weights", before=hold)
        update = asyncio.create_task(runtime.update_weights(lease.payload(), 2))
        await asyncio.wait_for(started.wait(), timeout=30)

        assert runtime._pending_install is None
        assert runtime.current_policy_version == 1

        finish.set()
        await asyncio.wait_for(update, timeout=60)

        assert runtime._pending_install is None
        assert runtime.current_policy_version == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["active", "cold", "parked", "shutdown"])
async def test_runtime_releases_the_payload_once_installed_or_dropped(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, state
) -> None:
    """The runtime never retains a policy payload past its install: an active
    install, a staged install applied by a cold start or a wake, and a staged
    install dropped by shutdown all release it."""

    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        if state in ("active", "parked"):
            await runtime.activate()
        if state == "parked":
            await runtime.offload()
        payload = lease.payload()
        tensor_ref = weakref.ref(next(iter(payload.values())))

        await runtime.update_weights(payload, 2)
        del payload
        gc.collect()
        if state != "active":
            # Staged while inactive: the runtime holds it until it can install.
            assert tensor_ref() is not None
            if state == "shutdown":
                await runtime.shutdown()
            else:
                await runtime.activate()
            gc.collect()

        assert tensor_ref() is None
        assert runtime._pending_install is None


@pytest.mark.asyncio
async def test_active_update_failure_preserves_installed_version_and_terminates(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.activate()
        await runtime.update_weights(lease.payload(), 1)
        actors = lease.actors()

        with pytest.raises(Exception) as caught:
            await runtime.update_weights(lease.bad_payload(), 2)

        assert runtime._pending_install is None
        assert runtime.current_policy_version == 1
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert runtime._session is None
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_update_cleanup_failure_retains_session_for_retry(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.activate()
        await runtime.update_weights(lease.payload(), 1)
        session = runtime._session
        closes = Trace(monkeypatch)
        closes.watch(session, "close", "session.close")
        closes.fail("session.close", "worker cleanup failed")

        with pytest.raises(Exception) as caught:
            await runtime.update_weights(lease.bad_payload(), 2)

        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        assert runtime._session is session

        await runtime.shutdown()

        assert closes.events == ["session.close", "session.close"]
        assert runtime._session is None
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


# ------------------------------------------------ cold start (unpublished)


@pytest.mark.asyncio
async def test_cancelled_restore_closes_the_unpublished_candidate(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.update_weights(lease.payload(), 3)
        candidates: list[RayGenerationSession] = []
        restore_started = asyncio.Event()
        never = asyncio.Event()

        async def hold(candidate: Any, *_args: Any, **_kwargs: Any) -> None:
            candidates.append(candidate)
            restore_started.set()
            await never.wait()

        _gate_class(monkeypatch, RayGenerationSession, "update_weights", before=hold)
        activation = asyncio.create_task(runtime.activate())
        await asyncio.wait_for(restore_started.wait(), timeout=60)
        (candidate,) = candidates
        actors = [rank.actor for rank in candidate.rank_handles]

        activation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await activation

        assert candidate._force_close is True
        assert runtime._session is None
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_cancelled_launch_closes_the_fleet_its_thread_still_produces(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Actor startup runs in a thread cancellation cannot stop; its fleet is closed."""

    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        launcher = lease.launcher
        launch_started = threading.Event()
        finish_launch = threading.Event()
        launched: list[RayGenerationSession] = []
        real_launch = launcher._launch_session

        def launch(*args: Any, **kwargs: Any) -> RayGenerationSession:
            launch_started.set()
            assert finish_launch.wait(timeout=60)
            session = real_launch(*args, **kwargs)
            launched.append(session)
            return session

        monkeypatch.setattr(launcher, "_launch_session", launch)
        activation = asyncio.create_task(runtime.activate())
        assert await asyncio.to_thread(launch_started.wait, 60)

        activation.cancel()
        await asyncio.sleep(0)
        assert not activation.done()
        finish_launch.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(activation, timeout=120)

        (candidate,) = launched
        assert runtime._session is None
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert all(_is_dead(local_ray, rank.actor) for rank in candidate.rank_handles)


@pytest.mark.asyncio
async def test_activation_restore_failure_cleans_candidate_and_terminates(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.update_weights(lease.bad_payload(), 3)
        installs = Trace(monkeypatch)
        installs.watch(RayGenerationSession, "update_weights", "session.update_weights")

        with pytest.raises(Exception) as caught:
            await runtime.activate()

        (install,) = installs.calls
        candidate = install[1][0]
        assert candidate._force_close is True
        assert runtime._session is None
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert all(_is_dead(local_ray, rank.actor) for rank in candidate.rank_handles)


@pytest.mark.asyncio
async def test_cold_restore_timeout_force_kills_unpublished_candidate(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.update_weights(lease.payload(), 2)
        candidates: list[RayGenerationSession] = []
        releases: list[Trace] = []

        async def tighten(candidate: Any, *_args: Any, **_kwargs: Any) -> None:
            # Launch shares this RPC budget with the workers' policy load, so
            # it is tightened only for the restore; the worker cannot ACK in it.
            # The launch's own ray.get calls are over by now.
            candidates.append(candidate)
            releases.append(_watch_release(monkeypatch, local_ray))
            candidate.weight_sync.worker_rpc_timeout_s = 1e-6

        _gate_class(monkeypatch, RayGenerationSession, "update_weights", before=tighten)

        with pytest.raises(RayOperationTimeout) as caught:
            await runtime.activate()

        (candidate,) = candidates
        (release,) = releases
        assert "rollout.weight_sync" in str(caught.value)
        assert runtime._session is None
        assert runtime.current_policy_version == 2
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert candidate._force_close is True
        assert release.events == []
        assert all(_is_dead(local_ray, rank.actor) for rank in candidate.rank_handles)


@pytest.mark.asyncio
async def test_activation_launch_failure_terminalizes_without_publishing_candidate(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        launches = Trace(monkeypatch)
        launches.watch(lease.launcher, "_launch_session", "launch")
        launches.fail("launch", "rollout worker startup failed")

        with pytest.raises(RuntimeError, match="rollout worker startup failed") as caught:
            await runtime.activate()

        assert launches.events == ["launch"]
        assert runtime._session is None
        assert runtime.lifecycle.failure is caught.value
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


# ------------------------------------------------------------------ shutdown


@pytest.mark.asyncio
async def test_on_demand_shutdown_is_idempotent_for_offloaded_workers(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        runtime = lease.runtime
        await runtime.activate()
        await runtime.offload()
        actors = lease.actors()
        closes = Trace(monkeypatch)
        closes.watch(runtime._session, "close", "session.close")

        await runtime.shutdown()
        await runtime.shutdown()

        assert closes.events == ["session.close"]
        assert runtime._session is None
        assert runtime._session_parked is False
        assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
        assert all(_is_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_terminated_runtime_rejects_activation(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _on_demand(monkeypatch, tmp_path, ray_sana_snapshot) as lease:
        launches = Trace(monkeypatch)
        launches.watch(lease.launcher, "_launch_session", "launch")
        await lease.runtime.shutdown()

        with pytest.raises(RuntimeLifecycleError, match="terminated"):
            await lease.runtime.activate()

        assert launches.events == []
