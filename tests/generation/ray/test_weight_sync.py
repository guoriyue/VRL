"""Commit-ACK tests for generation worker weight synchronization."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vrl.generation.ray.weight_sync as weight_sync_module
import vrl.ray.actor_pool as actor_pool_module
import vrl.ray.operation_deadline as deadline_module
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.ray.lifecycle_fsm import RuntimePhase
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout

# Carried by the two tests that drive `_FakeRay`. They are kept, not converted:
# `put_calls == [{"w": 1}]` is the only assertion anywhere that pins ONE put
# shared by N workers rather than N puts, and the real object store keeps no
# ledger that could replace it (reference counts are not a stable assertable
# interface). The half a fake `put()` structurally cannot reach -- Ray
# dereferencing the ref into the real dict before the worker method runs -- is
# what the real-cluster twin named here asserts from inside the actor process.
_OBJECT_STORE_LEDGER = pytest.mark.real_cover(
    "tests/generation/ray/test_weight_sync.py::test_real_ray_weight_sync_derefs_one_shared_put",
    why=(
        "a real ray.put returns an ObjectRef and records nothing, so 'one put shared by every "
        "worker' has no real-side observable; the fake's tuple return in turn can never exercise "
        "auto-deref across a process boundary, which is what the slow_test twin does"
    ),
)


def _runtime(
    executor: Any,
    *,
    weight_sync: Any | None = None,
) -> RayGenerationRuntime:
    return RayGenerationRuntime(
        session=RayGenerationSession(executor, weight_sync, []),
    )


class _LocalWorker:
    def __init__(self, installed_version: Any) -> None:
        self.installed_version = installed_version
        self.calls: list[tuple[Any, int]] = []

    def update_weights(self, state_ref: Any, policy_version: int) -> Any:
        self.calls.append((state_ref, policy_version))
        return self.installed_version


class _RemoteMethod:
    def __init__(self, installed_version: Any) -> None:
        self.installed_version = installed_version
        self.calls: list[tuple[Any, int]] = []

    def remote(self, state_ref: Any, policy_version: int) -> _FakeObjectRef:
        self.calls.append((state_ref, policy_version))
        return _FakeObjectRef(self.installed_version)


class _FakeObjectRef:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __await__(self):
        async def _resolve() -> Any:
            return self.value

        return _resolve().__await__()


class _NeverObjectRef:
    def __await__(self):
        async def _wait_forever() -> None:
            await asyncio.Event().wait()

        return _wait_forever().__await__()


class _GatedObjectRef:
    def __init__(self, gate: asyncio.Event, value: Any) -> None:
        self.gate = gate
        self.value = value

    def __await__(self):
        async def _wait() -> Any:
            await self.gate.wait()
            return self.value

        return _wait().__await__()


class _RemoteWorker:
    def __init__(self, installed_version: Any) -> None:
        self.update_weights = _RemoteMethod(installed_version)


class _FakeRay:
    def __init__(self) -> None:
        self.put_calls: list[Any] = []

    def put(self, value: Any) -> tuple[str, Any]:
        self.put_calls.append(value)
        return ("state", value)


def test_ray_worker_returns_core_install_ack() -> None:
    class _Core:
        def update_weights(self, state_ref: Any, policy_version: int) -> int:
            assert state_ref == {"w": 1}
            return policy_version

    worker = object.__new__(RayGenerationWorker)
    worker.core = _Core()

    assert worker.update_weights({"w": 1}, 6) == 6


@pytest.mark.asyncio
async def test_local_update_return_is_the_commit_ack() -> None:
    actor = _LocalWorker(installed_version=3)
    sync = RayGenerationWeightSync(
        [DistributedWorkerHandle(worker_id="rollout-0", actor=actor)],
        actor_dispatcher=RayActorDispatcher(("rollout-0",)),
        worker_rpc_timeout_s=30.0,
    )

    result = await sync.push_to_rollout_workers({"w": 1}, policy_version=3)

    assert result is None
    assert actor.calls == [({"w": 1}, 3)]


@pytest.mark.asyncio
async def test_invalid_policy_version_does_not_terminalize_resident_runtime() -> None:
    class _RecordingSync:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, int]] = []

        async def push_to_rollout_workers(
            self,
            state_ref: Any,
            policy_version: int,
        ) -> None:
            self.calls.append((state_ref, policy_version))

    sync = _RecordingSync()
    runtime = _runtime(SimpleNamespace(), weight_sync=sync)
    runtime.current_policy_version = 3

    with pytest.raises(TypeError):
        await runtime.update_weights({"w": 2}, policy_version=object())

    assert sync.calls == []
    assert runtime.current_policy_version == 3
    assert runtime.lifecycle.failure is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


@pytest.mark.asyncio
async def test_local_update_rejects_wrong_installed_version() -> None:
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=_LocalWorker(installed_version=2),
            ),
        ],
        actor_dispatcher=RayActorDispatcher(("rollout-0",)),
        worker_rpc_timeout_s=30.0,
    )

    with pytest.raises(RuntimeError, match=r"rollout-0.*version 2.*expected 3"):
        await sync.push_to_rollout_workers({"w": 1}, policy_version=3)


@_OBJECT_STORE_LEDGER
@pytest.mark.asyncio
async def test_remote_update_results_are_verified_without_second_ack_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray = _FakeRay()
    first = _RemoteWorker(installed_version=4)
    second = _RemoteWorker(installed_version=4)
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(worker_id="rollout-0", actor=first),
            DistributedWorkerHandle(worker_id="rollout-1", actor=second),
        ],
        actor_dispatcher=RayActorDispatcher(("rollout-0", "rollout-1")),
        worker_rpc_timeout_s=30.0,
    )

    await sync.push_to_rollout_workers({"w": 1}, policy_version=4)

    assert ray.put_calls == [{"w": 1}]
    shared_state = ("state", {"w": 1})
    assert first.update_weights.calls == [(shared_state, 4)]
    assert second.update_weights.calls == [(shared_state, 4)]


@_OBJECT_STORE_LEDGER
@pytest.mark.asyncio
async def test_remote_update_rejects_partial_wrong_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray = _FakeRay()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=_RemoteWorker(installed_version=5),
            ),
            DistributedWorkerHandle(
                worker_id="rollout-1",
                actor=_RemoteWorker(installed_version=4),
            ),
        ],
        actor_dispatcher=RayActorDispatcher(("rollout-0", "rollout-1")),
        worker_rpc_timeout_s=30.0,
    )

    with pytest.raises(RuntimeError, match=r"rollout-1.*version 4.*expected 5"):
        await sync.push_to_rollout_workers({"w": 1}, policy_version=5)


@_OBJECT_STORE_LEDGER
@pytest.mark.asyncio
async def test_remote_update_timeout_rejects_partial_ack_and_cancels_every_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_ref = _FakeObjectRef(7)
    stalled_ref = _NeverObjectRef()

    class _Method:
        def __init__(self, ref: Any) -> None:
            self.ref = ref

        def remote(self, _state_ref: Any, policy_version: int) -> Any:
            del policy_version
            return self.ref

    class _Ray(_FakeRay):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled: list[tuple[Any, bool]] = []

        def cancel(self, ref: Any, *, force: bool) -> None:
            self.cancelled.append((ref, force))

    ray = _Ray()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: ray)
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=type("_Worker", (), {"update_weights": _Method(completed_ref)})(),
            ),
            DistributedWorkerHandle(
                worker_id="rollout-1",
                actor=type("_Worker", (), {"update_weights": _Method(stalled_ref)})(),
            ),
        ],
        actor_dispatcher=RayActorDispatcher(("rollout-0", "rollout-1")),
        worker_rpc_timeout_s=0.01,
    )

    with pytest.raises(RayOperationTimeout, match=r"rollout\.weight_sync"):
        await sync.push_to_rollout_workers({"w": 1}, policy_version=7)

    assert (stalled_ref, False) in ray.cancelled
    assert all(force is False for _ref, force in ray.cancelled)


@pytest.mark.asyncio
async def test_weight_sync_gets_a_full_deadline_after_shared_worker_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy generation call may outlive the weight ACK timeout."""

    gate = asyncio.Event()
    generation_submitted = asyncio.Event()
    deadlines: list[tuple[str, float]] = []
    real_deadline = actor_pool_module.RayCallDeadline

    def recording_deadline(operation: str, timeout_s: float, **kwargs: Any) -> Any:
        deadlines.append((operation, timeout_s))
        return real_deadline(operation, timeout_s, **kwargs)

    def submit_generation(_payload: Any) -> _GatedObjectRef:
        generation_submitted.set()
        return _GatedObjectRef(gate, "generated")

    ray = _FakeRay()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    dispatcher = RayActorDispatcher(("rollout-0",))
    generation = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id="rollout-0",
                    remote_method=submit_generation,
                    payload=None,
                ),
            ],
            operation="rollout.generation.chunk",
            call_timeout_s=30.0,
        ),
    )
    await generation_submitted.wait()

    actor = _RemoteWorker(installed_version=3)
    sync = RayGenerationWeightSync(
        [DistributedWorkerHandle(worker_id="rollout-0", actor=actor)],
        actor_dispatcher=dispatcher,
        worker_rpc_timeout_s=0.01,
    )
    update = asyncio.create_task(
        sync.push_to_rollout_workers({"w": 1}, policy_version=3),
    )
    await asyncio.sleep(0.03)

    assert not update.done()
    assert actor.update_weights.calls == []
    assert deadlines == [("rollout.generation.chunk", 30.0)]

    gate.set()
    assert await generation == [(0, "generated")]
    await update
    assert actor.update_weights.calls == [(("state", {"w": 1}), 3)]
    assert deadlines == [
        ("rollout.generation.chunk", 30.0),
        ("rollout.weight_sync", 0.01),
    ]


@pytest.mark.asyncio
async def test_waiting_weight_sync_gets_fair_handoff_before_pending_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    first_submitted = asyncio.Event()
    submissions: list[str] = []

    def submit_generation(payload: str) -> Any:
        submissions.append(payload)
        if payload == "generation-0":
            first_submitted.set()
            return _GatedObjectRef(gate, payload)
        return _FakeObjectRef(payload)

    class _UpdateMethod:
        @staticmethod
        def remote(_state_ref: Any, policy_version: int) -> _FakeObjectRef:
            submissions.append("weight")
            return _FakeObjectRef(policy_version)

    ray = _FakeRay()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    dispatcher = RayActorDispatcher(("rollout-0",))
    generation = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(
                    job_index=index,
                    worker_id="rollout-0",
                    remote_method=submit_generation,
                    payload=f"generation-{index}",
                )
                for index in range(3)
            ],
            operation="rollout.generation.chunk",
            call_timeout_s=30.0,
        ),
    )
    await first_submitted.wait()

    actor = SimpleNamespace(update_weights=_UpdateMethod())
    sync = RayGenerationWeightSync(
        [DistributedWorkerHandle(worker_id="rollout-0", actor=actor)],
        actor_dispatcher=dispatcher,
        worker_rpc_timeout_s=30.0,
    )
    update = asyncio.create_task(
        sync.push_to_rollout_workers({"w": 1}, policy_version=3),
    )
    await asyncio.sleep(0)

    gate.set()
    await update
    assert await generation == [
        (0, "generation-0"),
        (1, "generation-1"),
        (2, "generation-2"),
    ]
    assert submissions == [
        "generation-0",
        "weight",
        "generation-1",
        "generation-2",
    ]


@pytest.mark.asyncio
async def test_cancelling_weight_sync_before_submission_keeps_runtime_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    generation_submitted = asyncio.Event()

    def submit_generation(_payload: Any) -> _GatedObjectRef:
        generation_submitted.set()
        return _GatedObjectRef(gate, "generated")

    ray = _FakeRay()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    dispatcher = RayActorDispatcher(("rollout-0",))
    generation = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "rollout-0", submit_generation, None)],
            operation="rollout.generation.chunk",
            call_timeout_s=30.0,
        ),
    )
    await generation_submitted.wait()

    actor = _RemoteWorker(installed_version=4)
    sync = RayGenerationWeightSync(
        [DistributedWorkerHandle(worker_id="rollout-0", actor=actor)],
        actor_dispatcher=dispatcher,
        worker_rpc_timeout_s=30.0,
    )
    runtime = _runtime(
        SimpleNamespace(actor_dispatcher=dispatcher),
        weight_sync=sync,
    )
    waiting = asyncio.create_task(
        runtime.update_weights({"w": 1}, policy_version=4),
    )
    await asyncio.sleep(0)
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError) as caught:
        await waiting
    assert caught.value.__cause__ is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
    assert actor.update_weights.calls == []

    gate.set()
    assert await generation == [(0, "generated")]
    await runtime.update_weights({"w": 2}, policy_version=4)
    assert runtime.current_policy_version == 4


@pytest.mark.asyncio
async def test_completed_weight_sync_wins_cancellation_and_publishes_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    submitted = asyncio.Event()

    class _GatedUpdateMethod:
        def remote(self, _state_ref: Any, policy_version: int) -> _GatedObjectRef:
            submitted.set()
            return _GatedObjectRef(gate, policy_version)

    ray = _FakeRay()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    dispatcher = RayActorDispatcher(("rollout-0",))
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=SimpleNamespace(update_weights=_GatedUpdateMethod()),
            ),
        ],
        actor_dispatcher=dispatcher,
        worker_rpc_timeout_s=30.0,
    )
    runtime = _runtime(
        SimpleNamespace(actor_dispatcher=dispatcher),
        weight_sync=sync,
    )
    runtime.current_policy_version = 1
    update = asyncio.create_task(
        runtime.update_weights({"w": 2}, policy_version=2),
    )
    await submitted.wait()

    gate.set()
    update.cancel()
    await update

    assert runtime.current_policy_version == 2
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


@pytest.mark.asyncio
async def test_completed_weight_sync_cancellation_still_validates_wrong_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    submitted = asyncio.Event()

    class _WrongAckMethod:
        @staticmethod
        def remote(_state_ref: Any, policy_version: int) -> _GatedObjectRef:
            del policy_version
            submitted.set()
            return _GatedObjectRef(gate, 99)

    ray = _FakeRay()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    dispatcher = RayActorDispatcher(("rollout-0",))
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=SimpleNamespace(update_weights=_WrongAckMethod()),
            ),
        ],
        actor_dispatcher=dispatcher,
        worker_rpc_timeout_s=30.0,
    )
    runtime = _runtime(
        SimpleNamespace(actor_dispatcher=dispatcher),
        weight_sync=sync,
    )
    runtime.current_policy_version = 1
    update = asyncio.create_task(
        runtime.update_weights({"w": 2}, policy_version=2),
    )
    await submitted.wait()

    gate.set()
    update.cancel()
    with pytest.raises(
        RuntimeError,
        match=r"rollout-0.*version 99.*expected 2",
    ):
        await update

    assert runtime.current_policy_version == 1
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_cancelling_partially_completed_weight_sync_terminalizes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy_gate = asyncio.Event()
    busy_submitted = asyncio.Event()
    w0_completed = asyncio.Event()
    cancelled_refs: list[Any] = []

    class _CompletedRef:
        def __await__(self):
            async def _resolve() -> int:
                w0_completed.set()
                return 2

            return _resolve().__await__()

    completed_ref = _CompletedRef()
    busy_ref = _GatedObjectRef(busy_gate, "generated")

    def occupy_w1(_payload: Any) -> _GatedObjectRef:
        busy_submitted.set()
        return busy_ref

    class _UpdateMethod:
        def __init__(self, ref: Any) -> None:
            self.ref = ref
            self.calls: list[int] = []

        def remote(self, _state_ref: Any, policy_version: int) -> Any:
            self.calls.append(policy_version)
            return self.ref

    class _Ray(_FakeRay):
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled_refs.append(ref)

    ray = _Ray()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: ray)
    dispatcher = RayActorDispatcher(("w0", "w1"))
    generation = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w1", occupy_w1, None)],
            operation="rollout.generation.chunk",
            call_timeout_s=30.0,
        ),
    )
    await busy_submitted.wait()

    w0_update = _UpdateMethod(completed_ref)
    w1_update = _UpdateMethod(_FakeObjectRef(2))
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="w0",
                actor=SimpleNamespace(update_weights=w0_update),
            ),
            DistributedWorkerHandle(
                worker_id="w1",
                actor=SimpleNamespace(update_weights=w1_update),
            ),
        ],
        actor_dispatcher=dispatcher,
        worker_rpc_timeout_s=30.0,
    )
    runtime = _runtime(
        SimpleNamespace(actor_dispatcher=dispatcher),
        weight_sync=sync,
    )
    runtime.current_policy_version = 1
    update = asyncio.create_task(
        runtime.update_weights({"w": 2}, policy_version=2),
    )
    await w0_completed.wait()
    for _ in range(20):
        if completed_ref not in dispatcher._active_refs:
            break
        await asyncio.sleep(0)
    assert completed_ref not in dispatcher._active_refs
    assert w0_update.calls == [2]
    assert w1_update.calls == []

    update.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await update

    assert isinstance(caught.value.__cause__, RayOperationCancelled)
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED
    assert runtime.current_policy_version == 1
    assert busy_ref in cancelled_refs
    assert completed_ref not in cancelled_refs

    generation.cancel()
    await asyncio.gather(generation, return_exceptions=True)


# ------------------------------------------------------- real cluster (real Ray)


class _InstallWorker:
    """Real Ray actor that receives the shared state ref and keeps what arrived."""

    def __init__(self, ack_offset: int = 0, stall_s: float = 0.0) -> None:
        self._ack_offset = int(ack_offset)
        self._stall_s = float(stall_s)
        self._installed: Any = None

    def update_weights(self, state_ref: Any, policy_version: int) -> int:
        # The claim only a real cluster can settle: production hands every worker
        # the same ObjectRef, and Ray dereferences it into the real dict before
        # this method body runs. A fake put() returning a tuple can never show it.
        assert isinstance(state_ref, dict), f"worker received {type(state_ref).__name__}"
        self._installed = state_ref
        if self._stall_s:
            time.sleep(self._stall_s)
        return policy_version + self._ack_offset

    def hold_default_slot(self, stall_s: float) -> str:
        time.sleep(stall_s)
        return "generated"

    def installed_sum(self) -> float:
        return float(self._installed["w"].sum())


def _install_fleet(
    ray: Any,
    *scripts: tuple[int, float],
) -> list[DistributedWorkerHandle]:
    """One real ``_InstallWorker`` actor per ``(ack_offset, stall_s)`` script."""

    actor_cls = ray.remote(num_cpus=0)(_InstallWorker)
    return [
        DistributedWorkerHandle(
            worker_id=f"rollout-{index}",
            actor=actor_cls.remote(ack_offset, stall_s),
        )
        for index, (ack_offset, stall_s) in enumerate(scripts)
    ]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_real_ray_weight_sync_derefs_one_shared_put(local_ray) -> None:
    """One real ``ray.put`` reaches two real worker processes as a real dict.

    The tensor is asserted on the far side of two process boundaries, so this
    covers what the in-process tests cannot: the state actually survives
    serialization and arrives dereferenced.
    """

    handles = _install_fleet(local_ray, (0, 0.0), (0, 0.0))
    try:
        sync = RayGenerationWeightSync(
            handles,
            actor_dispatcher=RayActorDispatcher(
                tuple(handle.worker_id for handle in handles),
            ),
            worker_rpc_timeout_s=30.0,
        )

        # Fixed tensor, no RNG: the sum below is the payload's identity.
        await sync.push_to_rollout_workers({"w": torch.arange(6)}, policy_version=4)

        installed = local_ray.get([handle.actor.installed_sum.remote() for handle in handles])
        assert installed == [15.0, 15.0]
    finally:
        for handle in handles:
            local_ray.kill(handle.actor, no_restart=True)


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_real_ray_weight_sync_attributes_a_wrong_ack_by_submission_order(
    local_ray,
) -> None:
    """A bad ACK is blamed on the worker that sent it, not the one that answered
    last.

    Production pairs ACKs back to workers with ``zip(remote_workers,
    installed_versions, strict=True)``, which is only correct because
    ``asyncio.gather`` preserves submission order. rollout-0 is made to finish
    LAST here, so an implementation that switched to ``asyncio.as_completed``
    would name rollout-0 in the error and redden this test.
    """

    # rollout-0 stalls half a second, so it is the LAST to complete; rollout-1
    # acks one version short and completes first.
    handles = _install_fleet(local_ray, (0, 0.5), (-1, 0.0))
    try:
        sync = RayGenerationWeightSync(
            handles,
            actor_dispatcher=RayActorDispatcher(
                tuple(handle.worker_id for handle in handles),
            ),
            worker_rpc_timeout_s=30.0,
        )

        with pytest.raises(RuntimeError, match=r"rollout-1.*version 4.*expected 5"):
            await sync.push_to_rollout_workers({"w": torch.arange(6)}, policy_version=5)
    finally:
        for handle in handles:
            local_ray.kill(handle.actor, no_restart=True)


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_real_ray_weight_sync_deadline_excludes_generation_admission_wait(
    local_ray,
) -> None:
    """A real synchronous actor cannot pre-queue weight sync behind generation."""

    handles = _install_fleet(local_ray, (0, 0.0))
    dispatcher = RayActorDispatcher(("rollout-0",))
    submitted = asyncio.Event()

    def submit_generation(stall_s: float) -> Any:
        submitted.set()
        return handles[0].actor.hold_default_slot.remote(stall_s)

    generation = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "rollout-0", submit_generation, 0.6)],
            operation="rollout.generation.chunk",
            call_timeout_s=10.0,
        ),
    )
    try:
        await submitted.wait()
        sync = RayGenerationWeightSync(
            handles,
            actor_dispatcher=dispatcher,
            worker_rpc_timeout_s=0.3,
        )
        started = time.monotonic()
        await sync.push_to_rollout_workers({"w": torch.arange(3)}, policy_version=8)
        elapsed = time.monotonic() - started

        assert elapsed > 0.3
        assert await generation == [(0, "generated")]
    finally:
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        for handle in handles:
            local_ray.kill(handle.actor, no_restart=True)
