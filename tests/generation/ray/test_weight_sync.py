"""Commit-ACK tests for generation worker weight synchronization."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import torch

import vrl.generation.ray.weight_sync as weight_sync_module
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.operation_deadline import RayOperationTimeout

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
        worker_rpc_timeout_s=30.0,
    )

    result = await sync.push_to_rollout_workers({"w": 1}, policy_version=3)

    assert result is None
    assert actor.calls == [({"w": 1}, 3)]


@pytest.mark.asyncio
async def test_local_update_rejects_wrong_installed_version() -> None:
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(
                worker_id="rollout-0",
                actor=_LocalWorker(installed_version=2),
            ),
        ],
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

        def remote(self, _state_ref: Any, _policy_version: int) -> Any:
            return self.ref

    class _Ray(_FakeRay):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled: list[tuple[Any, bool]] = []

        def cancel(self, ref: Any, *, force: bool) -> None:
            self.cancelled.append((ref, force))

    ray = _Ray()
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)
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
        worker_rpc_timeout_s=0.01,
    )

    with pytest.raises(RayOperationTimeout, match=r"rollout\.weight_sync"):
        await sync.push_to_rollout_workers({"w": 1}, policy_version=7)

    assert (stalled_ref, False) in ray.cancelled
    assert all(force is False for _ref, force in ray.cancelled)


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
        sync = RayGenerationWeightSync(handles, worker_rpc_timeout_s=30.0)

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
        sync = RayGenerationWeightSync(handles, worker_rpc_timeout_s=30.0)

        with pytest.raises(RuntimeError, match=r"rollout-1.*version 4.*expected 5"):
            await sync.push_to_rollout_workers({"w": torch.arange(6)}, policy_version=5)
    finally:
        for handle in handles:
            local_ray.kill(handle.actor, no_restart=True)
