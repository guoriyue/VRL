"""Commit-ACK tests for generation worker weight synchronization."""

from __future__ import annotations

from typing import Any

import pytest

import vrl.generation.ray.weight_sync as weight_sync_module
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker


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
    )

    with pytest.raises(RuntimeError, match=r"rollout-0.*version 2.*expected 3"):
        await sync.push_to_rollout_workers({"w": 1}, policy_version=3)


@pytest.mark.asyncio
async def test_remote_update_results_are_verified_without_second_ack_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray = _FakeRay()
    first = _RemoteWorker(installed_version=4)
    second = _RemoteWorker(installed_version=4)
    monkeypatch.setattr(weight_sync_module, "require_ray", lambda: ray)

    async def _detached_thread_wait_is_forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("weight ACK must stay on the owner event loop")

    monkeypatch.setattr(
        weight_sync_module.asyncio,
        "to_thread",
        _detached_thread_wait_is_forbidden,
    )
    sync = RayGenerationWeightSync(
        [
            DistributedWorkerHandle(worker_id="rollout-0", actor=first),
            DistributedWorkerHandle(worker_id="rollout-1", actor=second),
        ],
    )

    await sync.push_to_rollout_workers({"w": 1}, policy_version=4)

    assert ray.put_calls == [{"w": 1}]
    shared_state = ("state", {"w": 1})
    assert first.update_weights.calls == [(shared_state, 4)]
    assert second.update_weights.calls == [(shared_state, 4)]


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
    )

    with pytest.raises(RuntimeError, match=r"rollout-1.*version 4.*expected 5"):
        await sync.push_to_rollout_workers({"w": 1}, policy_version=5)
