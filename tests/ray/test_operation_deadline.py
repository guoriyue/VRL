"""Deterministic tests for driver-owned Ray operation deadlines."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import vrl.ray.actor_group as actor_group_module
from vrl.generation.ray.launcher import _all_ranks_support_versioned_slots
from vrl.ray.actor_group import RayActorGroup, RayActorHandle
from vrl.ray.operation_deadline import (
    RayCallDeadline,
    RayOperationTimeout,
    cancel_ray_refs,
    get_ray_refs,
)
from vrl.utils.deadline import validate_timeout


class _CancelLedger:
    def __init__(self) -> None:
        self.cancelled: list[tuple[Any, bool]] = []

    def cancel(self, ref: Any, *, force: bool) -> None:
        self.cancelled.append((ref, force))


@pytest.mark.parametrize("timeout_s", [0.0, float("nan")])
def test_ray_deadline_rejects_invalid_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
        validate_timeout(timeout_s)
    with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
        RayCallDeadline("test.operation", timeout_s)


class _GetTimeoutError(TimeoutError):
    pass


class _SyncTimeoutRay(_CancelLedger):
    exceptions = SimpleNamespace(GetTimeoutError=_GetTimeoutError)

    def __init__(self) -> None:
        super().__init__()
        self.get_timeout_s: float | None = None

    def get(self, refs: list[Any], *, timeout: float) -> None:
        del refs
        self.get_timeout_s = timeout
        raise _GetTimeoutError("driver wait expired")


def test_sync_deadline_cancels_refs_and_preserves_timeout_cause() -> None:
    ray = _SyncTimeoutRay()
    refs = [object(), object()]

    with pytest.raises(RayOperationTimeout) as caught:
        get_ray_refs(
            ray,
            refs,
            operation="test.sync_barrier",
            timeout_s=2.0,
        )

    assert caught.value.operation == "test.sync_barrier"
    assert isinstance(caught.value.__cause__, _GetTimeoutError)
    assert ray.get_timeout_s is not None and 0 < ray.get_timeout_s <= 2.0
    assert ray.cancelled == [(refs[0], False), (refs[1], False)]


def test_cancel_failure_is_attached_without_replacing_root_error() -> None:
    root = RayOperationTimeout("test.cancel", 1.0)

    class _Ray:
        @staticmethod
        def cancel(_ref: Any, *, force: bool) -> None:
            assert force is False
            raise RuntimeError("cancel transport failed")

    cancel_ray_refs(_Ray(), [object(), object()], root_error=root)

    assert root.__notes__ == [
        "Ray ref cancellation incomplete: 2 cancellation attempt(s) failed; "
        "first=RuntimeError('cancel transport failed')",
    ]


class _RemoteCall:
    def __init__(self, ref: Any) -> None:
        self.ref = ref

    def remote(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.ref


class _StartupActor:
    def __init__(self, startup_ref: Any) -> None:
        self.load_policy = _RemoteCall(startup_ref)
        self.worker_metadata = _RemoteCall(object())


class _RemoteWorkerClass:
    def __init__(self, actors: list[_StartupActor]) -> None:
        self.actors = actors

    def options(self, **_kwargs: Any) -> _RemoteWorkerClass:
        return self

    def remote(self, *_args: Any, **_kwargs: Any) -> _StartupActor:
        return self.actors.pop(0)


class _ActorLaunchRay(_SyncTimeoutRay):
    def __init__(self, actors: list[_StartupActor]) -> None:
        super().__init__()
        self.actors = actors

    def remote(self, **_kwargs: Any):
        return lambda _worker_cls: _RemoteWorkerClass(list(self.actors))


class _MetadataTimeoutRay(_ActorLaunchRay):
    def __init__(self, actors: list[_StartupActor]) -> None:
        super().__init__(actors)
        self.get_calls = 0

    def get(self, refs: list[Any], *, timeout: float) -> list[None]:
        self.get_calls += 1
        self.get_timeout_s = timeout
        if self.get_calls == 1:
            return [None for _ in refs]
        raise _GetTimeoutError("metadata wait expired")


def test_actor_group_timeout_kills_every_candidate_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_refs = [object(), object()]
    actors = [_StartupActor(ref) for ref in startup_refs]
    ray = _ActorLaunchRay(actors)
    killed: list[Any] = []

    monkeypatch.setattr(actor_group_module, "require_ray", lambda: ray)

    def kill(_ray: Any, candidates: list[Any]) -> list[Any]:
        killed.extend(candidates)
        return []

    monkeypatch.setattr(actor_group_module, "kill_actors", kill)

    with pytest.raises(RayOperationTimeout, match=r"rollout\.startup\.load_policy"):
        RayActorGroup.launch(
            worker_cls=object,
            worker_configs=[{}, {}],
            worker_ids=["w0", "w1"],
            num_cpus=0.0,
            num_gpus=0.0,
            rpc_timeout_s=0.01,
            operation_prefix="rollout",
            startup_method="load_policy",
        )

    assert killed == actors
    assert ray.cancelled == [(startup_refs[0], False), (startup_refs[1], False)]


def test_actor_group_metadata_timeout_has_a_fresh_budget_and_kills_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_refs = [object(), object()]
    actors = [_StartupActor(ref) for ref in startup_refs]
    metadata_refs = [actor.worker_metadata.ref for actor in actors]
    ray = _MetadataTimeoutRay(actors)
    killed: list[Any] = []

    monkeypatch.setattr(actor_group_module, "require_ray", lambda: ray)
    monkeypatch.setattr(
        actor_group_module,
        "kill_actors",
        lambda _ray, candidates: killed.extend(candidates) or [],
    )

    with pytest.raises(RayOperationTimeout, match=r"rollout\.startup\.worker_metadata"):
        RayActorGroup.launch(
            worker_cls=object,
            worker_configs=[{}, {}],
            worker_ids=["w0", "w1"],
            num_cpus=0.0,
            num_gpus=0.0,
            rpc_timeout_s=0.01,
            operation_prefix="rollout",
            startup_method="load_policy",
        )

    assert ray.get_calls == 2
    assert killed == actors
    assert ray.cancelled == [(metadata_refs[0], False), (metadata_refs[1], False)]


def test_capability_timeout_is_not_downgraded_to_safe_false() -> None:
    refs = [object(), object()]
    actors = [
        SimpleNamespace(
            supports_versioned_trainable_state=_RemoteCall(ref),
        )
        for ref in refs
    ]
    workers = [
        RayActorHandle(worker_id=f"w{index}", actor=actor) for index, actor in enumerate(actors)
    ]
    ray = _SyncTimeoutRay()

    with pytest.raises(RayOperationTimeout, match=r"rollout\.startup\.versioned_slots"):
        _all_ranks_support_versioned_slots(
            ray,
            workers,
            weight_sync=object(),
            worker_rpc_timeout_s=0.01,
        )

    assert ray.cancelled == [(refs[0], False), (refs[1], False)]
