"""Driver-owned Ray operation deadlines, on the package's real Ray cluster.

Every barrier here waits on real ObjectRefs from real actors that sleep past a
short real deadline; cancellation and actor kills go through the real ``ray``
module, recorded by wrappers that delegate to it.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

import vrl.ray.actor_group as actor_group_module
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.operation_deadline import (
    RayCallDeadline,
    RayOperationTimeout,
    cancel_ray_refs,
    get_ray_refs,
)
from vrl.utils.deadline import require_timeout


class _StartupWorker:
    """A real actor body whose startup and metadata calls take configured time."""

    def __init__(self, worker_id: str, config: dict) -> None:
        self.worker_id = worker_id
        self.config = dict(config)

    def load_policy(self) -> None:
        time.sleep(float(self.config.get("load_s", 0.0)))

    def worker_metadata(self) -> dict:
        from vrl.ray.dependencies import current_gpu_ids, current_node_ip

        time.sleep(float(self.config.get("metadata_s", 0.0)))
        return {
            "worker_id": self.worker_id,
            "node_ip": current_node_ip(),
            "gpu_ids": current_gpu_ids(),
        }

    def ping(self) -> bool:
        return True


class _Sleeper:
    def sleep(self, seconds: float) -> float:
        time.sleep(seconds)
        return seconds


def _record_cancels(monkeypatch, ray: Any) -> list[tuple[Any, bool]]:
    """Record every ``ray.cancel`` the code under test issues; the real call still runs."""

    cancelled: list[tuple[Any, bool]] = []
    real_cancel = ray.cancel

    def cancel(ref: Any, *, force: bool = False, **kwargs: Any) -> Any:
        cancelled.append((ref, force))
        return real_cancel(ref, force=force, **kwargs)

    monkeypatch.setattr(ray, "cancel", cancel)
    return cancelled


def _record_kills(monkeypatch) -> list[Any]:
    """Record the actors the group kills; the real kill still runs."""

    killed: list[Any] = []
    real_kill = actor_group_module.kill_actors

    def kill(ray: Any, actors: list[Any]) -> list[tuple[Any, Exception]]:
        killed.extend(actors)
        return real_kill(ray, actors)

    monkeypatch.setattr(actor_group_module, "kill_actors", kill)
    return killed


def _record_barriers(monkeypatch) -> list[tuple[str, list[Any]]]:
    """Record each ``get_ray_refs`` barrier the group waits on: operation and refs."""

    barriers: list[tuple[str, list[Any]]] = []
    real_get = actor_group_module.get_ray_refs

    def get(ray: Any, refs: Any, *, operation: str, **kwargs: Any) -> Any:
        barriers.append((operation, list(refs)))
        return real_get(ray, refs, operation=operation, **kwargs)

    monkeypatch.setattr(actor_group_module, "get_ray_refs", get)
    return barriers


def _dead(ray: Any, actor: Any, *, timeout_s: float = 30.0) -> bool:
    """Whether the actor is gone within ``timeout_s``; ``ray.kill`` is asynchronous."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            ray.get(actor.ping.remote(), timeout=5)
        except ray.exceptions.RayActorError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("timeout_s", [0.0, float("nan")])
def test_ray_deadline_rejects_invalid_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
        require_timeout(timeout_s)
    with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
        RayCallDeadline("test.operation", timeout_s)


def test_sync_deadline_cancels_refs_and_preserves_timeout_cause(local_ray, monkeypatch) -> None:
    ray = local_ray
    actor = ray.remote(num_cpus=0)(_Sleeper).remote()
    try:
        # Two calls that outlive the deadline. Ray cannot interrupt a sync
        # actor task; correctness comes from the owner's actor kill.
        refs = [actor.sleep.remote(3.0), actor.sleep.remote(3.0)]
        cancelled = _record_cancels(monkeypatch, ray)

        with pytest.raises(RayOperationTimeout) as caught:
            get_ray_refs(ray, refs, operation="test.sync_barrier", timeout_s=1.0)

        assert caught.value.operation == "test.sync_barrier"
        assert isinstance(caught.value.__cause__, ray.exceptions.GetTimeoutError)
        assert cancelled == [(refs[0], False), (refs[1], False)]
    finally:
        ray.kill(actor, no_restart=True)


def test_cancel_failure_is_attached_without_replacing_root_error(local_ray) -> None:
    root = RayOperationTimeout("test.cancel", 1.0)

    # Real ``ray.cancel`` refuses anything that is not an ObjectRef.
    failures = cancel_ray_refs(local_ray, [object(), object()], root_error=root)

    assert [type(error) for error in failures] == [TypeError, TypeError]
    (note,) = root.__notes__
    assert note.startswith(
        "Ray ref cancellation incomplete: 2 cancellation attempt(s) failed; first=TypeError("
    )
    assert "ray.cancel() only supported for object refs" in note


def test_actor_group_timeout_kills_every_candidate_actor(local_ray, monkeypatch) -> None:
    ray = local_ray
    cancelled = _record_cancels(monkeypatch, ray)
    killed = _record_kills(monkeypatch)
    barriers = _record_barriers(monkeypatch)

    with pytest.raises(RayOperationTimeout, match=r"rollout\.startup\.load_policy"):
        RayActorGroup.launch(
            worker_cls=_StartupWorker,
            worker_configs=[{"load_s": 30.0}, {"load_s": 30.0}],
            worker_ids=["w0", "w1"],
            num_cpus=0.0,
            num_gpus=0.0,
            rpc_timeout_s=3.0,
            operation_prefix="rollout",
            startup_method="load_policy",
        )

    ((operation, startup_refs),) = barriers
    assert operation == "rollout.startup.load_policy"
    assert cancelled == [(ref, False) for ref in startup_refs]
    assert len(killed) == 2
    assert all(_dead(ray, actor) for actor in killed)


def test_actor_group_metadata_wait_has_a_fresh_budget(local_ray, monkeypatch) -> None:
    """Startup and metadata each take most of one budget; only per-barrier budgets fit."""

    barriers = _record_barriers(monkeypatch)
    group = RayActorGroup.launch(
        worker_cls=_StartupWorker,
        worker_configs=[{"load_s": 2.5, "metadata_s": 2.5}],
        worker_ids=["w0"],
        num_cpus=0.0,
        num_gpus=0.0,
        rpc_timeout_s=4.0,
        operation_prefix="rollout",
        startup_method="load_policy",
    )
    try:
        assert [operation for operation, _ in barriers] == [
            "rollout.startup.load_policy",
            "rollout.startup.worker_metadata",
        ]
        (handle,) = group.handles
        assert handle.worker_id == "w0"
        assert handle.node_ip == local_ray.util.get_node_ip_address()
    finally:
        group.shutdown()


def test_actor_group_metadata_timeout_kills_candidates(local_ray, monkeypatch) -> None:
    ray = local_ray
    cancelled = _record_cancels(monkeypatch, ray)
    killed = _record_kills(monkeypatch)
    barriers = _record_barriers(monkeypatch)

    with pytest.raises(RayOperationTimeout, match=r"rollout\.startup\.worker_metadata"):
        RayActorGroup.launch(
            worker_cls=_StartupWorker,
            worker_configs=[{"metadata_s": 30.0}, {"metadata_s": 30.0}],
            worker_ids=["w0", "w1"],
            num_cpus=0.0,
            num_gpus=0.0,
            rpc_timeout_s=3.0,
            operation_prefix="rollout",
            startup_method="load_policy",
        )

    assert [operation for operation, _ in barriers] == [
        "rollout.startup.load_policy",
        "rollout.startup.worker_metadata",
    ]
    metadata_refs = barriers[1][1]
    assert cancelled == [(ref, False) for ref in metadata_refs]
    assert len(killed) == 2
    assert all(_dead(ray, actor) for actor in killed)
