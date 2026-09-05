"""Tests for the launched Ray generation session resource owner."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from tests.generation.ray._helpers import ResolvedRef as _ResolvedRef
from tests.generation.ray._helpers import parking_snapshot as _parking_snapshot
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.session import RayGenerationSession
from vrl.ray.actor_group import RayActorHandle


class _RemoteCall:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def remote(self) -> _ResolvedRef:
        self.calls += 1
        return _ResolvedRef(self.value)


class _Actor:
    def __init__(self, worker_id: str) -> None:
        self.sleep = _RemoteCall(_parking_snapshot(worker_id))
        self.wake = _RemoteCall(None)
        self.release_policy = _RemoteCall(None)


class _Executor:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def execute(self, request: Any) -> Any:
        self.calls.append(("generate", request))
        return "generated"

    async def probe_batch_sizes(self, request: Any, *, max_samples: int) -> list[Any]:
        self.calls.append(("probe", request, max_samples))
        return [{"samples_per_generation_batch": max_samples}]


class _WeightSync:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def push_to_rollout_engines(self, state_ref: Any, policy_version: int) -> None:
        self.calls.append((state_ref, policy_version))


class _Ray:
    def __init__(self, *, failed_actor: Any | None = None) -> None:
        self.failed_actor = failed_actor
        self.get_calls: list[tuple[list[Any], float]] = []
        self.killed: list[Any] = []

    def get(self, refs: list[Any], *, timeout: float) -> list[None]:
        self.get_calls.append((refs, timeout))
        return [None for _ in refs]

    def kill(self, actor: Any, *, no_restart: bool) -> None:
        assert no_restart is True
        if actor is self.failed_actor:
            raise RuntimeError("kill failed")
        self.killed.append(actor)


def _session(
    *actors: _Actor | None,
    weight_sync: Any | None = None,
    supports_non_draining_weight_sync: bool = False,
) -> RayGenerationSession:
    return RayGenerationSession(
        _Executor(),
        weight_sync,
        [
            RayGenerationEngine(
                f"rollout-{index}",
                [RayActorHandle(worker_id=f"rollout-{index}", actor=actor)],
            )
            for index, actor in enumerate(actors)
        ],
        supports_non_draining_weight_sync,
    )


@pytest.mark.asyncio
async def test_session_owns_execution_and_delegates_weight_sync() -> None:
    executor = _Executor()
    weight_sync = _WeightSync()
    session = RayGenerationSession(
        executor,
        weight_sync,
        [],
        supports_non_draining_weight_sync=True,
    )
    await session.update_weights("weights", 7)

    assert session.executor is executor
    assert session.weight_sync is weight_sync
    assert executor.calls == []
    assert weight_sync.calls == [("weights", 7)]
    assert session.supports_non_draining_weight_sync is True


@pytest.mark.asyncio
async def test_session_rejects_weight_update_without_sync_owner() -> None:
    session = _session()

    with pytest.raises(RuntimeError, match="has no GenerationWeightSync"):
        await session.update_weights({}, 1)


def test_session_rejects_split_actor_admission_owners() -> None:
    executor = _Executor()
    executor.actor_dispatcher = object()
    weight_sync = _WeightSync()
    weight_sync.actor_dispatcher = object()

    with pytest.raises(ValueError, match="must share one actor dispatcher"):
        RayGenerationSession(executor, weight_sync, [])


@pytest.mark.asyncio
async def test_session_parks_and_wakes_the_complete_worker_fleet() -> None:
    actors = (_Actor("rollout-0"), _Actor("rollout-1"))
    session = _session(*actors)

    snapshots = await session.sleep_engines()
    await session.wake_engines()

    assert tuple(snapshot.worker_id for snapshot in snapshots) == (
        "rollout-0",
        "rollout-1",
    )
    assert [actor.sleep.calls for actor in actors] == [1, 1]
    assert [actor.wake.calls for actor in actors] == [1, 1]


@pytest.mark.asyncio
async def test_session_parking_failure_does_not_translate_or_close_resources() -> None:
    actor = _Actor("wrong-worker")
    session = _session(actor)

    with pytest.raises(RuntimeError, match="mismatched rank memory-parking report"):
        await session.sleep_engines()

    assert session.engines[0].primary.actor is actor
    assert actor.release_policy.calls == 0


def test_session_rejects_duplicate_engine_ids() -> None:
    engines = [
        RayGenerationEngine(
            "rollout-0",
            [RayActorHandle(worker_id="rollout-0", actor=_Actor("rollout-0"))],
        ),
        RayGenerationEngine(
            "rollout-0",
            [RayActorHandle(worker_id="rollout-0", actor=_Actor("rollout-0"))],
        ),
    ]

    with pytest.raises(RuntimeError, match="duplicate generation engine ids"):
        RayGenerationSession(_Executor(), None, engines)


@pytest.mark.asyncio
async def test_close_releases_policies_then_kills_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.session as session_module

    actors = (_Actor("rollout-0"), _Actor("rollout-1"))
    ray = _Ray()
    monkeypatch.setattr(session_module, "require_ray", lambda: ray)
    session = _session(*actors)

    await session.close(force=False)
    await session.close(force=False)

    assert [actor.release_policy.calls for actor in actors] == [1, 1]
    assert len(ray.get_calls) == 1
    assert ray.get_calls[0][1] == 60
    assert ray.killed == list(actors)
    assert session.engines == []


@pytest.mark.asyncio
async def test_close_retains_only_actor_handles_that_failed_to_die(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.session as session_module

    first = _Actor("rollout-0")
    failed = _Actor("rollout-1")
    ray = _Ray(failed_actor=failed)
    monkeypatch.setattr(session_module, "require_ray", lambda: ray)
    session = _session(first, failed)

    with pytest.raises(RuntimeError, match="1 rank actor kill") as caught:
        await session.close(force=False)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert ray.killed == [first]
    # The engine whose rank failed to die is retained; the killed one is gone.
    assert [engine.engine_id for engine in session.engines] == ["rollout-1"]
    assert session.engines[0].primary.actor is failed


@pytest.mark.asyncio
async def test_force_close_marks_cleanup_forceful_without_killing_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.session as session_module

    actor = _Actor("rollout-0")
    ray = _Ray()
    monkeypatch.setattr(session_module, "require_ray", lambda: ray)
    session = _session(actor)

    session.force_close()

    assert actor.release_policy.calls == 0
    assert ray.get_calls == []
    assert ray.killed == []
    assert session.engines

    await session.close(force=False)

    assert ray.killed == [actor]
    assert session.engines == []


@pytest.mark.asyncio
async def test_force_close_interrupts_graceful_release_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.session as session_module

    release_started = threading.Event()
    finish_release = threading.Event()

    class _BlockingRay(_Ray):
        def get(self, refs: list[Any], *, timeout: float) -> list[None]:
            self.get_calls.append((refs, timeout))
            release_started.set()
            finish_release.wait(timeout=1)
            return [None for _ in refs]

    actor = _Actor("rollout-0")
    ray = _BlockingRay()
    monkeypatch.setattr(session_module, "require_ray", lambda: ray)
    session = _session(actor)
    close = asyncio.create_task(session.close(force=False))
    await asyncio.to_thread(release_started.wait, 1)

    session.force_close()
    await close
    finish_release.set()

    assert ray.killed == [actor]
    assert session.engines == []
