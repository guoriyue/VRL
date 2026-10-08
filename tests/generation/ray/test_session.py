"""Tests for the launched Ray generation session resource owner.

Every session here is the one the real launcher built: real
``RayGenerationWorker`` actors serving tiny SANA on the package cluster, the
real executor and weight sync (``ray_sana_runtime``). Cleanup is observed by
recording the real ``ray.get`` / ``ray.kill`` calls the session makes; faults
are injected around those real calls.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from typing import Any

import pytest

from tests.generation.ray._helpers import ray_sana_runtime
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.session import RayGenerationSession

_TWO_ENGINES = ("distributed.resources.rollout.num_engines=2",)


def _record(
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
    name: str,
    log: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
) -> None:
    """Record every call of ``target.name`` (args and kwargs), then run it."""

    real = getattr(target, name)

    def recorded(*args: Any, **kwargs: Any) -> Any:
        log.append((name, args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, recorded)


def _alive(ray: Any, actor: Any) -> bool:
    try:
        ray.get(actor.worker_metadata.remote(), timeout=30)
    except ray.exceptions.RayActorError:
        return False
    return True


def _dead(ray: Any, actor: Any, *, timeout_s: float = 30.0) -> bool:
    """Whether the actor is gone within ``timeout_s``; ``ray.kill`` is asynchronous."""

    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _alive(ray, actor):
            return True
        time.sleep(0.05)
    return False


def _killed(log: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
    return [args[0] for name, args, _ in log if name == "kill"]


@pytest.mark.asyncio
async def test_session_owns_execution_and_delegates_weight_sync(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        session = ray_run.runtime._session

        await session.update_weights(ray_run.trainable_state(), 7)

        # The fleet installed version 7: a request stamped 7 is served by it.
        output = await session.executor.execute(ray_run.request(["a cat"], policy_version=7))
        assert output.output.shape[0] == 2
        assert ray_run.runtime.current_policy_version == 0


@pytest.mark.asyncio
async def test_session_parks_and_wakes_the_complete_worker_fleet(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        session = ray_run.runtime._session

        snapshots = await session.sleep_engines()
        await session.wake_engines()

        assert tuple(snapshot.worker_id for snapshot in snapshots) == (
            "rollout-0",
            "rollout-1",
        )
        # Woken workers still hold their policy and serve generation.
        output = await session.executor.execute(ray_run.request(["a cat"]))
        assert output.output.shape[0] == 2


@pytest.mark.asyncio
async def test_session_parking_failure_does_not_translate_or_close_resources(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """An engine whose rank handle names another worker's actor gets that
    worker's parking report: the mismatch is raised as-is and nothing closes."""

    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        real = ray_run.runtime._session
        miswired = RayGenerationEngine(
            "rollout-0",
            [replace(real.engines[0].primary, actor=real.engines[1].primary.actor)],
        )
        miswired_executor = RayGenerationExecutor(
            [miswired],
            real.executor.gatherer,
            actor_dispatcher=real.executor.actor_dispatcher,
            generation_stall_timeout_s=real.executor.generation_stall_timeout_s,
        )
        session = RayGenerationSession(miswired_executor, real.weight_sync)
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        _record(monkeypatch, local_ray, "get", calls)
        _record(monkeypatch, local_ray, "kill", calls)

        with pytest.raises(RuntimeError, match="mismatched rank memory-parking report"):
            await session.sleep_engines()

        assert session.engines == [miswired]
        assert calls == []
        assert _alive(local_ray, miswired.primary.actor)


@pytest.mark.asyncio
async def test_close_releases_policies_then_kills_workers(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        session = ray_run.runtime._session
        actors = [rank.actor for rank in session.rank_handles]
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        _record(monkeypatch, local_ray, "get", calls)
        _record(monkeypatch, local_ray, "kill", calls)

        await session.close(force=False)
        await session.close(force=False)

        # One graceful wait over both release refs, then one kill per rank.
        assert [name for name, _, _ in calls] == ["get", "kill", "kill"]
        _, (release_refs,), wait = calls[0]
        assert len(release_refs) == 2
        assert wait == {"timeout": 60}
        assert _killed(calls) == actors
        assert session.engines == []
        assert all(_dead(local_ray, actor) for actor in actors)


@pytest.mark.asyncio
async def test_close_kills_finalizers_without_releasing_a_policy(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=("distributed.rollout.pipelined=true",),
    ) as ray_run:
        session = ray_run.runtime._session
        (rank,) = session.rank_handles
        (finalizer,) = session.finalizer_handles
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        _record(monkeypatch, local_ray, "get", calls)
        _record(monkeypatch, local_ray, "kill", calls)

        await session.close(force=False)

        # Only the rank holds a policy to release; both actors are killed.
        _, (release_refs,), _ = calls[0]
        assert len(release_refs) == 1
        assert _killed(calls) == [rank.actor, finalizer.actor]
        assert session.finalizer_handles == []


@pytest.mark.asyncio
async def test_close_retains_only_actor_handles_that_failed_to_die(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(
        monkeypatch, tmp_path, ray_sana_snapshot, overrides=_TWO_ENGINES
    ) as ray_run:
        session = ray_run.runtime._session
        first, failed = (rank.actor for rank in session.rank_handles)
        real_kill = local_ray.kill
        killed: list[Any] = []

        def kill(actor: Any, *, no_restart: bool) -> None:
            if actor is failed and failed not in killed:
                killed.append(failed)
                raise RuntimeError("kill failed")
            real_kill(actor, no_restart=no_restart)

        monkeypatch.setattr(local_ray, "kill", kill)

        with pytest.raises(RuntimeError, match="1 actor kill") as caught:
            await session.close(force=False)

        assert isinstance(caught.value.__cause__, RuntimeError)
        assert _dead(local_ray, first)
        assert session.engines == []
        assert [rank.actor for rank in session.rank_handles] == [failed]

        await session.close(force=True)

        assert session.rank_handles == []
        assert _dead(local_ray, failed)


@pytest.mark.asyncio
async def test_force_close_marks_cleanup_forceful_without_killing_early(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        session = ray_run.runtime._session
        (rank,) = session.rank_handles
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        _record(monkeypatch, local_ray, "get", calls)
        _record(monkeypatch, local_ray, "kill", calls)

        session.force_close()

        # Marking the cleanup forceful touches no actor.
        assert calls == []
        assert session.engines

        await session.close(force=False)

        # The forced close skips the graceful release wait and only kills.
        assert [name for name, _, _ in calls] == ["kill"]
        assert _killed(calls) == [rank.actor]
        assert session.engines == []


@pytest.mark.asyncio
async def test_force_close_interrupts_graceful_release_wait(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
        session = ray_run.runtime._session
        (rank,) = session.rank_handles
        release_started = threading.Event()
        finish_release = threading.Event()
        real_get = local_ray.get

        def blocked_get(refs: list[Any], *, timeout: float) -> list[None]:
            # A wedged graceful release: the wait never returns on its own.
            release_started.set()
            finish_release.wait(timeout=30)
            return [None for _ in refs]

        monkeypatch.setattr(local_ray, "get", blocked_get)
        close = asyncio.create_task(session.close(force=False))
        assert await asyncio.to_thread(release_started.wait, 30)

        session.force_close()
        await close
        finish_release.set()

        assert session.engines == []
        monkeypatch.setattr(local_ray, "get", real_get)
        assert _dead(local_ray, rank.actor)
