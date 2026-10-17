"""Tests for the launched Ray generation session resource owner.

Every session here is the one the real launcher built: real
``RayGenerationWorker`` actors serving tiny SANA on the package cluster, the
real executor and weight sync (``ray_sana_runtime``). Cleanup is observed by
recording the real ``ray.get`` / ``ray.kill`` calls the session makes; faults
are injected around those real calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.generation.ray._helpers import is_dead, ray_sana_runtime

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


def _killed(log: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
    return [args[0] for name, args, _ in log if name == "kill"]


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
        assert all(is_dead(local_ray, actor) for actor in actors)


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
        assert is_dead(local_ray, first)
        assert session.engines == []
        assert [rank.actor for rank in session.rank_handles] == [failed]

        await session.close(force=True)

        assert session.rank_handles == []
        assert is_dead(local_ray, failed)
