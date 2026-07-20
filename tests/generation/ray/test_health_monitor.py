"""Background liveness probing of Ray rollout workers."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.ray.health_monitor import (
    RolloutWorkerHealthMonitor,
    RolloutWorkerUnreachable,
)
from vrl.generation.ray.lifecycle_fsm import RuntimeLifecycle, RuntimePhase


class _ProbeRef:
    """A probe ObjectRef that carries the answer its actor is scripted to give."""

    def __init__(self, behaviour: Any) -> None:
        self.behaviour = behaviour


class _ProbeMethod:
    def __init__(self, behaviour: Any) -> None:
        self._behaviour = behaviour
        self.calls = 0

    def remote(self) -> _ProbeRef:
        self.calls += 1
        return _ProbeRef(self._behaviour)


class _Actor:
    def __init__(self, behaviour: Any = None) -> None:
        self.health = _ProbeMethod(behaviour)


class _FakeRay:
    """Resolve probe refs to whatever their issuing actor was scripted with."""

    def __init__(self, actors: list[_Actor] | None = None) -> None:
        del actors  # refs are self-describing; kept for call-site readability
        self.get_timeouts: list[float] = []
        self.killed: list[Any] = []

    def get(self, ref: _ProbeRef, timeout: float | None = None) -> Any:
        self.get_timeouts.append(float(timeout))
        if isinstance(ref.behaviour, BaseException):
            raise ref.behaviour
        return ref.behaviour

    def kill(self, actor: Any, *, no_restart: bool = False) -> None:
        self.killed.append(actor)


def _runtime(*actors: _Actor) -> Any:
    return SimpleNamespace(
        _owned_workers=[
            DistributedWorkerHandle(worker_id=f"rollout-{i}", actor=actor)
            for i, actor in enumerate(actors)
        ],
        lifecycle=RuntimeLifecycle(),
    )


def _monitor(runtime: Any, **kwargs: Any) -> RolloutWorkerHealthMonitor:
    defaults = {"interval_s": 0.01, "timeout_s": 0.5, "first_wait_s": 0.0}
    defaults.update(kwargs)
    return RolloutWorkerHealthMonitor(runtime, **defaults)


def _install_ray(monkeypatch: pytest.MonkeyPatch, ray: _FakeRay) -> None:
    monkeypatch.setattr(
        "vrl.generation.ray.health_monitor.require_ray",
        lambda: ray,
    )
    monkeypatch.setattr(
        "vrl.generation.ray.health_monitor.kill_actors",
        lambda _ray, actors: [ray.kill(actor) for actor in actors] and [],
    )


def _drive_probe(monitor: RolloutWorkerHealthMonitor, actors: list[_Actor]) -> None:
    """Run one probe pass synchronously instead of racing the monitor thread."""

    del actors  # named at call sites to show which fleet is under test
    monitor._run_probes()


def test_interval_zero_disables_the_monitor() -> None:
    monitor = _monitor(_runtime(_Actor()), interval_s=0.0)

    assert monitor.start() is False
    assert monitor._thread is None


def test_healthy_workers_are_probed_within_the_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actors = [_Actor("rollout-0"), _Actor("rollout-1")]
    runtime = _runtime(*actors)
    ray = _FakeRay(actors)
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime, timeout_s=7.0)

    _drive_probe(monitor, actors)

    assert [actor.health.calls for actor in actors] == [1, 1]
    assert ray.get_timeouts == [7.0, 7.0]
    assert ray.killed == []
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


def test_unresponsive_worker_fails_the_runtime_and_kills_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung worker must terminalize the runtime, not just be logged.

    Killing the fleet is what unblocks the driver: its pending Ray calls raise
    RayActorError and the attempt unwinds into the existing terminal path.
    """

    healthy = _Actor("rollout-0")
    wedged = _Actor(TimeoutError("ray.get timed out"))
    actors = [healthy, wedged]
    runtime = _runtime(*actors)
    ray = _FakeRay(actors)
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime, timeout_s=0.5)

    _drive_probe(monitor, actors)

    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    failure = runtime.lifecycle.failure
    assert isinstance(failure, RolloutWorkerUnreachable)
    assert failure.worker_id == "rollout-1"
    assert failure.timeout_s == 0.5
    assert ray.killed == [healthy, wedged]


def test_probing_stops_after_the_first_unreachable_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fleet is already doomed; probing the rest would only delay the kill."""

    wedged = _Actor(TimeoutError("hung"))
    trailing = _Actor("rollout-1")
    actors = [wedged, trailing]
    ray = _FakeRay(actors)
    _install_ray(monkeypatch, ray)
    monitor = _monitor(_runtime(*actors))

    _drive_probe(monitor, actors)

    assert wedged.health.calls == 1
    assert trailing.health.calls == 0


def test_paused_monitor_does_not_probe_parked_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offloaded workers are intentionally silent; probing them is a false death."""

    actors = [_Actor(TimeoutError("parked"))]
    runtime = _runtime(*actors)
    ray = _FakeRay(actors)
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime, interval_s=0.01)
    monitor.pause()

    assert monitor.start() is True
    try:
        time.sleep(0.1)
    finally:
        monitor.stop()

    assert actors[0].health.calls == 0
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


def test_resume_rearms_the_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker woken from host RAM needs time to restore before answering."""

    monitor = _monitor(_runtime(_Actor()), first_wait_s=5.0)
    monitor._needs_first_wait = False

    monitor.resume()

    assert monitor._needs_first_wait is True


def test_stop_joins_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    actors = [_Actor("rollout-0")]
    ray = _FakeRay(actors)
    _install_ray(monkeypatch, ray)
    monitor = _monitor(_runtime(*actors), interval_s=0.01)

    assert monitor.start() is True
    monitor.stop()

    assert monitor._thread is None
    assert not any(
        thread.name == "RolloutWorkerHealthMonitor" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_workers_without_a_health_method_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local (non-Ray) worker fakes have no remote probe and must not fail closed."""

    runtime = SimpleNamespace(
        _owned_workers=[DistributedWorkerHandle(worker_id="local-0", actor=object())],
        lifecycle=RuntimeLifecycle(),
    )
    ray = _FakeRay([])
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime)

    monitor._run_probes()

    assert ray.killed == []
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
