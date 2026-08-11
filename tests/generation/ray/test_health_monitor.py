"""Background liveness probing of Ray rollout workers."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from vrl.generation.ray.health_monitor import (
    RolloutWorkerHealthMonitor,
    RolloutWorkerUnreachable,
)
from vrl.utils.lifecycle import RuntimeLifecycle, RuntimePhase
from vrl.ray.actor_group import RayActorHandle
from vrl.runtime_errors import (
    TerminalRuntimeError,
    failure_identity_cause,
    find_error_cause,
)

# Carried by every test that drives `_FakeRay`. The double is the Ray wire, not
# the monitor: scripting a probe's answer is how pause/stop/skip behaviour stays
# in the default lane at zero cost. The wire itself -- a real ray.get deadline
# expiring against a really blocked actor, and the RayActorError that killing the
# fleet raises in the driver -- is pinned by the real-cluster twin named here.
_SCRIPTED_RAY_WIRE = pytest.mark.real_cover(
    "tests/generation/ray/test_health_monitor.py"
    "::test_real_wedged_worker_times_out_and_the_fleet_really_dies",
    why=(
        "an in-process fake cannot block a real actor process, let a real ray.get(timeout=) "
        "expire, or make a healthy actor's pending call raise RayActorError once the fleet is "
        "killed; those three are what the slow_test twin runs for real"
    ),
)


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
        # Recorded, not defaulted: production's kill_actors runs for real against
        # this fake, so a caller that forgot no_restart=True would show up here.
        self.killed.append((actor, no_restart))


class _BlockingFailureRay(_FakeRay):
    """Hold one probe until a lifecycle transition has completed."""

    def __init__(self, actors: list[_Actor]) -> None:
        super().__init__(actors)
        self.probe_started = threading.Event()
        self.release_probe = threading.Event()

    def get(self, ref: _ProbeRef, timeout: float | None = None) -> Any:
        self.get_timeouts.append(float(timeout))
        self.probe_started.set()
        if not self.release_probe.wait(timeout=1):
            raise AssertionError("test did not release the blocked health probe")
        if isinstance(ref.behaviour, BaseException):
            raise ref.behaviour
        return ref.behaviour


class _ControlledStopEvent:
    """Expose two first-wait windows without sleeping in the test."""

    def __init__(self, *, grace_timeout: float) -> None:
        self._event = threading.Event()
        self._grace_timeout = grace_timeout
        self._lock = threading.Lock()
        self._grace_calls = 0
        self.grace_started = (threading.Event(), threading.Event())
        self.release_grace = (threading.Event(), threading.Event())

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        if timeout == self._grace_timeout:
            with self._lock:
                index = self._grace_calls
                self._grace_calls += 1
            if index < len(self.grace_started):
                self.grace_started[index].set()
                if not self.release_grace[index].wait(timeout=1):
                    raise AssertionError("test did not release the health grace period")
                return self._event.is_set()
        return self._event.wait(timeout=timeout)


def _runtime(*actors: _Actor) -> Any:
    return SimpleNamespace(
        _owned_workers=[
            RayActorHandle(worker_id=f"rollout-{i}", actor=actor) for i, actor in enumerate(actors)
        ],
        lifecycle=RuntimeLifecycle(),
    )


def _monitor(runtime: Any, **kwargs: Any) -> RolloutWorkerHealthMonitor:
    defaults = {"interval_s": 0.01, "timeout_s": 0.5, "first_wait_s": 0.0}
    defaults.update(kwargs)
    return RolloutWorkerHealthMonitor(runtime, **defaults)


def _install_ray(monkeypatch: pytest.MonkeyPatch, ray: _FakeRay) -> None:
    """Swap the Ray module for the scripted one -- and nothing else.

    Production's ``kill_actors`` runs for real here: ``_FakeRay.kill`` already
    accepts the ``no_restart=`` keyword it passes, so ``ray.killed`` records what
    the shipped cleanup loop did rather than what a lambda in this file did.
    """

    monkeypatch.setattr(
        "vrl.generation.ray.health_monitor.require_ray",
        lambda: ray,
    )


def _drive_probe(monitor: RolloutWorkerHealthMonitor, actors: list[_Actor]) -> None:
    """Run one probe pass synchronously instead of racing the monitor thread."""

    del actors  # named at call sites to show which fleet is under test
    monitor._run_probes(resume_epoch=monitor._resume_epoch)


def test_interval_zero_disables_the_monitor() -> None:
    """interval_s=0 is the off switch, and it must not leave a thread behind."""

    monitor = _monitor(_runtime(_Actor()), interval_s=0.0)

    assert monitor.start() is False
    assert monitor._thread is None


@_SCRIPTED_RAY_WIRE
def test_healthy_workers_are_probed_within_the_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every owned worker is probed once, and the configured timeout is the one
    handed to ray.get -- an unbounded get would wedge the monitor with the fleet."""

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


@_SCRIPTED_RAY_WIRE
def test_unresponsive_worker_fails_the_runtime_and_kills_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung worker must terminalize the runtime, not just be logged.

    Killing the fleet is what unblocks the driver: its pending Ray calls raise
    RayActorError and the attempt unwinds into the existing terminal path.
    """

    healthy = _Actor("rollout-0")
    probe_timeout = TimeoutError("ray.get timed out")
    wedged = _Actor(probe_timeout)
    actors = [healthy, wedged]
    runtime = _runtime(*actors)
    ray = _FakeRay(actors)
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime, timeout_s=0.5)

    _drive_probe(monitor, actors)

    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    failure = runtime.lifecycle.failure
    assert isinstance(failure, RolloutWorkerUnreachable)
    assert isinstance(failure, TerminalRuntimeError)
    assert failure.worker_id == "rollout-1"
    assert failure.timeout_s == 0.5
    assert failure.__cause__ is probe_timeout
    assert find_error_cause(failure, TimeoutError) is probe_timeout
    assert failure_identity_cause(failure) is failure
    # Production's kill_actors did this, in fleet order, with no_restart=True --
    # a restarting actor would answer the next probe and hide the failure.
    assert ray.killed == [(healthy, True), (wedged, True)]


@_SCRIPTED_RAY_WIRE
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


@_SCRIPTED_RAY_WIRE
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


@_SCRIPTED_RAY_WIRE
def test_pause_ignores_an_in_flight_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _Actor(TimeoutError("late timeout"))
    runtime = _runtime(actor)
    ray = _BlockingFailureRay([actor])
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime)
    probe_thread = threading.Thread(
        target=monitor._run_probes,
        kwargs={"resume_epoch": monitor._resume_epoch},
    )
    probe_thread.start()
    assert ray.probe_started.wait(timeout=1)

    monitor.pause()
    ray.release_probe.set()
    probe_thread.join(timeout=1)

    assert not probe_thread.is_alive()
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
    assert ray.killed == []


@_SCRIPTED_RAY_WIRE
def test_new_resume_ignores_a_probe_from_the_previous_active_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _Actor(TimeoutError("stale timeout"))
    runtime = _runtime(actor)
    ray = _BlockingFailureRay([actor])
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime)
    monitor.resume()
    probe_epoch = monitor._resume_epoch
    probe_thread = threading.Thread(
        target=monitor._run_probes,
        kwargs={"resume_epoch": probe_epoch},
    )
    probe_thread.start()
    assert ray.probe_started.wait(timeout=1)

    monitor.pause()
    monitor.resume()
    ray.release_probe.set()
    probe_thread.join(timeout=1)

    assert not probe_thread.is_alive()
    assert monitor._resume_epoch != probe_epoch
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
    assert ray.killed == []


@_SCRIPTED_RAY_WIRE
def test_stop_ignores_an_in_flight_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _Actor(TimeoutError("late timeout"))
    runtime = _runtime(actor)
    ray = _BlockingFailureRay([actor])
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime)

    assert monitor.start() is True
    monitor.resume()
    assert ray.probe_started.wait(timeout=1)
    stop_thread = threading.Thread(target=monitor.stop)
    stop_thread.start()
    assert monitor._stop.wait(timeout=1)
    ray.release_probe.set()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert monitor._thread is None
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING
    assert ray.killed == []


@_SCRIPTED_RAY_WIRE
def test_resume_during_first_wait_restarts_the_complete_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer wake cannot consume an older activation's partial grace."""

    actor = _Actor("rollout-0")
    ray = _FakeRay([actor])
    _install_ray(monkeypatch, ray)
    monitor = _monitor(_runtime(actor), first_wait_s=5.0)
    controlled_stop = _ControlledStopEvent(grace_timeout=5.0)
    monitor._stop = controlled_stop

    assert monitor.start() is True
    monitor.resume()
    assert controlled_stop.grace_started[0].wait(timeout=1)
    monitor.pause()
    monitor.resume()
    controlled_stop.release_grace[0].set()
    assert controlled_stop.grace_started[1].wait(timeout=1)

    assert actor.health.calls == 0
    stop_thread = threading.Thread(target=monitor.stop)
    stop_thread.start()
    assert controlled_stop.wait(timeout=1)
    controlled_stop.release_grace[1].set()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert actor.health.calls == 0
    assert monitor._thread is None


@_SCRIPTED_RAY_WIRE
def test_stop_joins_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """stop() is a real join, not a flag: a surviving daemon thread would keep
    probing a fleet the caller already tore down."""

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


@_SCRIPTED_RAY_WIRE
def test_workers_without_a_health_method_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local (non-Ray) worker fakes have no remote probe and must not fail closed."""

    runtime = SimpleNamespace(
        _owned_workers=[RayActorHandle(worker_id="local-0", actor=object())],
        lifecycle=RuntimeLifecycle(),
    )
    ray = _FakeRay([])
    _install_ray(monkeypatch, ray)
    monitor = _monitor(runtime)

    monitor._run_probes(resume_epoch=monitor._resume_epoch)

    assert ray.killed == []
    assert runtime.lifecycle.phase is RuntimePhase.RUNNING


# ------------------------------------------------------- real cluster (real Ray)


class _HealthyWorker:
    """Real Ray actor that answers its probe while a long driver call is in flight.

    ``generate()`` is the long call in production, and it is the thing the monitor
    exists to unblock, so the test needs one outstanding. Two concurrency slots
    keep ``health`` answerable while ``generate`` sits there, which is exactly how
    the real worker behaves (its probe is served off the actor's asyncio loop).
    """

    def health(self) -> str:
        return "ok"

    def generate(self) -> str:
        time.sleep(300)
        return "never"


class _WedgedWorker:
    """Real Ray actor whose probe never returns inside the driver's timeout."""

    def ready(self) -> str:
        return "ready"

    def health(self) -> str:
        time.sleep(300)
        return "never"


@pytest.mark.slow_test
def test_real_wedged_worker_times_out_and_the_fleet_really_dies(local_ray) -> None:
    """The whole mechanism on a live cluster: a real ``ray.get`` deadline expires
    against a genuinely blocked actor process, and the kill that follows unblocks
    the driver's outstanding call to the HEALTHY worker.

    That last part is the invariant the production module docstring states, and
    nothing in-process can show it: the attempt unwinds because the driver's
    pending Ray call starts raising ``RayActorError``, not because the wedged
    worker was singled out. Asserting it on a *pending* ref is also the only
    race-free way to assert it -- ``ray.kill`` is asynchronous, so a call
    submitted after the kill can still be answered by a not-yet-dead actor.
    """

    healthy = local_ray.remote(num_cpus=0, max_concurrency=2)(_HealthyWorker).remote()
    wedged = local_ray.remote(num_cpus=0)(_WedgedWorker).remote()
    assert local_ray.get(
        [healthy.health.remote(), wedged.ready.remote()],
        timeout=5,
    ) == ["ok", "ready"]
    handles = [
        RayActorHandle(worker_id="rollout-0", actor=healthy),
        RayActorHandle(
            worker_id="rollout-1",
            actor=wedged,
        ),
    ]
    runtime = SimpleNamespace(_owned_workers=handles, lifecycle=RuntimeLifecycle())
    monitor = _monitor(runtime, timeout_s=0.5)

    # The driver call the monitor has to rescue: 300s of work on a healthy worker.
    blocked_driver_call = healthy.generate.remote()

    started = time.monotonic()
    monitor._run_probes(resume_epoch=monitor._resume_epoch)
    elapsed = time.monotonic() - started

    # Without this, an actor that raised something else immediately would leave
    # the test just as green and "the timeout expired" would be unverifiable.
    assert elapsed >= 0.5
    assert runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
    failure = runtime.lifecycle.failure
    assert isinstance(failure, RolloutWorkerUnreachable)
    assert failure.worker_id == "rollout-1"
    with pytest.raises(local_ray.exceptions.RayActorError):
        local_ray.get(blocked_driver_call)
