"""Background liveness monitoring for Ray generation workers.

Operation deadlines bound active business RPCs but do not prove actor-process
reachability during idle windows. This complementary monitor probes every owned
worker on its own OS thread and, when a worker stops answering, retains the
failure and kills the fleet so an active or subsequent foreground operation
enters the existing terminal path.

Reachability is not residency. The worker's ``health`` method answers from its
own Ray concurrency group and touches no model or GPU state, so a parked
(offloaded) worker is exactly as probe-able as an active one -- and the trainer's
long GPU turns, when the workers are parked, are the idle windows this monitor
exists for. The monitor therefore never pauses across parking; the only probe it
discards is one for an actor the runtime no longer owns (a session replaced or
torn down while the probe was in flight).

Why a thread and not an asyncio task: the trainer's event loop is blocked for
the whole of each forward/backward step (it yields only between timesteps), so a
loop-resident monitor would both poll late and mis-measure its own timeout,
reporting a false expiry whose magnitude is the block duration. See
``vrl/rollouts/orchestration/continuous/owner.py`` for the same reasoning
applied to rollout admission.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from vrl.ray.dependencies import kill_actors, require_ray
from vrl.runtime_errors import TerminalRuntimeError
from vrl.utils.lifecycle import RuntimeLifecycleError

if TYPE_CHECKING:
    from vrl.generation.ray.runtime import RayGenerationRuntime
    from vrl.ray.actor_group import RayActorHandle

logger = logging.getLogger(__name__)

# Bounds how long stop() waits beyond one full probe cycle before reporting a
# stuck monitor thread rather than hanging the caller's shutdown.
_STOP_JOIN_GRACE_S = 5.0


class RolloutWorkerHealthMonitor:
    """Probe owned rollout workers and terminalize the runtime when one hangs."""

    def __init__(
        self,
        runtime: RayGenerationRuntime,
        *,
        interval_s: float,
        timeout_s: float,
    ) -> None:
        self._runtime = runtime
        self._interval_s = float(interval_s)
        self._timeout_s = float(timeout_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Serializes stop against a probe thread publishing a failure.
        self._transition_lock = threading.Lock()

    def start(self) -> bool:
        """Start probing. Returns False when monitoring is disabled or running.

        A fleet that has not been launched yet has no owned ranks, so the first
        probe passes find nothing to probe.
        """

        if self._interval_s <= 0:
            return False
        if self._thread is not None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="RolloutWorkerHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> bool:
        """Return whether the monitor thread has relinquished its resources."""
        thread = self._thread
        if thread is None:
            return True
        with self._transition_lock:
            self._stop.set()
        thread.join(timeout=self._timeout_s + self._interval_s + _STOP_JOIN_GRACE_S)
        if thread.is_alive():
            logger.warning("rollout health monitor thread did not exit; retaining it for join")
            return False
        self._thread = None
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._run_probes()
            self._stop.wait(timeout=self._interval_s)

    def _run_probes(self) -> None:
        workers = list(self._runtime._owned_ranks)
        if not workers:
            return
        try:
            ray = require_ray()
        except BaseException:
            logger.debug("rollout health probe skipped: Ray unavailable", exc_info=True)
            return
        for worker in workers:
            if self._stop.is_set():
                return
            try:
                # Bounded on the driver side, unlike a plain ray.get: a wedged
                # actor process must not also wedge its own monitor. An actor
                # with no probe face at all is unmonitorable, which is the same
                # thing this monitor exists to report, so its AttributeError
                # takes the same path as a failed probe rather than being
                # skipped -- a monitor that silently watches nothing is worse
                # than one that fails loudly.
                ray.get(worker.actor.health.remote(), timeout=self._timeout_s)
            except BaseException as error:
                self._terminalize(ray, worker, error)
                return

    def _terminalize(self, ray: Any, worker: RayActorHandle, error: BaseException) -> None:
        """Close admission and destroy actors so active/next foreground work fails."""

        failure = RolloutWorkerUnreachable(worker.worker_id, self._timeout_s, error)
        # Publish under the transition lock, never holding it across logging or
        # Ray control-plane calls. A probe that returns after stop, or for an
        # actor the runtime has since released (session replaced or torn down),
        # says nothing about the fleet the runtime owns now.
        with self._transition_lock:
            if self._stop.is_set():
                return
            if not any(owned.actor is worker.actor for owned in self._runtime._owned_ranks):
                return
            try:
                self._runtime.lifecycle.fail(failure, only_if_running=True)
            except RuntimeLifecycleError:
                return
        logger.error(
            "rollout worker %s failed its liveness probe after %.0fs; "
            "killing the fleet so active or subsequent runtime work fails closed",
            worker.worker_id,
            self._timeout_s,
            exc_info=error,
        )
        # Only synchronous work from this thread. lifecycle.fail atomically
        # closes admission; the async shutdown path belongs to whoever observes
        # the RayActorError that killing the actors raises in the driver.
        actors = [owned.actor for owned in self._runtime._owned_ranks]
        if actors:
            kill_actors(ray, actors)


class RolloutWorkerUnreachable(TerminalRuntimeError):
    """A rollout worker stopped answering its liveness probe."""

    def __init__(self, worker_id: str, timeout_s: float, cause: BaseException) -> None:
        self.worker_id = worker_id
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"rollout worker {worker_id!r} did not answer a liveness probe "
            f"within {self.timeout_s:g}s: {cause!r}",
        )
        self.__cause__ = cause


__all__ = ["RolloutWorkerHealthMonitor", "RolloutWorkerUnreachable"]
