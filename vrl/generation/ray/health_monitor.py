"""Background liveness monitoring for Ray generation workers.

Operation deadlines bound active business RPCs but do not prove actor-process
reachability during idle windows. This complementary monitor probes every owned
worker on its own OS thread and, when a worker stops answering, retains the
failure and kills the fleet so an active or subsequent foreground operation
enters the existing terminal path.

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

from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.ray.dependencies import require_ray
from vrl.ray.resource_cleanup import kill_actors
from vrl.runtime_errors import TerminalRuntimeError

if TYPE_CHECKING:
    from vrl.generation.ray.runtime import RayGenerationRuntime

logger = logging.getLogger(__name__)

# Bounds how long stop() waits beyond one full probe cycle before reporting a
# stuck monitor thread rather than hanging the caller's shutdown.
_STOP_JOIN_GRACE_S = 5.0


class RolloutWorkerHealthMonitor:
    """Probe owned rollout workers and terminalize the runtime when one hangs.

    The rollout schedule pauses probing across parking transitions and parked
    intervals because they are explicitly outside the active serving SLA. This
    is a lifecycle policy, not a claim that the lightweight health method cannot
    technically answer while model state is offloaded.
    """

    def __init__(
        self,
        runtime: RayGenerationRuntime,
        *,
        interval_s: float,
        timeout_s: float,
        first_wait_s: float = 0.0,
    ) -> None:
        self._runtime = runtime
        self._interval_s = float(interval_s)
        self._timeout_s = float(timeout_s)
        self._first_wait_s = float(first_wait_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._transition_lock = threading.Lock()
        self._resume_epoch = 0
        self._completed_grace_epoch = -1

    def start(self) -> bool:
        """Start probing. Returns False when monitoring is disabled or running."""

        if self._interval_s <= 0:
            return False
        if self._thread is not None:
            return False
        self._stop.clear()
        # Start paused: workers may not be activated yet, and probing a fleet
        # that has not been launched would kill it before its first request.
        self._paused.set()
        self._thread = threading.Thread(
            target=self._loop,
            name="RolloutWorkerHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._transition_lock:
            self._stop.set()
            # Clear the pause gate too, so a paused thread can observe the stop.
            self._paused.clear()
        thread.join(timeout=self._timeout_s + self._interval_s + _STOP_JOIN_GRACE_S)
        if thread.is_alive():
            logger.warning("rollout health monitor thread did not exit; abandoning it")
        self._thread = None

    def pause(self) -> None:
        with self._transition_lock:
            self._paused.set()

    def resume(self) -> None:
        # Re-arm the grace period on every resume: workers woken from host RAM
        # need time to restore before they can answer a probe.
        with self._transition_lock:
            self._resume_epoch += 1
            self._paused.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._transition_lock:
                paused = self._paused.is_set()
                resume_epoch = self._resume_epoch
                needs_grace = self._completed_grace_epoch != resume_epoch
            if paused:
                self._stop.wait(timeout=0.5)
                continue
            if needs_grace:
                if self._first_wait_s > 0 and self._stop.wait(timeout=self._first_wait_s):
                    return
                with self._transition_lock:
                    if (
                        self._stop.is_set()
                        or self._paused.is_set()
                        or self._resume_epoch != resume_epoch
                    ):
                        # A new resume owns a complete grace period; never let an
                        # older wait consume or shorten it.
                        continue
                    self._completed_grace_epoch = resume_epoch
            self._run_probes(resume_epoch=resume_epoch)
            self._stop.wait(timeout=self._interval_s)

    def _owned_workers(self) -> list[DistributedWorkerHandle]:
        """Read the active session fleet through the runtime adapter."""

        return [worker for worker in self._runtime._owned_workers if worker.actor is not None]

    def _run_probes(self, *, resume_epoch: int) -> None:
        workers = self._owned_workers()
        if not workers:
            return
        try:
            ray = require_ray()
        except BaseException:
            logger.debug("rollout health probe skipped: Ray unavailable", exc_info=True)
            return
        for worker in workers:
            with self._transition_lock:
                if (
                    self._stop.is_set()
                    or self._paused.is_set()
                    or self._resume_epoch != resume_epoch
                ):
                    return
            probe = getattr(worker.actor, "health", None)
            remote = getattr(probe, "remote", None)
            if not callable(remote):
                continue
            try:
                # Bounded on the driver side, unlike a plain ray.get: a wedged
                # actor process must not also wedge its own monitor.
                ray.get(remote(), timeout=self._timeout_s)
            except BaseException as error:
                self._terminalize(
                    ray,
                    worker.worker_id,
                    error,
                    resume_epoch=resume_epoch,
                )
                return

    def _terminalize(
        self,
        ray: Any,
        worker_id: str,
        error: BaseException,
        *,
        resume_epoch: int,
    ) -> None:
        """Close admission and destroy actors so active/next foreground work fails."""

        failure = RolloutWorkerUnreachable(worker_id, self._timeout_s, error)
        # A probe submitted before pause/stop may return late. Publish the
        # failure under the same lock as those transitions, but never hold that
        # lock across logging or Ray control-plane calls.
        with self._transition_lock:
            if self._stop.is_set() or self._paused.is_set() or self._resume_epoch != resume_epoch:
                return
            try:
                self._runtime.lifecycle.fail(failure)
            except BaseException:
                logger.debug("lifecycle.fail rejected the probe failure", exc_info=True)
            self._paused.set()
        logger.error(
            "rollout worker %s failed its liveness probe after %.0fs; "
            "killing the fleet so active or subsequent runtime work fails closed",
            worker_id,
            self._timeout_s,
            exc_info=error,
        )
        # Only synchronous work from this thread. lifecycle.fail atomically
        # closes admission; the async shutdown path belongs to whoever observes
        # the RayActorError that killing the actors raises in the driver.
        actors = [worker.actor for worker in self._owned_workers()]
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
