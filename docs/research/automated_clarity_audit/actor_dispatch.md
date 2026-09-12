# Shared actor admission and transport deadlines

Reviewed complete actor_pool.py and operation_deadline.py, generation executor
and weight-sync consumers, cancellation/admission regressions, and real CPU Ray
dispatch tests. Previous isolated audit commit: d5d4e4f33.

## Change

Remove the separate CancelledError catch beside an identical BaseException catch
when inspecting already-completed calls during cancellation. BaseException
already includes CancelledError. Failure capture and subsequent classification
are unchanged; no new checker, wrapper, or test is needed for this consolidation.

## Retain and why

- RayActorDispatcher owns fleet admission across callers. Moving its state into
  individual execute/sync methods would permit mailbox queues outside the driver
  deadline. Pending work intentionally has no execution deadline until admitted.
- _AdmissionWaiter has identity equality because equal candidate sets do not
  mean equal queue positions. The identity removal helper, first-waiter lookup,
  candidate synchronization and removal all serve shared per-worker FIFO state.
  They are not independent business operations needing additional owner classes.
- _try_acquire mutates admission; _can_acquire only observes it while waiting.
  Combining them would either acquire too early or add a behavior flag that hides
  this difference. _register, _finish and _release track separate ref/slot events;
  _close retains the first terminal error and wakes other callers.
- run's local dispatch, submission, waiter spawning and success helpers share one
  invocation's jobs, telemetry and refs. Moving them into dispatcher instance
  fields would mix concurrent runs; adding a separate session object solely to
  remove nested functions is not justified by this review.
- run_one is the progress-aware protocol adapter used by pipelined generation.
  Its custom waiter differs from run's fixed-deadline job set. Preserve these
  two APIs and their explicit cancellation decisions. In particular, run can
  return completed results when cancellation races with all-success completion,
  allowing weight-sync ACK processing; run_one instead propagates ordinary
  cancellation after a completed result and releases the known-idle slot.
- Worker identity, callable submission, duplicate ref and double release checks
  protect routing and physical slot ownership. This is an external/framework
  boundary with multiple callers, not a repeated tensor dtype check to remove.
- cancel_ray_refs and get_ray_refs are shared Ray transport boundaries for
  executor, engine, startup and placement. Their free-function shape is useful:
  they own no persistent state. Cancellation preserves the operation root and
  expands aggregate engine refs without importing generation into generic Ray.
  Best-effort cancellation does not promise to stop synchronous GPU kernels;
  actor teardown belongs to the runtime owner.
- RayCallDeadline and RayOperationTimeout specialize the existing generic
  deadline core, keeping transport labels/error types out of that core. Keep
  the thin subclasses for cross-transport consistency. __all__ is the public
  facade; neither module contains an ALL_CAPS business vocabulary to relocate.

Non-goals: a scheduler redesign, cross-thread dispatcher support, forced kernel
cancellation, changing FIFO/LPT policy, or deleting lifecycle guards for LOC.
The dispatcher assumes one owning event loop. This review does not establish
atomic fleet weight installation or a GPU throughput improvement.

## Validation

45 batch-dispatch, operation-deadline and actor-pool tests passed. Coverage
includes cancellation versus completed success/failure, cancellation while
queued, middle-waiter identity, partial submission failure, real ObjectRef
awaiting, and two real synchronous actor calls each receiving their own budget
after admission. Ruff check and format check passed for actor_pool.py.
