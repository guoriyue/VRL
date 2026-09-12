# Health monitor thread ownership

Reviewed complete `generation/ray/health_monitor.py`, runtime start/teardown
callers, pause/resume/stop race tests and the real wedged-worker test. Runtime's
larger lifecycle module still awaits a complete review.

## Change

After a bounded join timed out, stop discarded `_thread` even though it remained
alive. A later start could then clear the shared stop event and create a second
thread while the first was still returning from a blocked probe. This violated
start's documented refusal to start an already-running monitor.

Keep the live thread reference when join times out. Leave stop set and let a
subsequent stop join/clear that same thread. The wait remains bounded; no new
exception or retry thread is introduced. Ordinary successful stop is unchanged.
Restart after a timed-out stop requires completing stop once the thread exits;
it no longer treats a lost handle as proof that no thread exists.

A real-thread regression blocks the existing fake Ray probe across the short
join budget. It failed before the change because `_thread` was None while the
captured thread was alive. It now verifies retained identity, start refusal,
the still-set stop event and successful later join. Cleanup always releases and
joins the controlled thread, including on assertion failure.

## Retain and why

- The monitor uses an OS thread because a training step can block the async
  event loop. Moving it onto that loop would change liveness timing semantics.
  The health actor method's independent concurrency group is complementary.
- Pause, resume and stop share a transition lock with failure publication.
  `_resume_epoch` rejects late failures from an earlier active window, and
  `_completed_grace_epoch` prevents an older wait from consuming a later resume's
  grace. These are state-machine facts, not redundant integer checks.
- `_run_probes` owns one bounded probe pass; `_terminalize` owns synchronized
  failure publication and fleet kill. Keep the lock out of Ray control-plane
  calls. Joining these methods would obscure the differing lock boundaries.
- RolloutWorkerUnreachable carries a terminal domain identity plus the original
  cause. A dependency TimeoutError alone does not describe a lost fleet.
- Kill failures remain with session/runtime ownership for later teardown; the
  monitor does not clear actor handles. Existing skip behavior for unavailable
  Ray or actors without a health endpoint is retained, not expanded.
- `_STOP_JOIN_GRACE_S` is the deliberately isolated join-limit policy value,
  separate from configurable probe interval/timeout. Export names are API
  declarations. No misplaced business vocabulary table exists here.

Non-goals: promise preemption of ray.get or synchronous GPU work, redesign
restart policy, enable arbitrary concurrent start/stop callers, remove resume
epochs, or claim a bounded stop guarantees its thread has exited. Runtime may
continue teardown with stop set; retaining the handle prevents an accidental
restart from reviving that old thread.

## Validation

107 tests passed across health monitoring, runtime lifecycle and parking leases,
including the real CPU Ray wedged-worker test; one upstream Ray warning.
Ruff check and format check
passed for both changed Python files. The real-thread fault test controls the
wire's blocking behavior, not a live Ray control-plane outage.
Previous isolated audit commit: `c2a786b7e`.
