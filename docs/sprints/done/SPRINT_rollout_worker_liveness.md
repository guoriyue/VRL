# SPRINT: Bounded rollout worker liveness and supervisor handoff

Status: **DONE (2026-07-19)**. A background liveness monitor probes owned rollout
workers and kills the fleet when one stops answering, so the attempt fails closed
and the supervisor resumes from the latest complete checkpoint.

## Decision

This sprint has one liveness outcome: a rollout worker RPC must not keep a
training attempt alive forever. When a worker stops answering, the attempt fails
closed, releases its owned Ray resources, and exits non-zero. The existing
repository supervisor then starts a fresh process from the latest complete
checkpoint.

The accepted cost is losing work since that checkpoint. This is a correctness
and operability boundary, not a performance optimization.

There is deliberately no in-process actor recovery. The failed process does not
create a replacement fleet, retry the request, or continue the current
iteration.

### Superseded approach: per-operation deadlines

This sprint was first implemented as a typed per-operation deadline
(`RayOperationDeadline` / `RayOperationTimeout`) threaded through every Ray wait,
with a `distributed.rollout.worker_rpc_timeout_s` budget shared by all waits in
one operation. That version was reverted before it shipped.

Why: bounding each wait in-band required a `deadline` argument in roughly fifteen
signatures across `vrl/ray/` and `vrl/generation/ray/`, plus a phase label at
every call site whose only consumer was the error message. The ongoing cost — a
concept every future Ray call site must thread — was judged too high for the
liveness outcome it bought, which an out-of-band probe delivers on its own.

What the deadline version bought that the monitor does not: a typed
`RayOperationTimeout` in `run_verdict.json`, distinguishable from a genuine actor
crash. The monitor instead kills the fleet, so the driver observes `RayActorError`
and the verdict reports that class. The supervisor's "same class twice -> stop"
policy still terminates a repeating hang; it can no longer tell a hang from a
crash. That was accepted.

What the monitor buys that deadlines did not: coverage of the idle window between
operations. A per-operation budget only runs while an operation is in flight; a
worker that dies between requests is invisible to it until the next request.

Prior art: slime's `RolloutHealthMonitor`
(`slime/utils/health_monitor.py`). One deviation is deliberate — slime wraps its
probe in an unbounded `ray.get(engine.health_generate.remote(timeout=...))`, where
the timeout reaches the actor but not the driver, so a wedged actor process also
wedges its monitor. This implementation bounds the driver side:
`ray.get(ref, timeout=...)`.

Second deviation, forced by the runtime model: slime probes an HTTP endpoint that
SGLang serves out-of-band, while a Ray actor is single-threaded by default, so a
probe would queue behind `execute_chunk` and measure queue depth rather than
liveness. Rollout actors therefore declare a Ray concurrency group for
`RayGenerationWorker.health()`, which touches no model or GPU state.

A plain `max_concurrency=2` was rejected: it applies to the whole actor, so it
would also let two chunks run concurrently on one GPU worker whenever
`max_inflight_chunks_per_worker` exceeds one. A concurrency group gives the probe
its own thread while the default group keeps its existing serialization.

## Starting boundary

The following ownership was already implemented when this sprint started and
remains authoritative:

- the rollout schedule owns admission and normal drain;
- the runtime owns activation, offload, and terminal cleanup;
- terminal state remains `RUNNING -> SHUTTING_DOWN -> TERMINATED`;
- `vrl.scripts.train` writes `run_verdict.json` when an error unwinds;
- `vrl.scripts.supervise` starts each attempt in its own process group and
  resumes the latest complete checkpoint after a failed attempt.

The supervisor was not an external hang detector: it waited for the child to
exit. The missing boundary was therefore inside the child. These worker waits
could remain unresolved indefinitely before this sprint:

- actor policy load and worker metadata collection;
- worker capability queries and the placement metadata probe;
- generation ObjectRefs, including standard, dynamic, and pipelined dispatch;
- all-worker weight-update acknowledgements.

Placement-group readiness, the chunk-size probe, policy release, and worker
sleep/wake already have time bounds. This sprint may route those existing waits
through the same error path, but does not redesign their lifecycle or tune their
latency.

## Failure contract

```text
rollout worker misses its liveness probe
  -> close runtime admission (lifecycle.fail)
  -> kill every owned worker actor
  -> blocked driver calls raise RayActorError and unwind
  -> force terminal cleanup of remaining owned actors and placement resources
  -> write a failed run verdict and exit non-zero
  -> supervisor resumes the latest complete checkpoint in a new process
```

Killing the fleet is the mechanism, not a side effect: it is what makes an
already-blocked `ray.get` on the driver return. A monitor that only logged would
leave the attempt hung.

Probing pauses whenever workers are parked in host RAM. An offloaded worker is
intentionally unresponsive, so probing it would read as a death and kill a
healthy fleet. The rollout schedule owns those transitions; `sleep_workers`
pauses and `wake_workers` resumes, and every resume re-arms the first-probe grace
period so a restoring worker is not probed mid-wake.

## Implementation scope

### 1. Config

Three fields under `distributed.rollout`, resolved once into
`RayGenerationConfig`: `health_check_interval_s` (default 30.0; `<= 0` disables
monitoring entirely), `health_check_timeout_s` (default 30.0), and
`health_check_first_wait_s` (default 0.0). The schema validator rejects a
non-finite interval, a non-positive timeout while monitoring is enabled, and a
negative first wait.

`distributed.rollout.worker_rpc_timeout_s` was removed with the deadline
mechanism; it had no remaining reader.

### 2. Out-of-band probe

`RayGenerationWorker.health()` returns the worker id and touches no model or GPU
state. It is bound to its own Ray concurrency group, so it answers on a dedicated
actor thread while generation occupies the default group — verified against a
real Ray session: the probe returns immediately while `execute_chunk` is mid
flight, and two default-group calls still serialize.

### 3. Monitor thread

`RolloutWorkerHealthMonitor` owns one daemon OS thread. It must not be an asyncio
task: the trainer's loop is blocked for the whole of each forward/backward step
and yields only between timesteps, so a loop-resident monitor would both poll
late and mis-measure its own timeout, reporting a false expiry whose magnitude is
the block duration. The driver-side wait is `ray.get(ref, timeout=...)`.

The monitor lives on the runtime that owns actors. The on-demand facade owns
none, so its monitor stays disabled; the inner runtime the launcher builds
carries the live one.

### 4. Finish the handoff

`RolloutWorkerUnreachable` names the worker and the elapsed probe budget. It is
recorded through `lifecycle.fail` so terminal cleanup keeps it as the root cause;
the run verdict then reports whichever class reaches the top of the unwind.

## Finishing criteria

### Deterministic CPU tests

- a worker that stops answering marks the runtime failed with
  `RolloutWorkerUnreachable` naming that worker, and kills every owned actor;
- the driver-side probe wait is bounded, so a wedged actor cannot wedge its
  own monitor;
- probing stops at the first unreachable worker (the fleet is already doomed);
- a paused monitor never probes parked workers, and every resume re-arms the
  first-probe grace period;
- `interval_s <= 0` starts no thread at all;
- `stop()` joins the thread and leaves none running;
- workers whose actor exposes no remote `health` are skipped rather than
  treated as dead (local worker fakes);
- existing activation, drain, shutdown, version-commit, OOM split/gather, and
  supervisor circuit-breaker tests remain green.

## Verification

Completed on 2026-07-19 with the following repository checks:

- `tests/generation`, `tests/ray`, `tests/rollouts`, `tests/config`, and
  `tests/scripts` passed;
- `tests/generation/ray/test_health_monitor.py` covers the criteria above and
  was run repeatedly to confirm it is not timing-flaky;
- schema defaults, the three rollout presets, and `RayGenerationConfig.from_cfg`
  resolve the new fields, and the validator rejects a non-finite interval, a
  non-positive timeout while enabled, and a negative first wait;
- Ruff check and format passed for every touched Python file.

## Explicitly deferred

- in-process actor restart or fleet rebuild;
- `WorkerFleet`, fleet epoch/identity, stale-event guards, and candidate publish;
- deterministic whole-request retry/replay after actor failure;
- actor-level retry budgets, backoff, and recovery circuit breakers;
- state-digest/schema ACK recovery protocols or committed-version history;
- public `RECOVERING`/`QUIESCING` phases or `OperationTicket`;
- actor death-record taxonomies, named-actor recovery identity, and GPU chaos;
- trainer/FSDP rank recovery;
- driver, GCS, raylet, or Ray shutdown watchdogs;
- any performance comparison or lifecycle retuning.

In-process recovery should be reconsidered only after production evidence shows
that worker-specific failures multiplied by checkpoint rollback cost consume a
material share of training. No speculative recovery design is retained in the
meantime.

## Architecture guardrails

### Change

- add the three health-check config fields and their behavior consumers;
- add the out-of-band worker probe and the actor concurrency it needs;
- add the monitor thread, its pause/resume wiring, and its tests.

### Keep unchanged

- `Runtime -> Executor -> Ray actor adapter -> WorkerCore` remains the real
  protocol/process boundary;
- `run_actor_jobs`, `RayActorGroup`, `GenerationWeightSync`, the launcher, and
  lifecycle FSM remain thin because they are shared dispatch, framework,
  protocol, public, and state-machine boundaries;
- schedule-owned drain, current integer version acknowledgements, transactional
  version publication, and trainer trajectory replay remain unchanged;
- `RUN_VERDICT_NAME`, checkpoint filenames, environment-variable names, and
  lifecycle enum values remain valid protocol constants.

Do not add `WorkerFleetManager`, `RecoveryHandler`, duplicated field-name sets,
an ALL_CAPS timeout table, or a per-operation deadline threaded through the Ray
call graph (see the superseded approach above). Cross-family boundary consistency is more valuable
than flattening these existing thin adapters for fewer lines.

## References

- `vrl/generation/ray/health_monitor.py`
- `vrl/generation/ray/worker.py`
- `vrl/ray/actor_group.py`
- `vrl/ray/placement.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/utils.py`
- `vrl/generation/ray/launcher.py`
- `vrl/config/schema.py`
- `vrl/scripts/train.py`
- `vrl/scripts/supervise.py`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
