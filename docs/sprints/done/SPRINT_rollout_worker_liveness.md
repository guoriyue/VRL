# SPRINT: Rollout worker process reachability and supervisor handoff

Status: **DONE (hardened 2026-07-22)**. A background health monitor probes owned rollout
worker processes out of band and kills the fleet when a health endpoint stops
answering, so that reachability failure unwinds to the supervisor when Ray's
actor-kill control path remains available.

## Decision

This sprint has one deliberately narrow outcome: an owned rollout worker process
that becomes unreachable must not leave the training attempt alive forever. When
its out-of-band health endpoint stops answering, the attempt fails closed,
releases its owned Ray resources, and exits non-zero. The existing repository
supervisor then starts a fresh process from the latest complete checkpoint.
Failures returned by `ray.kill` are retained and retried; a control-plane call
that itself never returns requires the separately deferred driver/GCS watchdog.

This is process/reachability health-monitor hardening, not a deadline for model
load, generation, capability, or weight-update RPCs. A reachable actor can keep
answering health probes while a business method in its default concurrency group
is blocked. The monitor therefore proves neither business-RPC completion nor
forward progress. A configured blocking-call deadline is still an unfinished,
independent reliability gate.

The accepted cost is losing work since that checkpoint. This is a correctness
and operability boundary, not a performance optimization.

There is deliberately no in-process actor recovery. The failed process does not
create a replacement fleet, retry the request, or continue the current
iteration.

### Superseded approach: per-operation deadlines

This sprint first explored a typed per-operation deadline prototype
(`RayOperationDeadline` / `RayOperationTimeout`) threaded through every Ray wait,
with a `distributed.rollout.worker_rpc_timeout_s` budget shared by all waits in
one operation. That prototype was reverted before it shipped and is not present
in the current implementation.

Why it was removed from this sprint: bounding each wait in-band required a
`deadline` argument in roughly fifteen
signatures across `vrl/ray/` and `vrl/generation/ray/`, plus a phase label at
every call site whose only consumer was the error message. The ongoing cost — a
concept every future Ray call site must thread — was judged too high for this
sprint's narrower process-reachability outcome. The out-of-band monitor replaces
only that outcome; it does not replace operation deadlines.

What the deadline version bought that the monitor does not: a bound on the
business operation itself, partial-result rejection at that operation boundary,
and a typed `RayOperationTimeout` in `run_verdict.json`. For an unreachable
process, the monitor instead kills the fleet, so the driver observes
`RayActorError` and the verdict reports that class. For a default-group business
method that is blocked while the health group still answers, the monitor does
nothing and no verdict is produced. Closing that gap remains the independent
configured blocking-call deadline gate.

What the monitor buys that deadlines did not: process-reachability coverage of
the idle window between operations. A per-operation budget only runs while an
operation is in flight; a worker that dies between requests is invisible to it
until the next request.

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

That concurrency boundary is intentionally asymmetric: the health group can
respond while the default group is busy or hung. A successful probe therefore
means only that the actor process and health handler are reachable. It is not
evidence that `execute_chunk`, `load_policy`, capability lookup, or weight-update
ACKs are making progress.

A plain `max_concurrency=2` was rejected: it applies to the whole actor, so it
would also let two chunks run concurrently on one GPU worker whenever
`max_inflight_chunks_per_worker` exceeds one. A concurrency group gives the probe
its own thread while the default group keeps its existing serialization.

## Starting boundary

The following ownership was already implemented when this sprint started and
remains authoritative:

- the rollout schedule owns admission and normal drain;
- the runtime owns activation, offload, and terminal cleanup;
- one collector-facing terminal state owns `OPEN -> CLOSING -> CLOSED`;
- `vrl.scripts.train` writes `run_verdict.json` when an error unwinds;
- `vrl.scripts.supervise` starts each attempt in its own process group and
  resumes the latest complete checkpoint after a failed attempt.

The supervisor was not an external hang detector: it waited for the child to
exit. These business waits were unbounded when this sprint started:

- actor policy load and worker metadata collection;
- worker capability queries and the placement metadata probe;
- generation ObjectRefs, including standard, dynamic, and pipelined dispatch;
- all-worker weight-update acknowledgements.

The monitor starts only after policy load, metadata, and capability checks have
completed and the runtime candidate is ready to publish. It therefore provides
no startup coverage for those earlier waits. Once monitoring is active, it can
detect an unreachable process during generation or weight update, but those
business waits are still not individually time-bounded. If the process and
health concurrency group remain responsive while the default-group call is
blocked, the monitor stays green and the wait can remain unresolved. The native
configured blocking-call deadline gate must cover both startup and active-runtime
blocking calls.

Placement-group readiness, the chunk-size probe, policy release, and worker
sleep/wake already have time bounds. This sprint may route those existing waits
through the same error path, but does not redesign their lifecycle or tune their
latency.

## Failure contract

```text
rollout worker misses its health probe
  -> publish RolloutWorkerUnreachable to the public terminal state
  -> close runtime admission and retain the background failure
  -> kill every actor through the fleet's single worker-ownership lock
  -> after a successful kill, blocked driver calls raise RayActorError and unwind
  -> force terminal cleanup of remaining owned actors and placement resources
  -> the next foreground command fails closed with the retained root cause;
     an idle-window failure is delivered by the final recipe cleanup read
  -> write a failed run verdict and exit non-zero
  -> supervisor resumes the latest complete checkpoint in a new process
```

Killing the fleet is the mechanism, not a side effect: it is what makes an
already-blocked `ray.get` on the driver return. A monitor that only logged would
leave the attempt hung.

Probing remains active while model weights are parked in host RAM. Parking changes
model residency, not actor-process reachability: the dedicated health concurrency
group continues to answer without touching model or GPU state. The monitor
therefore has no parking pause/resume state or wake-up grace period.

## Implementation scope

### 1. Config

Two fields under `distributed.rollout`, resolved once into
`RayGenerationConfig`: `health_check_interval_s` (default 30.0; `<= 0` disables
monitoring entirely) and `health_check_timeout_s` (default 30.0). The schema
validator rejects a non-finite interval and a non-positive timeout while
monitoring is enabled.

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

The monitor lives on `RayGenerationWorkerFleet`, the concrete owner of executor,
weight sync, actor handles, parking, and teardown. The fleet has no terminal FSM
and is not a second `GenerationRuntime`; it reports failures through a callback
to the one collector-facing `RayGenerationRuntime`. Resident launch creates one
fleet immediately. On-demand launch attaches a fleet candidate during the one
runtime's activation transition, and a final terminal admission check prevents
activation from succeeding if monitoring failed during candidate restore.

### 4. Finish the handoff

`RolloutWorkerUnreachable` names the worker and the elapsed probe budget. It is
recorded separately as `background_failure`, while the terminal state also
closes admission and retains the first overall failure. Foreground delivery is
in-band and needs no schedule-level polling: the next runtime command fails at
`require_open` with the retained root cause, and killing the fleet makes an
already-blocked driver `ray.get` raise `RayActorError`. The only window with no
foreground call left to fail is after the last command, so `RolloutCollector`
snapshots the failure after runtime shutdown and then releases both the runtime
and provider. Schedules expose that snapshot as a read-only property, and final
online cleanup performs the one explicit read, applying one priority: an
existing training error wins, then the background failure, then cleanup errors.
`shutdown()` itself stays resource cleanup, never an error-delivery API.

## Finishing criteria

### Deterministic CPU tests

- a worker that stops answering marks the runtime failed with
`RolloutWorkerUnreachable` naming that worker, and kills every owned actor;
- the driver-side probe wait is bounded, so a wedged actor cannot wedge its
  own monitor;
- probing stops at the first unreachable worker (the fleet is already doomed);
- parked workers remain reachable through the dedicated health concurrency
  group and monitoring continues across residency transitions;
- failed actor kills retain their handles and are retried without a tight loop;
- `interval_s <= 0` starts no thread at all;
- `stop()` joins the thread and leaves none running; a join timeout retains the
  live handle and permanently prevents a duplicate monitor from starting;
- concurrent `start()`/`stop()` cannot publish an unstarted thread handle;
- terminal transitions are serialized across the monitor and event-loop
  threads, so a late failure cannot resurrect a closed runtime;
- monitor join and actor kills do not block the asyncio loop;
- a missing actor handle or remote `health` endpoint is a terminal contract
  error rather than a production branch weakened for local fakes;
- monitor and shutdown cleanup share one locked actor owner, so overlapping
  paths issue one kill per handle;
- on-demand worker health failure closes the facade's one public terminal state;
- the collector snapshots the background failure, releases the provider and
  runtime, and final recipe cleanup still delivers an idle failure;
- existing activation, drain, shutdown, version-commit, OOM split/gather, and
  supervisor circuit-breaker tests remain green.

## Verification

Completed and revalidated on 2026-07-22 with the following repository checks:

- the full repository suite passed with `2710 passed, 17 skipped`;
- the broad generation/Ray/collector/orchestration/online selection passed with
  `531 passed`;
- generation-layer architecture and runtime-protocol boundaries passed with
  `22 passed`;
- real-Ray cases in that selection cover worker ownership, on-demand reuse, and
  the blocking default-group acceptance case;
- schema defaults, the three rollout presets, and `RayGenerationConfig.from_cfg`
  resolve the two fields, and both config boundaries reject a non-finite
  interval and a non-positive timeout while enabled;
- the real-Ray concurrency-group test blocks one default-group call, confirms a
  health call still returns, and confirms a second default-group call remains
  serialized;
- Ruff check and format passed for every touched Python file.

## Explicitly deferred

- configured deadlines for provider startup, capability, generation, and
  weight-update blocking calls, including partial-result rejection at the
  operation boundary;
- in-process actor restart or fleet rebuild;
- fleet epoch/identity, stale-event guards, or recovery-time candidate replacement;
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

- add the two health-check config fields and their behavior consumers;
- add the out-of-band worker probe and the actor concurrency it needs;
- add the continuously running monitor thread and its tests.
- make monitor start/stop transitions lock-protected, retain failed kill
  ownership, and serialize the shared terminal state;
- give health and shutdown one canonical actor-kill owner;
- snapshot the background failure in the collector while releasing the runtime
  and read it once in final recipe cleanup (foreground delivery stays in-band
  via terminal admission);
- keep Ray kill and monitor join off the asyncio loop;
- centralize cleanup-note attachment for the supported Python 3.10 boundary.

### Keep unchanged

- `Runtime -> WorkerFleet -> Executor -> Ray actor adapter -> WorkerCore` remains the real
  execution/process boundary, with `RayGenerationWorkerFleet` owning the live
  executor and actor resources below the public runtime;
- `run_actor_jobs`, `RayActorGroup`, `GenerationWeightSync`, the launcher, and
  terminal state remain thin because they are shared dispatch, framework,
  protocol, public, and state-machine boundaries;
- schedule-owned drain, current integer version acknowledgements, transactional
  version publication, and trainer trajectory replay remain unchanged;
- `RUN_VERDICT_NAME`, checkpoint filenames, environment-variable names, and
  terminal enum values remain valid protocol constants.
- `HEALTH_CONCURRENCY_GROUP` remains an ALL_CAPS constant because the decorator
  and Ray actor creation APIs share it as a protocol name;
- `_STOP_JOIN_GRACE_S` remains an ALL_CAPS cleanup boundary rather than domain
  vocabulary;
- `health_monitor.py`, `RuntimeTerminalState`, and the thin `health()` method remain
  separate because they are thread, state-machine, and Ray adapter boundaries;
- `worker_fleet.py` remains separate because actor inventory, monitor, parking,
  weight sync, and retryable teardown form one real resource-ownership boundary;
- `request_stop()` and `stop()` remain two thin methods because non-blocking
  stop signaling and thread joining are different lifecycle operations;
- `add_cleanup_note()` remains a thin compatibility boundary that prevents
  cleanup diagnostics from replacing the primary error on Python 3.10.

Do not add `WorkerFleetManager`, `RecoveryHandler`, duplicated field-name sets,
or an ALL_CAPS timeout table. Do not smuggle a blocking-call deadline into this
completed process-health sprint; that unfinished requirement belongs to its own
configured reliability gate. Cross-family boundary consistency is more valuable
than flattening these existing thin adapters for fewer lines.

## References

- `vrl/generation/ray/health_monitor.py`
- `vrl/generation/ray/worker.py`
- `vrl/ray/actor_group.py`
- `vrl/ray/placement.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/worker_fleet.py`
- `vrl/generation/ray/utils.py`
- `vrl/generation/ray/launcher.py`
- `vrl/config/schema.py`
- `vrl/utils/exceptions.py`
- `vrl/scripts/train.py`
- `vrl/scripts/supervise.py`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
- `docs/sprints/SPRINT_native_generation_engine_program.md`
