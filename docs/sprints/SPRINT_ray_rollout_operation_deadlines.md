# SPRINT: Bounded Ray rollout operations and supervisor handoff

Status: **active (2026-07-18)**.

## Decision

This sprint has one liveness outcome: a rollout worker RPC must not keep a
training attempt alive forever. When a worker operation exceeds its configured
deadline, the current attempt fails closed, releases its owned Ray resources,
and exits non-zero. The existing repository supervisor then starts a fresh
process from the latest complete checkpoint.

The accepted cost is losing work since that checkpoint. This is a correctness
and operability boundary, not a performance optimization.

There is deliberately no in-process actor recovery. The failed process does not
create a replacement fleet, retry the request, or continue the current
iteration.

## Current boundary

The following ownership is already implemented and remains authoritative:

- the rollout schedule owns admission and normal drain;
- the runtime owns activation, offload, and terminal cleanup;
- terminal state remains `RUNNING -> SHUTTING_DOWN -> TERMINATED`;
- `vrl.scripts.train` writes `run_verdict.json` when an error unwinds;
- `vrl.scripts.supervise` starts each attempt in its own process group and
  resumes the latest complete checkpoint after a failed attempt.

The supervisor is not an external hang detector: it waits for the child to
exit. The missing boundary is therefore inside the child. These worker waits
can currently remain unresolved indefinitely:

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
worker RPC exceeds its deadline
  -> reject all output from the current operation/request
  -> best-effort cancel every outstanding ObjectRef
  -> close runtime admission
  -> force terminal cleanup of all owned actors and placement resources
  -> preserve the timeout as the root failure
  -> write a failed run verdict and exit non-zero
  -> supervisor resumes the latest complete checkpoint in a new process
```

Already completed chunks are not returned after a request timeout. A weight
update timeout leaves the installed state unknown, so the current version is not
advanced and the runtime is terminated. Timeout cleanup must not wait for a
graceful actor RPC from the actor that just stopped responding.

`ray.cancel` is best-effort. Correctness comes from refusing partial output and
destroying the owned runtime before process exit, not from assuming cancellation
succeeded.

## Implementation scope

### 1. One typed timeout source

Add one finite, positive `distributed.rollout.worker_rpc_timeout_s` field and
resolve it once into `RayGenerationConfig`. Use a conservative default matching
the existing 600-second worker probe bound; recipes with legitimately longer
worker calls can override it in YAML.

Each top-level startup, generation, and weight-update operation receives one
wall-clock budget. Sequential waits consume the remaining budget instead of
resetting the full timeout for every worker or phase. The placement metadata
probe may reuse its owner's existing readiness budget; it must not remain
unbounded.

Do not add a `ReliabilityConfig`, a timeout matrix, or module-level timeout
constants. The resolved field must be passed to real wait/control-flow
consumers; a log-only config field does not count as implementation.

### 2. One stable timeout error

Use one `RayOperationTimeout(TimeoutError)` protocol error carrying the stable
operation name, configured timeout, and available request/worker context. Its
class becomes the existing supervisor's same-cause verdict key.

Do not create worker/cluster failure taxonomies, retryable flags, or recovery
result types. Existing Ray exceptions continue to propagate through the same
terminal path.

### 3. Bound every worker wait

Apply the budget at the current owners:

- `RayActorGroup` for startup/load and metadata;
- the placement owner for its worker metadata probe;
- the capability query for versioned slots;
- `run_actor_jobs` and the direct pipelined path for generation;
- `RayGenerationWeightSync` for the all-worker acknowledgement barrier.

On timeout, cancel asyncio waiters and call best-effort `ray.cancel` for all
submitted, incomplete refs. The generation path must discard its complete and
partial results together. Startup must kill every actor it created before a
candidate runtime becomes visible. Weight sync must reuse the existing failed
update terminal path and must not publish the candidate policy version.

### 4. Finish the handoff

The timeout path closes admission and performs forceful terminal cleanup without
calling a potentially hung worker for graceful release. Cleanup errors may be
attached to the timeout but must not replace it as the run verdict's root cause.

The child must exit after the operation deadline plus a short, bounded cleanup
grace while the Ray driver/control plane remains responsive. A driver, GCS,
raylet, `ray.kill`, or `ray.shutdown` hang is a separate process-watchdog problem
and is not silently claimed by this sprint.

## Finishing criteria

### Deterministic CPU tests

- never-resolving load, metadata, capability, placement-probe, generation, and
  weight-update refs raise `RayOperationTimeout` within the controlled deadline;
- every submitted incomplete ref receives best-effort cancellation and every
  asyncio waiter is collected;
- a generation timeout returns no partial chunk or group;
- standard round-robin, dynamic placement, and pipelined dispatch share the
  same timeout semantics;
- an update timeout preserves the previously committed policy version, closes
  admission, and kills the owned runtime;
- terminal cleanup preserves the timeout root cause even when cleanup also
  reports an error;
- normal completion before the deadline retains existing behavior;
- the timeout verdict causes the supervisor to launch the next attempt with
  `trainer.resume_from=<latest complete checkpoint>` and `model.lora.path=`;
- existing activation, drain, shutdown, version-commit, OOM split/gather, and
  supervisor circuit-breaker tests remain green.

### Isolated real-Ray CPU test

Run one subprocess test through the production dispatch boundary with an actor
method that blocks. Assert deadline, no partial result, cancellation attempt,
owned-actor cleanup, non-zero child exit, and supervisor checkpoint resume. Use
an isolated local Ray session and `ray.shutdown()`; never use `ray stop`.

This is a correctness test. It requires no GPU, chaos matrix, throughput report,
or sleep/wake/relaunch comparison.

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

- add the single typed timeout field and its behavior consumers;
- add the stable timeout protocol error;
- add deadline, cancellation, terminal-handoff, and supervisor tests.

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
or an ALL_CAPS timeout table. Cross-family boundary consistency is more valuable
than flattening these existing thin adapters for fewer lines.

## References

- `vrl/ray/actor_pool.py`
- `vrl/ray/actor_group.py`
- `vrl/ray/placement.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/utils.py`
- `vrl/generation/ray/weight_sync.py`
- `vrl/scripts/train.py`
- `vrl/scripts/supervise.py`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
