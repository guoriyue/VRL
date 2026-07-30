# SPRINT: Rollout worker process reachability and supervisor handoff

Status: **DONE (2026-07-22; documentation reconciled 2026-07-30)**.

## Decision

This sprint added an out-of-band health monitor for owned rollout worker
processes. If a worker stops answering its dedicated health endpoint, the
runtime closes admission, retains the failure, and kills its owned actor fleet.
That releases an active or subsequently attempted foreground operation into the
existing terminal path.

The accepted cost is losing work since the latest complete checkpoint. There is
no in-process actor replacement, request replay, or continuation of the current
iteration.

Health and operation progress are deliberately separate facts:

- the health monitor proves actor-process and health-handler reachability;
- the later
  [Ray rollout operation deadline sprint](SPRINT_ray_rollout_operation_deadlines.md)
  bounds startup, generation, capability, chunk-size-probe, and weight-ACK
  business waits.

A healthy probe is not generation progress. The health concurrency group can
answer while the actor's default group is busy or hung.

## Historical design correction

An early version of this sprint tried to thread one shared deadline object
through roughly fifteen call signatures. That prototype was reverted before the
health sprint shipped because it mixed process reachability with business-call
progress and spread mutable deadline plumbing across layers.

The later deadline sprint did not restore that design. It uses two scalar
configuration fields at real owners, a shared domain-neutral Ray deadline
adapter, per-submitted-ref generation deadlines, and explicit pipelined chunk
progress. This preserves the health sprint's narrow responsibility.

## Runtime boundary

The current ownership is:

- the rollout schedule owns admission and normal drain;
- `RayGenerationRuntime` owns activation, offload, terminal state, health
  monitoring, and actor teardown;
- `RayGenerationExecutor` owns request planning and dispatch;
- `RayGenerationWorker` is the Ray adapter around `GenerationWorkerCore`;
- `vrl.scripts.train` writes `run_verdict.json`;
- `vrl.scripts.supervise` applies bounded attempt and same-cause policies after
  a failed verdict; a permitted retry resumes the latest complete checkpoint.

The monitor starts only after startup load, metadata, and capability checks
complete. Operation deadlines independently protect those pre-publication
waits.

## Reachability failure contract

```text
worker misses its health probe
  -> publish RolloutWorkerUnreachable
  -> close runtime admission and retain the first failure
  -> kill every owned actor with no_restart=True
  -> blocked business ObjectRefs fail, or the next foreground command is rejected
  -> terminal cleanup retains any failed actor handles for retry
  -> the foreground failure writes a failed run verdict and exits non-zero
  -> the supervisor applies bounded restart policy
  -> a permitted retry resumes the latest complete checkpoint
```

Killing the fleet is the mechanism that releases already-blocked driver calls.
A monitor that only logged the failure would leave the attempt hung.

There is one known idle-window gap: if the health miss happens after the final
foreground operation, no caller necessarily reads the retained lifecycle
failure. A successful cleanup-only `shutdown()` does not currently re-raise
that retained failure, so this edge case is not guaranteed to write a failed
verdict or hand control back to the supervisor. Closing that gap requires an
explicit final failure-reader boundary and is not claimed as part of this
completed sprint.

Driver, GCS, raylet, `ray.kill`, and process-level hangs remain a separate
watchdog problem.

## Configuration

The public rollout schema owns three health fields:

```python
health_check_interval_s: float = 30.0
health_check_timeout_s: float = 30.0
health_check_first_wait_s: float = 0.0
```

`health_check_interval_s <= 0` disables monitoring. Enabled monitoring requires
a finite positive timeout, and the first-wait grace must be finite and
non-negative. `RolloutWorkerConfig` projects these fields without duplicating
defaults.

The separate deadline sprint adds:

```python
worker_rpc_timeout_s: float = 600.0
generation_stall_timeout_s: float = 3600.0
```

These fields do not change what a health probe means.

## Dedicated Ray concurrency group

`RayGenerationWorker.health()` touches no model or GPU state and runs in
`HEALTH_CONCURRENCY_GROUP`. The default concurrency group remains serialized,
so enabling health checks cannot accidentally allow two chunks to execute
concurrently on one GPU worker.

A plain `max_concurrency=2` remains incorrect because it would apply to every
actor method instead of only the reachability adapter.

## Parking behavior

Monitoring is paused before workers enter the parked/offloaded state and resumed
after wake completes. A parked worker is intentionally outside the active
serving contract; probing it as though it were serving would turn a residency
transition into a false death report.

`health_check_first_wait_s` applies again after resume, providing an explicit
wake-up grace without adding a second monitor state machine.

## Verification

The original sprint was completed and revalidated on 2026-07-22:

- full repository suite: 2710 passed, 17 skipped;
- broad generation/Ray/collector/orchestration selection: 531 passed;
- generation architecture and runtime protocol checks: 22 passed;
- real-Ray coverage proved that health responds while a default-group call is
  blocked and that default-group calls remain serialized;
- scoped Ruff checks and formatting passed.

The 2026-07-30 deadline sprint additionally includes a real-Ray CPU case where a
business call hangs, the health concurrency group remains responsive, and the
business deadline still fires. That test demonstrates why both mechanisms are
necessary.

## Architecture hygiene

### Keep

- `RolloutWorkerHealthMonitor`: one real thread/lifecycle boundary;
- `RolloutWorkerUnreachable`: stable reachability failure identity;
- `RayGenerationWorker.health()`: thin Ray concurrency-group adapter;
- `RayGenerationRuntime`: actor ownership and terminal state;
- `HEALTH_CONCURRENCY_GROUP`: protocol name shared by the health/progress
  decorators and actor construction;
- `_STOP_JOIN_GRACE_S`: monitor-thread cleanup boundary.

These thin functions/classes are justified by framework, thread, protocol, or
resource-ownership boundaries. Flattening them would mix responsibilities
without removing complexity.

### Do not add

- a worker fleet manager or recovery handler;
- an ALL_CAPS health/timeout taxonomy table;
- duplicated schema field-name lists;
- model- or family-specific reachability policies.

## Non-goals

- in-process actor restart or fleet rebuild;
- deterministic whole-request replay;
- actor-level retry budgets and recovery circuit breakers;
- public recovery lifecycle phases;
- driver/GCS/raylet/process watchdogs;
- GPU chaos or throughput benchmarking;
- business-operation deadlines, which are owned by the completed independent
  deadline sprint rather than this health monitor.

## References

- `vrl/generation/ray/health_monitor.py`
- `vrl/generation/ray/worker.py`
- `vrl/generation/ray/runtime.py`
- `vrl/ray/actor_group.py`
- `vrl/ray/resource_cleanup.py`
- `vrl/scripts/train.py`
- `vrl/scripts/supervise.py`
- `docs/sprints/done/SPRINT_ray_rollout_operation_deadlines.md`
