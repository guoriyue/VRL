# SPRINT: Explicit rollout activation and schedule-owned draining

Status: **DONE (2026-07-18)**. The contract and its regression coverage are
implemented. Process-reachability monitoring and supervisor handoff after an
unreachable probe were subsequently completed by the
[worker process-health sprint](SPRINT_rollout_worker_liveness.md). That monitor
does not bound business RPCs; the later
[Ray rollout operation deadline sprint](SPRINT_ray_rollout_operation_deadlines.md)
completed that independent gate.

## Decision

The rollout schedule is the single owner of admission and drain barriers. The
generation runtime owns Ray resources, explicit activation/offload, transactional
weight installation, and terminal teardown.

This follows the control-plane shape used by SGLang/Miles without adopting their
LLM-specific engine stack:

```text
pause admission -> drain -> update weights -> resume admission
```

Visual rollout remains native to this repository: diffusion requests finish their
full denoising trajectory instead of using token-level retract/resume.

## Runtime contract

```text
activate()       launch or wake workers, then complete a pending policy install
generate()       require an explicitly active runtime
update_weights() install immediately when active, otherwise stage an install
offload()        park already-idle workers; never drain generation itself
shutdown()       join activation/offload tasks and tear down owned resources
```

`release()` and `with_release_after_collect()` compatibility facades have been
removed. The canonical lifecycle contract uses `activate()` and `offload()`;
`RayGenerationLauncher.create_runtime()` selects the resident or on-demand
implementation from the resolved topology.

## Ownership

- `StrictOnPolicyRolloutSchedule` activates before collection. Awaiting collection
  is its drain barrier; it offloads in `finally`.
- `ContinuousRolloutSchedule` pauses producer admission and drains in-flight work
  before weight sync unless workers advertise versioned trainable-state slots.
- Continuous scheduling rejects a reward topology that requires mid-iteration
  rollout offload. That topology needs strict phase ownership or a dedicated
  reward GPU.
- Online recipe shutdown stops/joins the schedule before collector/runtime
  teardown.

## State

The terminal admission state has only:

```text
RUNNING -> SHUTTING_DOWN -> TERMINATED
```

There is no runtime-level `QUIESCING`, `OperationTicket`, active-operation map, or
waiter map. Activate and offload each use a single-flight task; offload explicitly
waits for an in-flight activation, while shutdown uses one shared cleanup task.
Worker policy install cannot overlap an in-flight activation. The implementation
was subsequently consolidated: on-demand launch attaches an inner
`RayGenerationRuntime`, and that inner runtime directly owns its executor, weight
sync, actors, monitor, parking, and teardown. There is no
`RayGenerationWorkerFleet` or second public lifecycle owner; the outer facade
retains only an unacknowledged policy install and exposes one collector-facing
terminal boundary.

On-demand policy state distinguishes:

- `current_policy_version`: latest target accepted by the facade, including a staged
  target that inactive workers have not acknowledged yet;
- `_pending_policy`: full trainer snapshot retained only until a worker fleet
  acknowledges that target;
- `_active_policy_version`: version acknowledged by the active worker fleet.

An inactive update advances the accepted target and stages its payload without
claiming that workers are active on that version. An active update advances both
version facts only after every required worker acknowledgement. Successful active,
cold, and wake installs release the full CPU payload immediately; parking keeps the
installed model, so only version scalars need to survive. A partial update failure
does not publish a new active version, closes admission, and destroys the unknown
worker state through terminal cleanup.

## Non-goals

- No SGLang or Megatron dependency.
- No token-level pause/retract for diffusion.
- No change to family executors, gatherers, trajectory formats, or replay.
- No removal of retryable partial resource cleanup.
- No flattening of protocol adapters merely to reduce line count.

## Verification

Tests cover explicit activation, activation/offload single-flight, staged policy
restore, worker acknowledgements, update failure cleanup, shutdown joining
runtime-owned control tasks, schedule-owned drain ordering, and rejection of an
unsupported continuous mid-iteration handoff.
