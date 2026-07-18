# SPRINT: Explicit rollout activation and schedule-owned draining

Status: **DONE (2026-07-18)**. The contract and its regression coverage are
implemented; later worker-fleet recovery work remains in the Ray fault-tolerance
sprint.

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
activate()       launch or wake workers, then install desired_policy
generate()       require an explicitly active runtime
update_weights() install immediately when active, otherwise stage desired_policy
offload()        park already-idle workers; never drain generation itself
shutdown()       join activation/offload tasks and tear down owned resources
```

`release()` and `with_release_after_collect()` compatibility facades have been
removed. The canonical contract uses `offload()` and
`with_on_demand_activation()` only.

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

The terminal lifecycle has only:

```text
RUNNING -> SHUTTING_DOWN -> TERMINATED
```

There is no runtime-level `QUIESCING`, `OperationTicket`, active-operation map, or
waiter map. `activation_task`, `offload_task`, and `shutdown_task` are retained
because they each provide real single-flight ownership for one concrete operation.

On-demand policy state distinguishes:

- `desired_policy`: latest complete trainer snapshot accepted by the facade;
- `active_policy_version`: version acknowledged by the active worker fleet.

The desired version advances only after required worker acknowledgements when the
fleet is active. A partial update failure preserves the previous desired snapshot,
closes admission, and quarantines the fleet through terminal cleanup.

## Non-goals

- No SGLang or Megatron dependency.
- No token-level pause/retract for diffusion.
- No change to family executors, gatherers, trajectory formats, or replay.
- No removal of retryable partial resource cleanup.
- No flattening of protocol adapters merely to reduce line count.

## Verification

Tests cover explicit activation, activation/offload single-flight, staged policy
restore, worker acknowledgements, update failure quarantine, shutdown joining
runtime-owned control tasks, schedule-owned drain ordering, and rejection of an
unsupported continuous mid-iteration handoff.
