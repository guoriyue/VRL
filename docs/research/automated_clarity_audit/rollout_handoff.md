# Rollout phase and schedule ownership

Reviewed complete rollout_runtime.py, schedule.py and strict_on_policy.py;
continuous schedule's trainer-thread snapshot entrypoints; existing strict
failure, orchestration, driver-offload and topology tests. Previous audit
commit: 3f22c83fe.

## Changes

Consolidate identical cleanup exception capture in rollout_phase. Offload and
conditional trainer restore now execute sequentially in one try block. If
offload raises, restore is skipped; if restore raises, that same cleanup error
is retained. Body and cleanup failures still appear together through
RolloutPhaseCleanupError. No changes to admission, timing phases or cancellation
classification are intended.

Correct the module description: RolloutCollectorControl describes control
capabilities, not the complete collector API. Strict and continuous schedules
also directly call prompt generation/collection/scoring methods. Do not add
isinstance checks or pretend its existing partial protocol covers those calls.

## Retain and why

- The coordinator owns the handoff lifetime; schedules declare when a phase
  occurs. The async context manager is a real exception-safe lifetime boundary,
  not a forwarding function needing another class.
- Preparation and async weight push are separate because FSDP export may run
  collectives on the trainer thread. Continuous scheduling prepares before
  crossing into its owner loop. The initialized flag publishes only after a
  successful push. Snapshot detachment/CPU checks do not establish storage
  independence: copying remains the state getter's responsibility.
- current_policy_version reads the runtime or syncer's published version. This
  supports an unattached runtime, but never invents a version from push count.
  Invalid provider values and property failures remain visible. Do not add an
  inferred next-version counter or remove the provider boundary validation.
- Short parking, activation and offload methods attach consistent timing and
  are shared by strict and continuous schedules. shutdown_collector_runtime
  parks a shared trainer before a sleeping reward pool may wake during teardown.
  The top-level strategy restores only after cleanup succeeds. Keep these owned
  operations rather than scatter their bodies into schedules.
- RolloutSchedule is the trainer protocol. Strict's unused next_prompts argument
  and no-op reset preserve that protocol across scheduling modes. Its shutdown
  delegates lifetime ordering to the shared coordinator; that is useful uniform
  composition, not dead indirection.
- build_rollout_schedule owns one explicit mode dispatch. The algorithm supplies
  its off-policy capability as a fact, not a hardcoded algorithm-name list.
  validate_rollout_schedule_topology compares schedule semantics with resource
  ownership, so a free cross-type guard is appropriate. Runtime checks for direct
  schedule users are a separate entry boundary from config preflight.
- __all__ declarations are public facades. No module-level ALL_CAPS business
  vocabulary appears in these three modules. Keep protocol names and fixed
  timing keys; they are not data tables that need relocation.

Non-goals: parallelizing strict generation, changing accepted policy lag,
removing interfaces merely because implementations are short, or making a new
collector wrapper class. No change to training or reward mathematics.

## Validation and limits

84 orchestration, strict-failure, driver-frozen-offload and continuous-schedule
tests passed. The strict cases cover body failure, failed offload suppressing
restore, restore failure preserving the body error, and synchronization after
training rather than before collection. Snapshot tests cover the prepared
copy and failed push not publishing initialized state. Ruff check and format
check passed for rollout_runtime.py. No actual GPU handoff performance claim.

The control protocol remains a partial, currently unannotated description;
its runtime_checkable decorator is retained as an exported API capability, not
used to add a per-call structural check. A future collector API typing pass must
account for the collection methods as well. Repeated cancellation during async
cleanup depends on owner-level teardown; this pass does not claim cleanup is
uncancellable or that tensor containers are deeply immutable.
