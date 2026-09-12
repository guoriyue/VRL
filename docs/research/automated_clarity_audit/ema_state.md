# EMA shadow state ownership

Reviewed the full online/ema.py module, analytic EMA tests, online trainer
construction/update/restore callers and checkpoint export swap/rollback paths.
Previous audit commit: ed353a80d. Full trainer/checkpoint module coverage remains
pending; scoped caller inspection is not a full-module review.

## Change

Prepare restored scalar state, resharded tensors and device copies before
assigning live EMA fields. Previously decay changed before checking parameter
count, and shadows/update count changed before device copying could fail. A
non-strict trainer restore catches these failures and logs that EMA was skipped,
so preserving old live state matters. The loader now commits its prepared values
only after all local preparation succeeds, without adding a rollback object or
another validator layer.

Existing missing-field defaults, decay recurrence, update intervals and restored
dtypes are preserved. This is local object-state publication, not a distributed
transaction or a guarantee that a failed collective has no external effects.

## Retain and why

- EMA shadows, temporary raw-weight snapshots and num_updates serve distinct
  lifetimes. The update count controls whether checkpoint artifact export may
  use EMA. It cannot be inferred from the trainer step because updates have an
  interval and skipped optimizer updates do not update EMA.
- copy_ema_to owns the temporary snapshot and local rollback. Independent CPU
  storage is required even for CPU model weights. copy_temp_to retains its
  receipt if a copy fails so recovery is not falsely reported as complete.
  Checkpoint code separately coordinates peer success before another collective.
- state_dict exposes ordinary state while checkpoint_state_dict joins DTensor
  gathers on every rank and retains independent full CPU tensors only on the
  writer. Combining these interfaces would erase a real memory/ownership choice.
- step's foreach path and cross-device staging implement the same recurrence
  with different transfer needs. DTensor layout-aware restoration belongs to
  this shadow owner. Keep get_current_decay as the named warmup policy shared
  with its numerical tests; no namespace class or utility file is needed.
- to remains an existing device/dtype API even though load no longer calls it
  after mutating state. It handles non-floating tensors without dtype casting.
  No public API removal is necessary for the staging fix.
- Serialized keys are checkpoint schema. This module has no ALL_CAPS business
  vocabulary or backend taxonomy to relocate.

Non-goals: changing EMA math, configuring new warmup behavior, deleting snapshot
state, replacing framework tensor operations, or claiming CPU tests validate
multi-GPU DTensor collective failure recovery.

## Restore boundary follow-up

Closed in the follow-up to 581855168. The trainer's separate shape validator
returned early in non-strict mode, then still invoked the loader. Move its
list/count/tensor/full-shape contract into EMA.load_state_dict before any
redistribution or state mutation. Remove the trainer validator and its call.
Trainer retains only strict rethrow versus non-strict logged skip, so it can no
longer report successful loading of incompatible shadows.

The live EMA shadow layout supplies the expected shapes, rather than traversing
model parameters a second time in trainer. Direct loader callers now receive
the same compatibility check. Missing ema_parameters no longer means silently
retaining old shadows while loading new scalar state: it is an invalid EMA
payload, matching the existing strict trainer checkpoint contract. Optional
decay/num_updates defaults remain unchanged. No dtype conversion policy or
additional validator class is introduced.

The regression failed before the edit for non-strict restore: no skip warning
was emitted and the incompatible state was installed. Both strict and non-strict
cases now preserve the original shadow list, values, decay and update count;
strict raises and non-strict logs the skip. 157 EMA, online resume and checkpoint
tests passed, one skipped and two GPU tests deselected. Ruff passed for the three
changed Python files. This establishes local loader state behavior; it does not
make the entire trainer resume atomic or coordinate failures across DTensor
redistribution ranks. The state_dict alias versus independent checkpoint export
snapshot distinction remains.

## Validation

Both new failure cases failed before the edit because decay had changed: count
mismatch and a CPU staging failure represented with an unstaged meta tensor.
They now preserve original shadow-list identity, values, decay and update count.
A successful save/load test also verifies the next EMA update against the
uninterrupted recurrence. 126 EMA/checkpoint tests passed, one skipped. Ruff
check and format check passed for both changed Python files. No GPU run claimed.
