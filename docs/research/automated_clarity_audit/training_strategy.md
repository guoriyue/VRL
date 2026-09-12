# Training strategy and parking ownership

Reviewed all of strategy.py, the distributed process-group helpers, and relevant
single-process/DDP/FSDP parking, state export and MRO regression coverage.
Previous audit commit: 0aaf838d9. The FSDP tensor implementation in fsdp.py and
complete trainer orchestration are separate pending module reviews.

## Change

Remove _ParkedTrainingState.key. It duplicated identity derived from the retained
frozen TrainingMemoryState. Compare parked.state.identity_key directly with the
incoming state's identity instead. The record keeps strong references to the
owners, and the identity uses their object IDs and immutable torch device, not
mutable tensor contents. Existing idempotence and owner-change checks remain.
No new state manager, validation helper or identity test framework is added.

## Retain and why

- Strategy is a structural consumer protocol outside concrete-class MRO. Its
  apparent stubs must not become inherited implementations shadowing mixins.
  SingleProcess, DDP and FSDP retain uniform trainer-facing method signatures.
- _TrainingStateParking coordinates all live owners: model/ref, independent
  optimizer master parameters, moments, EMA, scaler and live gradients. The
  shared identity sets prevent moving aliased objects twice; module and tensor
  restore receipts describe different restoration operations. Keep those types.
- _module_tensors and _module_device serve discovery and restore location;
  _move_module includes a family frozen-component hook. Tensor-tree traversal
  preserves aliases, while _move_tensor_data targets a DTensor's local shard.
  These form shared mechanisms used across owner types, not independent public
  operations to move into unrelated classes. The release helper synchronizes
  before and after cache release; generic cache clearing is not equivalent.
- _ProcessGroupStrategy shares coordination between DDP/FSDP. Unsharded state
  methods serve single-process and DDP without forcing replicated tensors into
  FSDP gather machinery. Keep the thin checkpoint/optimizer adapters for that
  backend consistency and lazy dependency boundary.
- _trainable_module_handles requires a model-owned mapping and alias-updating
  writer, preventing family names in strategy dispatch. Its bound writers are
  actual wrapping adapters, not needless lambdas to replace with a new class.
- FSDP dtype preparation checks mixed/absent parameter types only when no target
  dtype is declared, before collective setup. Do not remove them merely because
  normal homogeneous models pass. Precision normalization belongs to the FSDP
  boundary; downstream repeated checks require separate evidence to justify.
- FSDP CPU-offloaded norm clipping moves only a scalar for reduction. DDP clips
  replicated gradients locally. These are different algorithms and remain
  separate; this audit makes no change to optimizer or gradient mathematics.
- build_strategy is the single concrete-backend construction boundary. The
  state-load callback selects two real scatter APIs with one module-root check.
  Keep these functions rather than inventing a namespace constructor class.
  __all__ is the only module ALL_CAPS declaration here and remains the facade.

Non-goals: collapsing strategy adapters for LOC, altering sharded checkpoint
formats, changing dtype policy or removing cancellation/rollback contracts.

## Findings requiring coherent follow-up

The process-group adoption finding in rank_launch_context.md is confirmed by
strategy consumers: CPU barriers silently skip without a CPU group, and success
reduction can fall back to NCCL. A new owner must cover creation/adoption,
subgroup creation failure and teardown, not merely replace the global with a
class variable. Preserve external-group test lifetimes when implementing it.

FSDP inherits _ProcessGroupStrategy.shutdown, which ignores restore_parked and
does not consume _parked_training_state. SingleProcessStrategy does restore or
explicitly abandon its ticket. Review trainer terminal cleanup and whether GPU
ownership is safe before unifying these paths; never restore automatically on
an unsafe shared-device shutdown.

DDP's class description refers to symmetric colocated online execution, but
validate_training_state_parking explicitly rejects shared-GPU parking. Verify
the complete online admission path before claiming this topology is supported.

Module parking records a single restore destination, not arbitrary per-component
device placement. DTensor shard moves use a private Torch field. Existing CUDA
round-trip tests are the stronger evidence for those mechanisms and were not
run in this CPU audit; avoid promising general placement preservation.

## Validation

44 strategy/MRO/DDP tests passed; three GPU tests were deselected. Two existing
FSDP parking preflight/peer-failure rollback tests also passed. The CPU lane
checks identity deduplication and live Adam/GradScaler state but does not prove
CUDA memory reclamation or multi-rank NCCL parking. Ruff check and format check
passed for strategy.py. No new tests mirror the removed cached field.
