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
- TrainingStateParking and its TrainingMemoryState input live in
  vrl/models/parking.py, without importing trainer strategies.
  TrainingStateParking inherits ModelParking's model/tensor restore ledger.
  Model and frozen-component moves, alias deduplication and local DTensor
  relocation are shared with generation's ordinary CPU parking backend.
  TrainingStateParking adds optimizer, EMA, scaler and live-gradient traversal.
- _TrainingParkingStrategy retains phase identity and rollback coordination;
  FSDP adds peer agreement. CuMem mappings and Accelerate hooks remain separate.
  Training allocator release still synchronizes before and after cache release;
  generic cache clearing is not equivalent.
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

Closed in the follow-up after 2bf88e724: FSDP inherited process-group shutdown
and ignored restore_parked, leaving its parking ticket live. The rollout phase
intentionally leaves the trainer parked after failed role release, and
_OnlineRecipeLifecycle passes pipeline cleanup success as restore_parked. This
is the explicit permission boundary; placement removal alone does not grant it.
Move the existing single-process shutdown body into _TrainingStateParking and
have FSDP compose that behavior with process-group shutdown in finally. False
abandons the ticket without GPU restoration; true restores it before group
cleanup. A restoration error retains the ticket and still attempts group cleanup.
Single-process behavior and DDP's process-group-only shutdown remain unchanged.
No new owner class, flag or module-level constant was introduced. The thin FSDP
override is necessary composition of two cleanup responsibilities, not a wrapper
to delete for line count. External process-group adoption remains a separate
open ownership issue above.

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

Shutdown follow-up: three new behavioral cases failed before the change. They
verify permitted restore before group cleanup, forbidden restore clearing the
ticket (including a later restore attempt), and group cleanup despite restore
failure. 39 parking/shutdown/strategy-contract/online-lifecycle tests passed;
86 unrelated or GPU cases were deselected. CPU model moves and an observed group
cleanup callback establish ordering and ticket behavior, not actual CUDA/NCCL
reclamation. Ruff check and formatting passed on strategy.py and test_fsdp.py.
