# Trainer weight export and snapshot boundary

Reviewed full trainers/weight_sync.py, both getter-factory production callers,
strategy export and coordinator snapshot consumers, bundle root declaration,
receiver namespace contract, and weight-sync/strategy/probe tests. Previous
audit commit: 140b0fe02. The performance probe was reviewed at its affected call
site; its full module remains pending.

## Change

Retire build_trainable_state_sync_getter and its nested callback. Both production
callers created the function and immediately called it; neither retained it as a
live getter. Strategy and performance probe now compose existing module
selection, flattening and CPU snapshot operations directly. Remove the test and
fixture that existed only to exercise this retired factory; existing export
tests cover actual parameter contents, wrapper names and storage independence.

TrainableStateGetter remains the trainer's callback type: recipe wiring binds a
strategy export that can perform FSDP gathers at the correct time. This real
interface is distinct from the unused factory abstraction. External direct
imports of the retired helper must migrate to the direct composition.

Correct if_supported's docstring: this optional-wiring factory requires the
explicit supports_weight_sync capability, while direct construction only
requires an update method. Do not claim a stronger constructor invariant than
the implementation enforces. Runtime behavior is unchanged.

## Retain and why

- WeightSyncer is the trainer's transport interface; RayRuntimeWeightSyncer owns
  allocation of monotonically increasing versions and serialized async pushes.
  The runtime publishes accepted versions. Advancing the allocator only after
  a successful await preserves failure behavior; the lock prevents concurrent
  pushes from reusing a version.
- require_trainable_modules is shared by checkpointing, strategy and sync. It
  expresses the trainer's nonempty root requirement for structurally supplied
  bundles. flatten_trainable_module_state owns flat wire naming; checkpoints
  instead use nested root mappings. These are real shared contracts, not a
  reason to introduce a stateless utility class.
- select_trainable_state serves both ordinary state_dict and gathered FSDP
  state. It rejects absent trainable keys and excludes frozen parameters. The
  unwrapped namespace must agree with named_parameters; compile/DDP stripping
  is shared with the receiver. Do not merge this selection into one backend.
- to_cpu_snapshot detaches and copies tensor leaves even when already on CPU.
  A plain cpu() call would alias live weights. map_tensor_tree is the existing
  shared traversal mechanism, so no new tensor conversion abstraction is needed.
  This promise concerns tensor storage, not deep copying arbitrary metadata.
- Version fields and profiling range names are protocol/observability schema.
  No ALL_CAPS algorithm or backend table is present in this module.

Non-goals: changing wire contents, model checkpoint ownership, live callback
timing, accepted direct sync adapters, transport version semantics, or flattening
framework/protocol boundaries simply to reduce function count.

## Follow-up and limits

Strategy snapshot export and syncer.push both copy tensor leaves. The coordinator
requires a prepared detached CPU snapshot but its generic syncer interface does
not encode exclusive immutable ownership. Removing the push copy globally would
break direct callers supplying live CPU state. Any copy elimination needs an
explicit prepared-payload ownership contract through the continuous owner;
do not infer immutability from device/detachment alone.

Flattening combines root-qualified names with dict.update. Arbitrary overlapping
root namespaces could collide; the bundle declaration does not validate such
names here. Real family root declarations and receiver routing need a coordinated
review before restricting this structural API or adding redundant per-key
guards. Ordinary model state_dict naming and existing family namespaces remain
unchanged by the factory deletion.

## Validation

41 weight-sync, strategy and DDP tests passed, three GPU tests deselected. The
strategy snapshot test mutates live CPU weights after export and verifies the
exported tensor's independent storage and unchanged contents. Existing sync tests
cover concurrent version allocation, frozen-parameter filtering and real compile
wrapper stripping. All 10 probe CLI tests passed, exercising the other migrated
caller with tiny CPU parameters and real Ray actors. Ruff check
and format check passed for the five changed Python files.
