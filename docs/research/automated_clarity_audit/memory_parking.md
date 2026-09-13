# Worker memory parking lifecycle

Reviewed the complete 586-line `vrl/generation/execution/memory_parking.py`,
worker load/release/sleep/wake callers and targeted failure/retry tests. Retain
the implementation in this batch; the small methods represent shared lifecycle
operations rather than independent utility clutter.

## Retain and why

- `_ParkingPlan` versus `_ParkingSession` distinguishes configuration before
  construction from committed backend ownership. `ModelParking` carries a
  shared model/tensor restore ledger; `_CumemParking` carries the allocation pool. A single bag of
  optional fields would permit combinations the current union excludes.
- `build` must wrap executor construction because CuMem allocation ownership
  starts during model construction. The callback lets this owner establish the
  pool before model tensors exist without importing each family's constructor.
  It is not a constructor function returned to the caller.
- `ModelParking` groups policy/frozen-component moves and restoration, shared
  with its training-state subclass. The worker decides whether to quarantine. A failed rollback quarantines the worker; a successful rollback
  permits a later sleep retry. Wake keeps the restore target until both moves
  finish. Terminal release drops ledger references before allocator cleanup.
  These conditions cannot be replaced by an unconditional parked flag.
- `require_active` adds a parked check to `require_healthy`. Both are shared by
  worker operations, and neither a parked model nor an unhealthy offload hook
  may execute. `_reset_pipeline_cpu_offload` centralizes recovery and diagnostic
  chaining for ordinary sleep and failed generation.
- `_quarantine` retains the first lightweight reason and never downgrades a
  broken CuMem mapping to an ordinary quarantine. CuMem partial unmap/remap
  failures forbid an in-process close; ordinary module move failures can be
  recoverable. This distinction is exercised by existing tests.
- `release_scope` wakes an intact parked pool before the worker drops the
  executor, then collects memory and closes the pool. It is a real lifetime
  boundary around reference removal. Its current body is the direct assignment
  `self.executor = None`; do not generalize its exception behavior into a
  promise that arbitrary failed teardown callbacks are always cleaned up.
- `_is_parked` is a shared state predicate, and `_loaded_session` is the single
  backend access boundary. Its non-required lazy session path also supports
  externally supplied executors in tests; a required parking plan must never
  silently manufacture a replacement backend.
- Enum members are a deliberately isolated state vocabulary. The imported
  residual-memory limit is shared physical handoff policy, and `__all__` is the
  module API. There are no mixed-in ALL_CAPS algorithm/model business tables to
  relocate here.

## Non-goals and open limits

Do not merge this state machine into the already large worker, replace its
callbacks with per-family constructors, remove physical GPU evidence because
module moves returned successfully, or change retry/termination semantics to
reduce lines. CPU tests establish transitions and ordering, not physical memory
reclamation or allocator correctness. The allocator initialization diagnostic
issue recorded in `utils.md` remains open and is not resolved by this review.

## Validation

`tests/generation/execution/test_worker_sleep.py`: 48 passed, 2 skipped with
CUDA disabled. Existing cases cover idempotence, move/rollback failures, failed
physical proof, CuMem terminal failures, hook recovery, release/reload and rank
ownership. No new defensive tests or production wrappers were introduced.
Previous isolated audit commit: `efaabbd6d`.
