# Chunk-autoregressive generation binding

Following c2c9c08ef, reviewed bindings/__init__.py and the three modules in
bindings/chunk_autoregressive_denoise. Followed CausVid's trainable result and
MAGI-1's generation-only result, shared sample ordering/replay gather utilities,
and existing binding/replay tests.

Change: use temporal chunk terminology in the executor documentation, keeping
transport batches distinct from temporal generation organization. No runtime
implementation change was justified by this review.

Retain and why:

- Executor and gatherer implement worker-side model execution versus driver-side
  assembly. The latter owns no model. Their separate files are actual protocol
  boundaries used across CausVid and MAGI-1, not arbitrary function grouping.
- The two old _cat_* wrappers are already absent. The remaining
  _order_and_validate_batches composes shared request coverage validation with
  chunk-specific axis/count/homogeneity checks. Keep that recognizable boundary.
- Result construction establishes complete trainable field presence; gather
  verifies sample/chunk/transition shape prefixes before concatenating. Result
  dataclasses are mutable transport payloads. Shape checks are not implied by
  their annotations or presence checks, and the validator has one production
  invocation site here. No additional validation passes are needed.
- Context consistency and declared replay axes matter: prompt embedding length
  and width can equal temporal/transition counts without representing those axes.
  The existing embedding regression proves declarations survive gather without
  guessing from equal dimensions. Cross-result axis agreement must remain.
- CausVid supplies recorded observations/actions/log-probs and finalized chunks;
  MAGI-1 supplies generated output only. Keep separate trajectory builders so the
  generation-only branch does not fabricate trainable policy facts.
- Optional KL is either absent everywhere or present everywhere. Partial KL
  cannot be silently concatenated with fewer sample rows. Shared replay-tensor
  gathering also distinguishes sample-aligned tensors from static values.
- Thin family executor subclasses retain the same cross-family protocol shape.
  Package exports expose that shared API. __all__ is the only ALL_CAPS declaration
  in this binding; field-name tuples express the replay tensor schema.

Non-goals: create a generic result validator class, merge driver and worker
ownership, delete shape checks based on typed annotations, change trajectory
axes or unify inference-only and training outputs. No new tests or gates.

Validation: 41 binding/CausVid tests initially passed; the packaged-source digest
test failed before exercising runtime code because this isolated worktree lacked
third_party/CausVid/causvid. The branch gitlink pins
adb6a5ecd07666b4d0290042915c8406e6d5ce22, also present in the manual repository's
submodule object store. A local shared clone was created only inside the audit
worktree and detached at that exact commit. The failed test then passed (42
existing cases validated across the two runs). The clone reads shared Git objects;
it does not use or modify the manual submodule's working files. Parent Git status
shows no submodule change. This dependency is local setup, not a committed source
change. Ruff check/format check passed. CPU coverage does not run either full GPU
model or the MAGI subprocess. Coverage: 206 reviewed, 293 pending modules.
