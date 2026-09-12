# Full-sequence executor and reference conditioning

Reviewed the complete full-sequence `executor.py`, Cosmos Predict2 `runtime.py`
and Wan 2.1 `runtime.py`. Inspected both models' encode outputs and registry
executor declarations, plus reference-conditioning, layout, storage and probe
tests. Model implementations and generic execution infrastructure remain pending
full review.

## Change: reuse the encoded reference image

ReferenceConditionedBatches loaded an image for encoding and then loaded it
again in build_prepare_kwargs. Cosmos build_batch_encoded also eagerly loaded
the fallback before calling dict.get, even when encoded already contained the
reference image. The inspected Cosmos and Wan encoders return that same image
in encoded["reference_image"].

Preparation now reuses a non-None encoded reference; absent/None still loads the
request image. Cosmos batch expansion only loads a fallback when the key is
absent, preserving its existing treatment of an explicitly present None. No
executor-level cache or lifetime manager is introduced: the existing per-batch
payload carries the image. This removes redundant file I/O and keeps encoding
and preparation on the same loaded image if the source file changes between
stages. A custom encoder's non-None reference is now honored during preparation
instead of being replaced with a new load from the request.

## Retained shapes and boundaries

- The shared reference-conditioning mixin serves two real family executors.
  `_reference_image_for_batch` selects the actual prompt input, supports both
  in-memory media and path-backed images, and owns the PIL file lifetime. Keep
  this shared operation rather than copying it into family adapters.
- Wan's small subclass supplies defaults and inherits real I2V conditioning;
  it is not equivalent to the generic T2V executor. Cosmos additionally expands
  prompt tensors while preserving reference conditioning. Keep these consistent
  family adapter shapes; no class elimination based on line count.
- The generic executor already covers families needing only configured defaults.
  `batch_passthrough_keys` declares shared encoded tensors whose leading axis is
  not a sample axis. This is a family payload schema, not a vocabulary to move
  into the workflow. There are no ALL_CAPS business tables here; __all__ is API.
- Encode, preparation, loop and decode hooks are actual family/test extension
  points. Their uniform keyword signatures and protocol adapters are useful.
  Normal forward and memory probe share the canonical private flow but select
  different execution/storage options; do not duplicate either stage sequence.
- Keep the preparation batch-width check: family hooks can replace the loop,
  while standalone buffer allocation also has its own entry boundary. Identical
  wording of a check does not establish identical call coverage.
- Keep storage policy before the wire and at collector construction. Only the
  worker-side application reduces transfer bytes. Keep decoded uint8 packing
  separate from training tensor storage and preserve existing optional proposal
  mean/reference-prediction/window exports.
- Stage timings, memory peaks and engine counter names are runtime observation
  contracts. No changes to their values, reset scopes or byte accounting.
- Pipelined execution uses the shared pipeline and order-preserving gatherer;
  no new threading or overlap scheme. Existing parity tests provide bounded CPU
  evidence, not a universal bit-exactness or GPU speed claim.

## Validation

152 existing/new full-sequence binding and preallocation tests passed, with 3
CUDA cases skipped. The new regression uses real PIL files and the actual
Cosmos/Wan executor hooks with an encoder fake matching the inspected payload:
one file open, shared image identity through expansion and preparation. Existing
tests cover absent-image fallback during Cosmos expansion and missing request
reference rejection. Ruff check/format and diff whitespace checks passed.

Previous isolated audit commit: `030ab8436`. No training math, sampling defaults,
model weights or public method signatures changed.
