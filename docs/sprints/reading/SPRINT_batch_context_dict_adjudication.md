# Batch context wire-shape decision: keep the plain dictionary

**Date:** 2026-07-18  **Status:** DECIDED — revalidated 2026-08-09 after
orchestration provenance was removed from `batch.context`

The args/settings audit considered replacing each denoise family's
`export_batch_context` / `restore_eval_state` string-keyed payload with a
family-specific dataclass. The decision is to keep `batch_context` as a plain
dictionary because it is an extensible wire payload, not one closed settings
object.

## Decision evidence

1. Generic orchestration deliberately does not write scheduling provenance into
   `batch.context`: `RolloutIteration` carries only trainer-consumed batches and
   typed stats. The remaining dictionary therefore belongs to model-family
   sampling/replay wire data, not to a shared scheduling metadata schema.
2. `vrl/models/steps/denoise/common/tensors.py::replay_tensor` accepts a tensor
   name at runtime and falls back to `batch_context[name]`. The key set is
   intentionally not closed at the shared layer.
3. Family producers and consumers remain close together: each model exports
   the replay context and reconstructs its sampling state in the same module.
   Existing family tests cover several export/restore paths, but not every
   family performs a literal round trip. That partial coverage is a regression
   aid, not the reason to claim a statically closed schema.

The correct boundary is therefore:

```text
family sampling state
    -> dict export
    -> trajectory
    -> family restore
```

## Separate live extension seam

`chunk_passthrough_keys` is not a dormant knob. The FLUX model preset declares
`[text_ids]`, and the full-sequence denoise executor consumes the list when projecting
chunk inputs. Unknown passthrough keys fail loudly. This is independent of the
batch-context wire-shape decision and must remain supported.

## Revisit trigger

Revisit this decision if replay lookup no longer accepts runtime-selected names
and every family payload has a closed key set. A future closed protocol could
then justify a typed payload; the current one cannot.

## Non-goals

- Do not introduce a shared-key base class or `TypedDict`; family-specific,
  runtime-selected keys make either declaration incomplete.
- Do not use this decision to block typed internal sampling-state dataclasses.
  Those are family-owned objects on one side of the wire boundary.
- Do not fold unrelated `sampling`, executor-construction, or request payload
  decisions into this record.
