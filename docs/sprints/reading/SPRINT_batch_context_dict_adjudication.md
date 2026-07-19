# Batch context wire-shape decision: keep the plain dictionary

**Date:** 2026-07-18  **Status:** DECIDED — source study and boundary record;
no implementation work

The args/settings audit considered replacing each denoise family's
`export_batch_context` / `restore_eval_state` string-keyed payload with a
family-specific dataclass. The decision is to keep `batch_context` as a plain
dictionary because it is an extensible wire payload, not one closed settings
object.

## Decision evidence

1. The generic orchestration layer merges runtime metadata into the same
   payload. `vrl/rollouts/orchestration/types.py::annotate_batch_context`
   combines `iteration.metadata` with `batch.context`; both strict and
   continuous schedules consume that behavior. A family dataclass would need a
   serialization wrapper at this boundary or would prevent generic metadata
   from being attached.
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
    -> trajectory plus orchestration metadata
    -> family restore
```

## Separate live extension seam

`chunk_passthrough_keys` is not a dormant knob. The FLUX model preset declares
`[text_ids]`, and the full-sequence denoise executor consumes the list when projecting
chunk inputs. Unknown passthrough keys fail loudly. This is independent of the
batch-context wire-shape decision and must remain supported.

## Revisit trigger

Revisit this decision only if generic metadata stops sharing `batch.context`
and replay lookup no longer accepts runtime-selected names. A future closed
protocol could then justify a typed payload; the current one cannot.

## Non-goals

- Do not introduce a shared-key base class or `TypedDict`; family-specific keys
  and generic runtime metadata make either declaration incomplete.
- Do not use this decision to block typed internal sampling-state dataclasses.
  Those are family-owned objects on one side of the wire boundary.
- Do not fold unrelated `sampling`, executor-construction, or request payload
  decisions into this record.
