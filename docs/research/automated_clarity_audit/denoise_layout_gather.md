# Full-sequence layout and gather boundary

Reviewed complete `layout.py`, `gather.py` and `__init__.py` under
`vrl/generation/bindings/full_sequence_denoise`. Inspected executor construction
of layouts, request-owned seed initialization, the shared covering-batch helper,
and layout/storage-adoption tests. The executor module remains pending full review.

## Change

Correct the gather module's import-boundary documentation. Although its own
DiffusionBatchResult import is TYPE_CHECKING-only, the parent package eagerly
exports executor classes. A fresh interpreter importing the gather submodule
confirmed the executor module is loaded. The useful existing boundary is that
the serializable gatherer needs no model instance, not that importing it avoids
all executor code. No runtime or serialization behavior changed.

## Retain

- DiffusionRequestLayout has actual executor-owned defaults; its constructor
  deliberately provides no second defaults. `text_encode_kwargs` is shared by
  encoder/preparation consumers and preserves an absent maximum text length.
  Keep it on the sampling-params owner rather than reproducing two projections.
- Frame-name compatibility and default SDE type are explicit supported request
  choices. Request geometry and max text length are validated at the conversion
  boundary; do not add another generic checker object.
- SDE window selection uses sampling.seed or the request's stored random seed.
  Request construction draws the latter once; serialized copies and OOM split
  retries preserve it. Re-parsing does not resample a new window. Keep the fixed
  XOR salt as part of seeded-stream compatibility, not a tunable algorithm fact.
- DiffusionBatchGatherer is a real GenerationBatchGatherer protocol adapter.
  Keep its dedicated serializable shape across families. Ordered coverage,
  sample concatenation, replay gathering and context gathering already share
  implementations. The explicit row-field tuple is a payload schema; replacing
  readable tensor names with reflective dynamic packing would not remove a
  duplicated algorithm.
- Keep nonempty context enforcement and explicit trajectory construction. The
  storage-adoption test exercises actual gather-to-collector flow and verifies
  policy casting preserves the resulting trajectory's identity.
- Package exports are a public facade, and __all__ is its API vocabulary. Do not
  remove exports merely to make the inaccurate lightweight-import claim true.
  If driver import weight becomes a requirement, review package lazy exports
  consistently across bindings and test cold imports/serialization together.

No training math, seed stream, accepted input, gathering order or package API
changed. No new class, compatibility fallback, numeric guard or test framework.

## Validation

113 existing full-sequence binding tests passed on CPU, including seeded and
unseeded windows, serialized OOM retry, parameter projection, encoded sample
expansion, serial/pipelined output equivalence and trajectory storage adoption.
This is not a multi-node launch or GPU performance measurement. Ruff passed on
the documented module. Previous audit commit: `2343674a9`.
