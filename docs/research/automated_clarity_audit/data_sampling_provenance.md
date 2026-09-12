# Prompt sampling and dataset provenance

Reviewed complete data/prompt_sampler.py, data/provenance.py and data/__init__.py,
online recipe sampling/preview/resume call sites, RNG capture/restore functions,
provenance builder report fields and loader/launch-boundary tests. Previous audit
commit: 532a00072. Large checkpoint and dataset artifact modules remain pending
their complete reviews.

## Changes

Correct sample's docstring: only random sampling advances the generator;
sequential windows derive from the supplied epoch. Add DatasetProvenance to the
package's TYPE_CHECKING exports, matching its existing lazy public runtime
export. No runtime selection, provenance requirements or dependency loading
behavior changes.

## Retain and why

- PromptBatchSampler owns fixed geometry and an externally supplied generator.
  sample and preview express different RNG lifetimes; _sample_with implements
  one selection algorithm for both. Preview clones generator state, making the
  next batch available without consuming the training draw. There is no second
  sampler cursor to infer or checkpoint.
- Every rank draws the same global permutation from identically initialized or
  restored prompt RNG, then slices its local share. Process RNG may be rank
  offset while prompt RNG deliberately is not. Sequential windows wrap modulo
  dataset length; random sampling requires enough examples for the global draw.
  Do not unify these different replacement/coverage semantics accidentally.
- The enum's historical str representation is retained. Its values and package
  _PUBLIC_EXPORTS are public/config boundaries. Lazy __getattr__ keeps schema
  imports torch-free; __dir__ supports discovery and TYPE_CHECKING supports
  static consumers. These thin functions serve a real import facade.
- SourceReport owns its JSON report schema; REQUIRED_KEYS are actual schema
  keys. PROVENANCE_SPECS is an explicitly isolated task taxonomy in a module
  named for this purpose. Keep it here rather than scattering task-name branches
  through launch workflow or introducing another generic contract layer.
- DatasetProvenanceSpec.load_manifest selects the configured loader. It does not
  need instance state, but belongs with the provenance loading contract; moving
  it to a new utility class would add no ownership. DatasetProvenance combines
  report and manifest evidence at launch, where filesystem IO belongs.
- The video-world metadata vocabulary is shared with artifact validation and
  builders. These are required provenance fields, not model architecture inputs
  or an algorithm whitelist. Image-to-video requires exact report row counts;
  video-world permits downstream re-splitting. Preserve those explicit policies.
- Report require_fields and manifest artifact checks validate different external
  records. Reward preflight remains responsible for media actually consumed by
  a reward; source provenance is not evidence that every reward can score it.

Non-goals: changing sample order or seeds, adding a sampler-state object,
mechanically turning methods into free functions or vice versa, removing the
lazy facade, or deleting provenance fields because there are several of them.

## Follow-up findings

- RNG exception suppression is fixed in the follow-up recorded in
  rng_checkpoint.md. Only absent optional NumPy may be omitted at capture;
  provider state errors and missing providers required by saved state propagate.
- SourceReport converts counts with int(payload.get(...) or 0), despite already
  requiring both keys. Fractional values can be truncated and strings coerced.
  Builders emit len(...) integers. Consolidate the report's count boundary with
  the repository's shared integer contract in a focused compatibility change;
  the current report check must not be described as strict integer validation.
- Task types without a provenance spec only receive path existence checks.
  Report content and file-vs-directory validation are not established there.
  Preserve this explicit limited policy rather than claim all tasks receive the
  video provenance checks.
- Production prompt generators are CPU generators. The sampler preview mirrors
  generator.device, but randperm does not specify a device. Arbitrary CUDA
  generators are not established as supported by this review or the CPU tests.

## Validation

Existing tests cover identical distributed global draws, non-consuming preview,
generator checkpoint roundtrip, configured loader equivalence, report count
mismatch and missing provenance metadata. The isolated package enum import also
confirmed torch was absent from sys.modules. Ruff passed for both changed Python
files. All 37 sampler, provenance and validation-tier tests passed. No new test
was added for annotation/docstring changes. No GPU sampling claim is made.
