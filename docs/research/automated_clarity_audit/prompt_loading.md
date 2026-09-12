# Prompt loading ownership

Reviewed complete trainers/data/prompts.py, DataConfig seed/loader declarations,
online/reward-preflight/target-encoding call sites, dataset facade exports and
existing prompt parsing/mixture tests. Previous audit commit: 88ad86fbc.

## Changes

Pass the already validated DataConfig.mix_seed directly to load_prompt_mixture.
The field is StrictInt and the loader checks for None before using it; another
int conversion adds no meaning. Correct JsonlPromptDataset's description: unknown
row fields are retained as metadata rather than being forbidden.

## Retain and why

- PromptExample owns the actual generation-conditioning versus reward-metadata
  projection. Targets stay out of GenerationInput, while reward metadata carries
  clean target identities and references. Do not merge these semantically distinct
  consumers into a generic field-copy helper.
- load_prompt_manifest dispatches file formats; the bytes parser lets callers
  consume an authenticated snapshot without reopening a path. These are public
  I/O and serialization boundaries, not constructors missing a receiver.
- load_prompt_mixture owns a local seeded RNG. Each rank must independently build
  the same indexable list; seed and draw order are meaningful training behavior.
  A new loader class would not simplify this state confined to one invocation.
- ImageCaptionPromptDataset adapts external image/caption field names, while
  JsonlPromptDataset adapts the native manifest. Their small Dataset methods follow
  the framework API. A shared base solely for len/getitem would add navigation
  without consolidating parsing or eliminating meaningful complexity.
- load_prompt_image_manifest is an exported list-returning facade over the image
  dataset. Preserve the established return type and public import rather than
  deleting it solely because production config loading uses the dataset directly.
- JSON object/string/mapping validation is at an external manifest boundary.
  Keep existing diagnostics; add no new hypothetical value checks or test cases.
  Known-field membership derives from the dataclass rather than another hardcoded
  vocabulary. There is no module-level ALL_CAPS business table to relocate.

Non-goals: change mixture sampling, defaults, metadata precedence, JSONL error
numbering, target/reference meaning or supported public file formats. Image-loader
preprocessing fallbacks correspond to optional schema fields and stay intact.

Validation: 49 prompt/config tests and 7 runtime loader tests passed; Ruff check
and format check passed for prompts.py. No new tests,
examples, checker classes or gates were added. These are CPU file/config tests,
not proof of distributed training or remote dataset availability.

Scoped artifact-module reads in this turn do not mark artifacts.py reviewed; its
remaining path-resolution functions and call sites still need a complete pass.
