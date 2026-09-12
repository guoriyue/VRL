# Artifact manifest ownership

Reviewed complete trainers/data/artifacts.py, provenance report construction,
online reference resolution, offline target encoding, Video2World derivation
schema use and existing artifact/reference tests. Previous audit commit: fc54f31ba.

Changes: read PromptExample.metadata directly instead of getattr/default/dict
copying a field used only for lookup. Both manifest loaders construct this field
as a mapping, and the same method already accesses example.metadata directly for
episode collection. Remove the local required_artifact_fields alias that merely
renamed artifact_fields in the Video2World report constructor.

Retain:

- ArtifactManifestReport owns report construction and serialization. from_manifest
  reads files; from_examples accepts already-loaded native/image-caption rows.
  The provenance consumer uses the latter, avoiding a second incompatible loader.
- SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS is an ordered manifest provenance
  schema consumed by validation and derivation, not a workflow vocabulary table.
  DEFAULT_ARTIFACT_FIELDS derives from PromptExample artifact annotations instead
  of maintaining another list. Both ALL_CAPS declarations have real boundaries.
- _artifact_values supports declared example fields and metadata-backed custom
  artifact fields. _assert_readable translates filesystem/image errors with row
  context. These are external input operations, not internal tensor gates.
- resolve_prompt_example_references serves training while preserving target
  identities for SFT shard lookup. resolve_prompt_example_artifacts also resolves
  targets for offline encoding. Merging them into one default path would change
  the target identity contract, which existing tests explicitly cover.
- resolve_required_reference_images_ fills the configured default and requires
  actual files for image-conditioned families. It is deliberately an in-place
  operation, unlike the copy-returning path functions. All reuse the established
  path mechanism where applicable; no additional safe-path owner is needed.
- Report episode intersection is a diagnostic warning and deterministic sorted
  output. Do not silently promote it to another training gate during cleanup.

Non-goals: new path validators, broader input-type support, image decoding policy
changes, deep copying examples, changing target identity or adding tests for
hypothetical malformed internal objects. Copy-returning reference resolution is
not a deep immutable snapshot of metadata/request_overrides.

33 existing artifact manifest, Video2World, reference metadata and Anima artifact
resolution tests passed on CPU, with dependency deprecation warnings. Ruff check
and format check passed for artifacts.py. No new tests, examples or gates were
added. The metadata cleanup relies on the typed PromptExample API; arbitrary
objects without that field are not an additional supported report input.
