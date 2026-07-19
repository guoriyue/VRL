# SPRINT: Remove fallback defaults from the full-sequence denoise layout value object

**Date:** 2026-07-18  **Status:** PLANNED

This is a focused follow-up from the args/settings audit. The target is the
second copy of executor fallback values stored on `DiffusionRequestLayout`, not
the legitimate family-specific executor construction defaults.

## Current evidence

`DiffusionChunkExecutorBase` declares five fallback values:

```text
default_samples_per_chunk = 1
default_num_frames = 1
default_fps = None
default_max_sequence_length = 512
sde_type = "flow_grpo"
```

`DiffusionRequestLayout` repeats the same five values as dataclass defaults.
The executor's `layout` property always supplies all five, so the layout-side
defaults are a second handwritten source on production execution paths.

The generic executor and custom family executors also expose constructor or
class fallbacks. Those values are not globally identical and are behaviorally
meaningful: generic execution commonly starts at eight samples per chunk while
several custom executors require one. They must not be collapsed into one
repository-wide value.

`samples_per_chunk: auto` is a separate mechanism. The Ray runtime probes the
active executor, rewrites the request with an integer, and only then delegates
generation. The generic constructor's default of eight is not the resolution
of `auto`.

Finally, `DiffusionChunkGatherer.gather_chunks` creates a zero-argument layout
only to call `ordered_chunks`. Sorting and row-coverage validation do not read
any parsing fallback, so that construction falsely couples gathering to layout
configuration.

## Planned change

1. Make all five `DiffusionRequestLayout` fields required. Executor instances
   continue to construct the layout explicitly from their resolved fallback
   values.
2. Extract chunk ordering and leading-row validation into a clearly named
   module-level function in `full_sequence_denoise/layout.py`. The executor and gatherer
   share it directly. This thin function is justified as the common
   executor/gatherer validation boundary; it is not a LOC-only extraction.
3. Update parsing tests to construct layouts with explicit fallback values and
   update ordering tests to call the pure ordering function.
4. Add or retain coverage showing that Ray `auto` resolution produces an
   integer request value before layout parsing.

## Completion criteria

- `DiffusionRequestLayout` contains no fallback values and has no zero-argument
  construction sites.
- Gathering does not instantiate a parsing layout.
- Generic and family-specific executor defaults remain unchanged.
- `auto` continues to be owned by the Ray runtime probe rather than any
  constructor fallback.

## Non-goals

- Do not type or replace the open `sampling` mapping in this sprint.
- Do not remove live `chunk_passthrough_keys`; FLUX consumes it for `text_ids`.
- Do not change the config-to-executor projection boundary.
- Do not force every family to use one chunk-size fallback. Cross-family
  consistency of the interface matters more than identical values.

## Verification

- Run `tests/generation/bindings/full_sequence_denoise` and the denoise-step suites.
- Run Ruff only on touched Python files using the repository's required
  fix/format/check sequence.
- Search for zero-argument `DiffusionRequestLayout()` calls and confirm none
  remain.
