# Model Taxonomy

VRL classifies the **trainable policy exposed by a registry entry**, not an
entire checkpoint and not its package directory. Hybrid checkpoints can contain
several generation stages with different mathematics; only the stage producing
RL actions determines its classification.

## Typed policy semantics

`ModelFamilyEntry.generation_regime` (typed by
`vrl.models.families.semantics.GenerationRegime`) records one fact:

| Field | Values | Meaning |
|---|---|---|
| `generation_regime` | `full_sequence` | Policy generation updates all output positions together in every denoise step |

Every registered policy is a continuous denoise step over a denoise-layout
trajectory, so the regime is the only routing fact the registry carries. The
token-autoregressive regime (Janus-Pro, NextStep-1, Emu3, GLM-Image, LlamaGen)
was removed in 2026-09 (tag `archive/token-ar-20260916`); the
chunk-autoregressive regime (CausVid, MAGI-1) was removed in 2026-10 (branch
`archive/main-20261016`).

These labels normalize paper-familiar vocabulary; they are not a formal
standard defined by one paper. `causal` remains available for an actual
dependency or attention property such as `causal_attention` or
`block_causal_attention`; it is not a generation-regime value. `full_sequence`
also does not mean one-shot generation: a denoise policy still takes multiple
solver steps, with each step updating the full output field.

## Current executable profiles

| Profile | Current families |
|---|---|
| `full_sequence` | SD3.5, Flux, Qwen-Image, SANA, Wan, Cosmos variants, MiniMax-H3 / VDN-H3 |

Two hybrid cases show why this scope matters:

- Cosmos3 contains a causal reasoner and a full-sequence vision generator, but
  VRL trains the vision policy stream. Its current entry is `full_sequence`.
- MiniMax-H3 denoises video and audio latents jointly in one packed
  sequence. VRL's action is the video latent (`full_sequence`); the
  audio rows are a deterministic side stream the family steps itself and
  records for replay, so they never enter the policy ratio.

If one checkpoint supports multiple executable policies, register distinct
entries or variants. Do not mutate a checkpoint-level label based on the
selected algorithm.

## Semantics versus runtime bindings

Semantics must not select unrelated implementation behavior. The registry binds
the executor and gatherer explicitly and separately publishes concrete runtime
capabilities such as torch-compile support, request chunk-size arguments, CuMem
parking, and frozen-component parking.

The former flattened `collector_kind` has been removed. Production code reads
`generation_regime` for policy classification and the explicit executor,
gatherer, or runtime capability for implementation behavior. A future
non-full-sequence entry must register its own regime; it cannot pretend to be
conventional full-sequence denoise merely to reuse a branch.

## Physical layout

The repository now uses a family-first physical layout. Directories answer
ownership questions; `generation_regime` answers classification questions:

```text
vrl/models/
  families/
    registry.py               canonical family wiring and lazy build descriptors
    names.py                  canonical names and external aliases
    semantics.py              task and trainable-policy taxonomy
    <family>/                 checkpoint-, backbone-, and replay-specific code
  steps/denoise/              shared model contracts, builders, and step helpers

vrl/generation/
  steps/denoise/              denoise config, hot loop, and TeaCache
  bindings/full_sequence/     full-sequence × denoise binding
  execution/                  step-neutral sample-batch planning, pipelining, and workers
  ray/                        distributed lifecycle and transport
```

Model and experiment presets are family-first too:
`vrl/config/presets/model/<family>/` and
`vrl/config/presets/experiment/<family>/`. Do not restore `ar/` or `diffusion/`
as routing directories. A family path is stable even if a later executable
variant uses different policy semantics.

There is intentionally no generalized `composition/` state-machine layer (the
one that existed served only the removed token regime).
`GenerationSampleBatch` is execution batching over requests/samples, not an
autoregressive temporal chunk.

Generation classes are named for the denoise step they run (`Denoise*`), not
for a model taxonomy. New imports should use the family-first and axis-based
package paths above.

## Architecture hygiene and non-goals

Thin family `runtime.py` and `runner.py` files stay only when they own a real
model protocol, lazy import, tensor adapter, or state machine. Binding
`__init__.py` facades and gatherers stay because registry import paths and the
driver/worker handoff are protocol boundaries. Cross-family consistency here is
more valuable than reducing a few lines.

The three package-root family modules remain import-light even though they are
co-located with the implementations. Config and generation may import the
registry; they must not import a concrete `<family>` package at module scope.

`FAMILY_REGISTRY` remains a deliberately isolated taxonomy/config table, and
`GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR` remains an import-path protocol value
used across the neutral registry/worker boundary. They are legitimate
module-level constants; do not duplicate them into parallel `SUPPORTED_*`
vocabularies or mix provider capability tables into workflow code.

This reorganization does not introduce a UNet/DiT taxonomy, rename mathematical
algorithm names such as DiffusionNFT, flatten family facades, or create symmetric
modules without a real owner and consumer.
