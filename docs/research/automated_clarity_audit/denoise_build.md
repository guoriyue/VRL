# Shared denoise construction and replay residency

Reviewed the complete steps/denoise/build.py, registry dispatch and all production
assemble_replay_bundle callers (Echo, MiniMax-H3, VDN-H3, CausVid, Anima, Cosmos3),
with scoped inspection of Cosmos3 loading/replay ownership and Ray's colocated
memory consumer. Previous audit commit: 38b48bdb6. Custom family modules remain
pending their own full producer/executor contracts.

## Changes

- assemble_replay_bundle accepts an explicit loads_full_generation_modules option,
  defaulting to the existing minimal-replay behavior. Cosmos3 declares True.
  Cosmos3Model.from_build loads a pipeline including VAE; Cosmos3ReplayModel
  retains that same pipeline in _pipeline. Previously the shared assembly hardcoded
  False, bypassing the colocated memory guard despite retaining generation weights.
- Correct Cosmos3ReplayModel's misleading "weights-free pipeline shell" docstring.
  Correct shared-builder documentation: registry dispatch, not direct Ray dispatch.
- Consume ModelBuild.torch_compile directly: None means disabled/out of scope,
  otherwise the property supplies an enabled configuration. Remove the redundant
  empty-dict fallback and second enable-key check in replay assembly.

The new option communicates a real family-owned memory fact, not a second
validation layer or a family-name lookup table. The shared assembler still owns
RuntimeBundle construction; no post-construction mutation or parallel constructor
is needed for this exception.

## Retain and why

- Shared rollout construction and replay assembly apply different lifecycle
  policies: replay must not apply rollout quantization or generation memory hooks.
  Keep both functions; merging them into one large flag-driven builder obscures
  the ordering and resource distinction.
- The local move_to_device callback is the optimization pass's placement seam.
  It closes over this model/build, placing quantized weights before compilation
  while respecting offload. It is not an independently reusable service class.
- _check_requires_lora shares the same early descriptor constraint across rollout
  and replay. Descriptor/type/family checks protect these directly callable APIs,
  even though normal registry dispatch already knows the entry.
- Generic replay validates its role before loading; assemble_replay_bundle also
  validates because custom builders call it directly. Removing the latter based
  only on the generic call stack would weaken the other entry paths.
- prepare_replay runs after transformer/scheduler creation and before shared
  LoRA/compile assembly. FLUX needs that family hook for dynamic timesteps.
- Custom builders retain real construction differences: two H3 schedulers, VDN
  attention grafting, non-Diffusers Echo loading, explicit Anima artifacts and
  Cosmos3 pipeline segment methods. No speculative descriptor fields were added
  to absorb those operations. __all__ is a public API list; there is no module
  business-vocabulary table to extract.

## Verification and compatibility

The new CPU test creates a real tiny Cosmos3 pipeline, substitutes checkpoint I/O
only, and runs the actual replay builder. It verifies the retained pipeline has
VAE parameters and declares generation residency. It failed on the old False
flag and passed after the change. 59 Cosmos3, minimal replay wiring and LoRA/
quantized construction tests passed. Four existing memory guard tests also passed,
including strict colocated rejection versus non-colocated acceptance. Ruff passed on
all four changed Python files.

Cosmos3 colocated runs now emit the existing memory warning; with
VRL_STRICT_REPLAY_MEMORY_GUARD enabled they reject construction rather than silently
bypassing the guard. Other callers retain False by default. Numerical computation,
loaded weights and device placement do not change. Tests do not measure GPU/host
memory or prove full pretrained model parity.

Deferred: making Cosmos3 replay genuinely minimal requires separating pipeline
segment-building helpers from VAE ownership (or storing sufficient packed replay
data). This fix reports current ownership accurately; it does not claim to reduce
memory. Some custom replay builders reach the role check only after loading;
normal registry dispatch supplies replay builds, but earlier direct-API rejection
can be assessed during those family reviews without duplicating checks everywhere.
