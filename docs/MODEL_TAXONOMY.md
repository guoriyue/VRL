# Model Taxonomy

VRL classifies the **trainable policy exposed by a registry entry**, not an
entire checkpoint and not its package directory. Hybrid checkpoints can contain
several generation stages with different mathematics; only the stage producing
RL actions determines `PolicySemantics`.

## Orthogonal policy axes

`vrl.families.semantics.PolicySemantics` records four independent facts:

| Field | Values | Meaning |
|---|---|---|
| `temporal_organization` | `joint`, `causal`, `causal_chunked` | Whether one policy step updates the whole output field, advances ordered positions from a prefix, or advances temporal chunks with carried state |
| `step_kind` | `denoise`, `token` | The unit advanced by the policy loop |
| `action_distribution` | `continuous`, `categorical` | The policy action space; continuous includes flow/Gaussian transitions |
| `trajectory_layout` | `denoise`, `token`, `multisegment_token` | The replay/reward record shape; multisegment is not a temporal organization |

`joint` is deliberately used instead of `bidirectional`. Diffusion solvers are
sequential over noise levels, while the model updates the output positions as
one field; “bidirectional” would incorrectly describe an attention
implementation rather than generation organization.

## Current executable profiles

| Profile | Current families |
|---|---|
| `joint + denoise + continuous + denoise` | SD3.5, Flux, Qwen-Image, SANA, Lumina2, Hunyuan image/video, Mochi, PixArt-Sigma, CogVideoX, Wan, Cosmos variants, Anima, Echo |
| `causal + token + categorical + token` | Janus-Pro, Emu3, GLM-Image, LlamaGen |
| `causal + token + continuous + token` | NextStep-1 |
| `causal + token + categorical + multisegment_token` | Janus-Pro R1 |
| `causal_chunked + denoise + continuous + denoise` | Reserved for Self-Forcing once its executable family lands |

Two hybrid cases show why this scope matters:

- GLM-Image exposes a trainable causal categorical-token prior followed by a
  frozen joint-denoise renderer. Its policy semantics are causal/token, not
  “mixed model”.
- Cosmos3 contains a causal reasoner and a joint vision generator, but VRL trains
  the vision policy stream. Its current entry is joint/denoise.

If one checkpoint supports multiple executable policies, register distinct
entries or variants. Do not mutate a checkpoint-level label based on the
selected algorithm; `janus_pro` and `janus_pro_r1` demonstrate this rule.

## Semantics versus runtime bindings

Semantics must not select unrelated implementation behavior. The registry binds
the executor and gatherer explicitly and separately publishes concrete runtime
capabilities such as torch-compile support, request chunk-size arguments, CuMem
parking, and frozen-component parking.

The former flattened `collector_kind` has been removed. Production code reads
`policy_semantics` for policy classification and the explicit executor,
gatherer, or runtime capability for implementation behavior. A future
causal-chunked denoise entry must therefore register that profile directly; it
cannot pretend to be conventional joint denoise merely to reuse a branch.

## Physical layout

The repository now uses a family-first physical layout. Directories answer
ownership questions; `PolicySemantics` answers classification questions:

```text
vrl/models/
  families/<family>/          checkpoint-, backbone-, and replay-specific code
  steps/{denoise,token}/      shared model contracts, builders, and step helpers

vrl/generation/
  steps/denoise/              denoise config, hot loop, and TeaCache
  steps/token/                token-step protocol
  composition/causal/         reusable ordered-prefix state machine
  bindings/joint_denoise/     request/layout/executor/gatherer for joint × denoise
  bindings/causal_token/      request/layout/executor/gatherer for causal × token
  execution/                  step-neutral chunk planning, pipelining, and workers
  ray/                        distributed lifecycle and transport
```

Model and experiment presets are family-first too:
`vrl/config/presets/model/<family>/` and
`vrl/config/presets/experiment/<family>/`. Do not restore `ar/` or `diffusion/`
as routing directories. A family path is stable even if a later executable
variant uses different policy semantics.

There is intentionally no empty `composition/joint` or
`composition/causal_chunked` package. Joint orchestration currently has one
concrete binding shape and remains in `joint_denoise`. Causal-chunked composition
should land with its first executable policy, then share machinery only when a
second implementation proves the common boundary. `SampleChunk` is execution
batching over requests/samples, not causal temporal chunking.

The older `Diffusion*` and `AR*` class names remain where renaming them would add
symbol churn without clarifying ownership. They are implementation API names,
not taxonomy. New imports should use the family-first and axis-based package
paths above.

## Architecture hygiene and non-goals

Thin family `runtime.py` and `runner.py` files stay only when they own a real
model protocol, lazy import, tensor adapter, or state machine. Binding
`__init__.py` facades and gatherers stay because registry import paths and the
driver/worker handoff are protocol boundaries. Cross-family consistency here is
more valuable than reducing a few lines.

`FAMILY_REGISTRY` remains a deliberately isolated taxonomy/config table, and
`GENERIC_DENOISE_EXECUTOR` remains an import-path protocol value used across the
neutral registry/worker boundary. They are legitimate module-level constants;
do not duplicate them into parallel `SUPPORTED_*` vocabularies or mix provider
capability tables into workflow code.

This reorganization does not introduce a UNet/DiT taxonomy, rename mathematical
algorithm names such as DiffusionNFT, flatten family facades, or create symmetric
modules without a real owner and consumer.
