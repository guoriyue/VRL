# Model Taxonomy

VRL classifies the **trainable policy exposed by a registry entry**, not an
entire checkpoint and not its package directory. Hybrid checkpoints can contain
several generation stages with different mathematics; only the stage producing
RL actions determines `PolicySemantics`.

## Typed policy semantics

`vrl.families.semantics.PolicySemantics` records four typed facts:

| Field | Values | Meaning |
|---|---|---|
| `generation_regime` | `full_sequence`, `token_autoregressive`, `chunk_autoregressive` | Whether policy generation updates all output positions together, advances one token from a prefix, or advances one temporal chunk from earlier chunks |
| `step_kind` | `denoise`, `token` | The unit advanced by the policy loop |
| `action_distribution` | `continuous`, `categorical` | The policy action space; continuous includes flow/Gaussian transitions |
| `trajectory_layout` | `denoise`, `token`, `multisegment_token` | The replay/reward record shape; multisegment is not a generation regime |

These three labels normalize paper-familiar vocabulary; they are not a formal
three-value standard defined by one paper. [ACDiT](https://arxiv.org/abs/2412.07720)
uses “full-sequence diffusion”, “token-wise autoregression”, and “blockwise
autoregressive”; [MAGI-1](https://arxiv.org/abs/2505.13211) uses chunk-wise
autoregressive denoising. `causal` remains available for an actual dependency or
attention property such as `causal_attention` or
`block_causal_attention`; it is not a generation-regime value. `full_sequence`
also does not mean one-shot generation: a denoise policy still takes multiple
solver steps, with each step updating the full output field.

## Current executable profiles

| Profile | Current families |
|---|---|
| `full_sequence + denoise + continuous + denoise` | SD3.5, Flux, Qwen-Image, SANA, Lumina2, Hunyuan image/video, Mochi, PixArt-Sigma, CogVideoX, Wan, Cosmos variants, Anima, Echo |
| `token_autoregressive + token + categorical + token` | Janus-Pro, Emu3, GLM-Image, LlamaGen |
| `token_autoregressive + token + continuous + token` | NextStep-1 |
| `token_autoregressive + token + categorical + multisegment_token` | Janus-Pro R1 |
| `chunk_autoregressive + denoise + continuous + denoise` | No executable families yet |

## Future chunk-autoregressive support

The first support targets for the empty chunk-autoregressive denoise profile are:

| Candidate | Planned profile | Integration position |
|---|---|---|
| [CausVid](https://github.com/tianweiy/CausVid) ([paper](https://arxiv.org/abs/2412.07772), [weights](https://huggingface.co/tianweiy/CausVid/tree/main/autoregressive_checkpoint)) | `chunk_autoregressive + denoise + continuous + denoise` | First technical integration candidate. It is built on Wan2.1-T2V-1.3B and advances multi-frame chunks causally while denoising within each chunk, so it is closest to VRL's existing Wan seam. Promotion is gated on upstream maturity and checkpoint licensing: the repository is WIP and the released weights are non-commercial. |
| [MAGI-1](https://github.com/SandAI-org/MAGI-1) ([paper](https://arxiv.org/abs/2505.13211), [weights](https://huggingface.co/sand-ai/MAGI-1)) | `chunk_autoregressive + denoise + continuous + denoise` | Contract and second integration candidate. It advances fixed 24-frame causal chunks and denoises every chunk jointly. Its custom, larger runtime makes it a useful semantic reference but a later implementation target than the Wan-based path. |

These are roadmap candidates, not `FAMILY_REGISTRY` entries or runnable recipes.
The first implementation should own its chunk/cache lifecycle in a family-specific
binding. Extract `composition/chunk_autoregressive` only after another
implementation proves a shared boundary; do not add placeholder packages for
the roadmap.

Self-Forcing remains a related candidate only when named by exact executable
variant. Its released chunk-wise DMD policy fits this profile, while the
Self-Forcing family name alone does not: the method also includes a frame-wise
variant with different generation regime.

Two hybrid cases show why this scope matters:

- GLM-Image exposes a trainable token-autoregressive categorical-token prior
  followed by a frozen full-sequence denoise renderer. Its policy semantics are
  `token_autoregressive + token`, not “mixed model”.
- Cosmos3 contains a causal reasoner and a full-sequence vision generator, but
  VRL trains the vision policy stream. Its current entry is
  `full_sequence + denoise`.

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
chunk-autoregressive denoise entry must therefore register that profile directly; it
cannot pretend to be conventional full-sequence denoise merely to reuse a branch.

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
  composition/token_autoregressive/   reusable ordered-prefix state machine
  bindings/full_sequence_denoise/     full-sequence × denoise binding
  bindings/token_autoregressive/      token-autoregressive binding
  execution/                  step-neutral chunk planning, pipelining, and workers
  ray/                        distributed lifecycle and transport
```

Model and experiment presets are family-first too:
`vrl/config/presets/model/<family>/` and
`vrl/config/presets/experiment/<family>/`. Do not restore `ar/` or `diffusion/`
as routing directories. A family path is stable even if a later executable
variant uses different policy semantics.

There is intentionally no empty `composition/full_sequence` or
`composition/chunk_autoregressive` package. Full-sequence orchestration currently
has one concrete binding shape and remains in `full_sequence_denoise`.
Chunk-autoregressive composition should land with its first executable policy,
then share machinery only when a second implementation proves the common
boundary. `SampleChunk` is execution batching over requests/samples, not an
autoregressive temporal chunk.

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
`GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR` remains an import-path protocol value
used across the neutral registry/worker boundary. They are legitimate
module-level constants; do not duplicate them into parallel `SUPPORTED_*`
vocabularies or mix provider capability tables into workflow code.

This reorganization does not introduce a UNet/DiT taxonomy, rename mathematical
algorithm names such as DiffusionNFT, flatten family facades, or create symmetric
modules without a real owner and consumer.
