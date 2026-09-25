# Sprint: LoRA configuration and previous-policy ownership

Status: completed implementation and focused CPU verification, 2026-09-17.
Implementation reviewed on `review/combined-20260916`, through `b6e9b3e2`.
This is a completed-work record, not a GPU training acceptance report.

## Outcome

Reviewed all 21 registered model families. Adapter configuration now resolves
through one typed `LoraSection`; shared model construction calls PEFT directly.
There is no custom LoRA manager or new `lora/` package.

The final ownership rule is:

- Model families describe supported forward interfaces and adapter defaults.
- Algorithm contracts declare whether a frozen previous policy is required.
- The config-to-build boundary derives the required adapter layout.
- PEFT owns model wrapping, adapter installation, and activation.
- VRL retains previous-policy copy/EMA scheduling, frozen-state checkpoint
  registration, topology validation, and model residency.

Selecting Predict2.5 alone no longer creates a previous adapter or prohibits
full-parameter training. The current NFT and V-GRPO implementations still
require LoRA because their previous-policy implementation is a PEFT mirror.
That limitation is now validated as an algorithm requirement.

## Motivation and corrected intermediate decision

Originally, initialization, adapter upcasting, and unconditional previous-policy
creation were model-class attributes. Additional settings lived in unrelated
places: Wan's top-level `lora_parameter_dtype`, Flux/SD3.5's
`nft_previous_adapter`, and an accepted but unconsumed `model.lora.init` alias.
`ModelBuild` also maintained a second, hand-written adapter-key projection.

The first cleanup moved defaults into lightweight family schemas and briefly
preserved Predict2.5's behavior as `always_previous_adapter=True`. That was
insufficient: moving a recipe decision into a model schema does not fix its
ownership. The subsequent commits removed that flag and the public
`model.lora.previous_adapter` setting. They must not be reintroduced as the
current design.

## Final implementation

### Typed adapter configuration

[`LoraSection`](../../../vrl/config/model_schema.py) owns rank, alpha, targets,
dropout, initialization, warm-start path, PEFT adapter upcasting, and the
optional FP32 parameter-storage override. `ModelSection.resolve_lora()` merges
explicit values over family defaults without mutating either.

[`ModelBuild`](../../../vrl/models/interfaces/runtime.py) exposes the typed
settings instead of another dictionary projection. The separate `lora_path`
view and dead `init` alias were removed. Model presets retain their existing
rank, alpha, and target-module lists; these are real recipe inputs, not
duplicates to replace with one universal list.

Defaults preserved:

| Family | Fresh initialization | PEFT adapter upcast |
| --- | --- | --- |
| Wan 2.1 / I2V, including Wan 2.2 presets using those families | `True` | `False` |
| CausVid | `True` | `True` |
| Other families using shared attachment | `"gaussian"` | `True` |

Both `True` and Gaussian initialization zero B and initially preserve the
base output; A initialization and RNG behavior differ. MAGI remains
generation-only without VRL-managed LoRA. Anima's rank/alpha and all other
family target lists were left unchanged.

### Algorithm-owned previous policy

[`AlgorithmConfigContract`](../../../vrl/algorithms/config_contract.py) declares
`requires_previous_adapter`. NFT and V-GRPO set it to true; other objectives
default to false. [`check_cross_section_rules`](../../../vrl/config/rules.py)
rejects unsupported model interfaces and full-parameter NFT/V-GRPO before
model loading.

[`ModelFamilyEntry.resolve_model_build`](../../../vrl/models/families/registry.py)
derives `ModelBuild.previous_policy_adapter` from the selected algorithm.
This is a resolved construction requirement, not a user-overridable YAML key
or an algorithm object embedded in the model.

Flux, SD3.5, and Predict2.5 retain `supports_previous_adapter` as a capability.
The shared [`DiffusionModelBase`](../../../vrl/models/steps/denoise/base.py)
attaches the mirror only when requested by the build. The mirror retains the
installed adapter's topology and effective storage policy. A real PEFT test
caught and fixed a BF16-default/FP32-previous mismatch when upcasting was off.

Both replay and rollout currently receive the same layout requirement. This
preserves existing Ray construction and identity matching; removing mirrors
from rollout workers was not part of this change. Consequently, evaluation
using a complete NFT/V-GRPO training config retains the mirror, while a
model-only generation config does not create one.

Predict2.5's recipe-specific `apply_full_finetune()` rejection was deleted.
It now inherits the shared full-finetune implementation. CPU forward/backward
coverage confirms this path works on a tiny real Cosmos transformer; it does
not establish a full-size GPU memory budget or a validated training recipe.

### Responsibilities deliberately retained

- Direct PEFT wrapping and `get_base_model()` use; no reimplementation of
  PEFT internals. Unwrapping for Wan block offload still has a real consumer.
- The shared previous-policy copy/EMA and checkpoint-registration helpers:
  these implement VRL's training lifecycle, not PEFT adapter management.
- Lightweight family schemas as capability/default boundaries, even where
  they are small. Uniform family entry points remain useful.
- Residency hooks, quantization ordering, synchronization, and the algorithms'
  existing optimizer-step update semantics.

No new business-vocabulary constants or generic manager abstractions were
introduced. The family checks remaining in checkpoint identity describe a
persisted format, not a policy-selection rule.

## Configuration and checkpoint migration

| Removed setting | Action |
| --- | --- |
| `model.lora_parameter_dtype` | Use `model.lora.parameter_dtype` |
| `model.nft_previous_adapter` | Remove; the algorithm requests the mirror |
| `model.lora.previous_adapter` | Remove; this intermediate spelling was also retired |
| `model.lora.init` | Remove; use `model.lora.init_lora_weights` if initialization was intended |

Removed spellings fail config validation rather than silently aliasing.
Historical run snapshots remain historical and need migration before reuse.

[`checkpoint_identity.py`](../../../vrl/models/checkpoint_identity.py) preserves
the v1 canonical names for stored identities. Flux/SD3.5 recorded
`nft_previous_adapter`; Predict2.5 originally implied a mirror without storing
that key. Existing NFT identities retain that representation. New Predict2.5
runs without the mirror record false explicitly, so old GRPO checkpoints
containing an unconditional mirror are not accepted as identical exact-resume
topologies. This is intentional, not evidence of a checkpoint conversion tool.

The earlier configuration-only migration was checked against the preceding
implementation on eight identity cases with exact equality. Final tests also
cover the algorithm-selected layout and replay/rollout identity agreement.
No full-size checkpoint restore or GPU training run was performed here.

## Miles reference and limits of the analogy

Inspected Miles Diffusion at pinned revision
`ebd55fc1e597520322e0225997552c0807e9293b`:

- [PEFT attachment](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/miles/backends/fsdp_utils/actor.py#L603)
  constructs `LoraConfig` and calls `get_peft_model` directly.
- [Reference selection](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/miles/utils/arguments.py#L1544)
  derives the default from the loss/reference settings, not model family.
- [Reference forward](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/miles/backends/fsdp_utils/actor.py#L553)
  uses EMA swapping or PEFT `disable_adapter()` for a base-model reference.

Borrow the separation of responsibilities, not different objective semantics:
EMA, a frozen base, and VRL's previous-policy mirror are not interchangeable.
This sprint did not replace VRL's reference policy with Miles's implementation.

## Reviewable commit sequence

These are dependent review slices, not claims that every intermediate commit
independently passes the complete final suite. Unrelated Ray commits between
the slices are outside this sprint.

| Commit | Change |
| --- | --- |
| `2fb8dd9f` | Unwrap through the public PEFT API |
| `1f5af3e9` | Install PEFT adapters directly in the shared model |
| `354036a1` | Centralize adapter settings in family schemas |
| `4865977a` | Expose typed settings from ModelBuild |
| `075b57e3` | Consume resolved settings and match previous-adapter dtype |
| `9fa2a11c` | Preserve identity across configuration migration |
| `15fd5609` | Document the first configuration cleanup |
| `5828b92b` | Move previous-policy requirements to algorithm contracts |
| `dc234620` | Derive adapter construction from those requirements |
| `ff77decc` | Remove Predict2.5's recipe-specific full-finetune rejection |
| `073d1f0b` | Track algorithm-selected topology in checkpoint identity |
| `b6e9b3e2` | Document the final ownership boundary |

## Verification and remaining boundaries

Final focused CPU regression, rerun before committing: **862 passed, 1 skipped,
17 deselected**. Ruff check/format checks and `git diff --check` passed.
The earlier LoRA configuration stage passed 728 tests; these counts represent
different suite selections/stages and must not be added together.

Coverage includes all-family default resolution, bundled configuration parsing,
algorithm admission, Ray build-payload round trips, replay/rollout identity,
real PEFT mirror dtype/freeze/copy behavior, warm-start topology checks,
checkpoint adapter export, NFT/V-GRPO tests, and tiny Cosmos full-parameter
forward/backward. Tests were extended in existing files; no standalone probe
or test framework was added.

Not implemented or validated by this sprint:

- Full-parameter previous-policy storage for NFT/V-GRPO.
- Full-size Predict2.5 GPU training, convergence, or memory feasibility.
- Trainer-only mirror allocation with mirror-free rollout workers.
- Automatic conversion of old Predict2.5 GRPO checkpoint topology.

User-facing configuration reference: [LoRA configuration](../../CONFIGURATION.md#lora-configuration).

## Follow-up: objective-neutral model surface (2026-09-17)

The sprint above left two objective-shaped pieces in the model package:
`supports_previous_adapter` on the Flux/SD3.5/Predict2.5 schemas and a
`diffusion_nft_prepare_transformer_input` method on the same three models (twice
on Predict2.5). Inspecting Miles Diffusion showed that its NFT and Flow-GRPO
losses share one family forward (`compute_noise_pred`) and no family carries an
objective flag. The follow-up applies that separation:

- `DiffusionModelBase.replay_forward_with_latents` gained a keyword-only
  `classifier_free_guidance` override. DiffusionNFT and V-GRPO call it
  directly on the re-noised clean latent under the `previous`, trainable, and
  adapter-free contexts, so the families' rollout forward is the only forward.
- The four `diffusion_nft_prepare_transformer_input` implementations, the
  `supports_previous_adapter` flag, and the Flux/SD3.5 `from_build` /
  `prepare_replay` re-runs of `require_lora_for_previous_policy_adapter` are
  deleted. The shared build entry points still run that guard once.
- `check_cross_section_rules` admits NFT/V-GRPO for any full-sequence family
  with a trainer replay recipe (registry facts, torch-free); `flow_time`
  remains the grid-domain guard at the first loss.
- Behaviour change on Predict2.5: the removed builder fed the raw `[0, 1000]`
  UniPC timestep to a transformer that takes the flow sigma in `[0, 1]`; the
  shared path feeds `scheduler.sigmas[idx]` like rollout and SDE replay. A
  real tiny-Cosmos test pins this. SD3.5 and Flux produce the same kwargs as
  before (the SD3.5 test asserts equality with the conditional `forward_step`).

Not validated here: full-size NFT/V-GRPO GPU runs on any family, and the
families newly admitted by the rule beyond SD3.5/Flux/Predict2.5.
