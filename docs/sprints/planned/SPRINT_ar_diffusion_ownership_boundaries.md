# Sprint: AR/diffusion role separation and engine-owned loading

Status: superseded in scope by the user-approved token-AR removal on 2026-09-16.
See `docs/sprints/SPRINT_drop_token_autoregressive.md` and archive tag
`archive/token-ar-20260916`. The AR family migration, token algorithm admission
and cross-token verification work below are historical proposals, NOT queued
work. Only the diffusion generation/replay and engine-owned loading principles
remain applicable; re-scope their implementation against the diffusion-only
tree before execution. Do not reintroduce AR to satisfy this old plan.

Original status: planned; audit and design only. No runtime migration is implemented by
this document. Final consolidated plan includes the requested Miles-style
loading/residency decision. Review began on 2026-09-16 at HEAD `1d9e8ef6`, including existing
uncommitted artifact/offload changes. During review, concurrent work committed
them as `5f1362c6`; final checks use that revision. Findings below distinguish
the old behavior from the partially completed transport migration. Do not
reimplement already-landed changes.

## Decision

Keep one repository and the shared training infrastructure. Separate policy
execution and resource ownership inside it. Do not create two copied trainers,
a universal model hierarchy, or a second public configuration system.

The relevant distinctions are independent:

- Generation order: full sequence, token autoregressive, or chunk autoregressive.
- Policy action: a denoise transition or token action.
- Distribution: categorical or continuous. NextStep is continuous token AR.
- Lifecycle role: generation versus replay. This is not diffusion versus AR.

The registry already models these distinctions in
`vrl/models/families/semantics.py::PolicySemantics`. Preserve real consumers;
do not introduce another `is_diffusion` switch or a parallel family taxonomy.
Shared GRPO concepts do not imply identical replay math. DiffusionNFT also
uses a direct replay-data objective rather than the log-prob evaluator path.

Loading decision: keep persistent generation workers and let their concrete
backend own model loading and residency. The trainer loads its replay model
separately. The framework retains admission, policy-version coordination and
safe resource handoff. Reuse existing lifecycle calls, not a renamed duplicate
sleep/wake API. Adopting SGLang is a separately gated backend integration, not
a prerequisite for removing Wan hook knowledge from the shared worker.

## Scope and evidence map

This is a repository-wide ownership review, not a claim that every model has
been run or every source line certified. The review follows public boundaries,
their concrete producers/consumers, family inheritance, and representative
tests across these domains:

| Domain | Current sources | Decision |
| --- | --- | --- |
| Family identity and capability selection | `vrl/models/families/{registry,semantics}.py` | Keep one authoritative registry; admit only meaningful algorithm/step combinations. |
| Runtime contracts and builders | `vrl/models/interfaces/{replay,runtime}.py`, `vrl/models/steps/{denoise,token}/build.py` | Keep narrow contracts and separate builders; separate generation ownership from replay reuse. |
| Family model implementations | `vrl/models/families/*/model.py`, nested Cosmos families | Migrate generation-inheriting replay implementations by family group, not one global base-class replacement. |
| Generation execution | `vrl/generation/{protocols,types}.py`, `execution/`, `ray/`, `bindings/` | Runtime owns scheduling; bindings own policy payloads; one explicit media boundary. |
| Rollout and scoring | `vrl/rollouts/collector/`, `orchestration/`, `vrl/rewards/{base,artifacts,protocols,runtime}.py` | Shared sample identity, scoring and orchestration; media ownership must work for every supported binding. |
| Replay and objectives | `vrl/rollouts/evaluators/{denoise,token}/`, `vrl/algorithms/` | Preserve distinct likelihood and trajectory semantics; share only actual common math. |
| Trainer, parking and persistence | `vrl/trainers/online/`, `vrl/trainers/{strategy,checkpointing,weight_sync}.py`, `vrl/models/parking.py` | Trainer owns optimizer/EMA/RNG and training residency; do not duplicate by model category. |
| Configuration and launch | `vrl/config/`, `vrl/run.py`, `vrl/scripts/common/{factory,online}.py`, `vrl/ray/resources.py` | Preserve composition and topology-derived lifecycle; reject incompatible combinations before expensive loading. |
| Optimizations and performance tools | `vrl/nn/optimization/`, `vrl/scripts/perf/` | Backend-specific optimizations remain opt-in; telemetry is not a training interface or proof of acceptance. |

### A. Replay inherits generation implementation on both sides

Examples:

```python
class WanT2VReplayModel(ReplayRolloutStubs, WanT2VDiffusersModel):
class SD3_5ReplayModel(DiffusersReplayModelBase, SD3_5Model):
class JanusProReplayModel(ARReplayRolloutStubs, JanusProModel):
```

Sources: `vrl/models/families/wan_2_1/model.py`,
`vrl/models/families/sd3_5/model.py`, `vrl/models/families/janus_pro/model.py`.
The stub implementations are in `vrl/models/steps/denoise/base.py` and
`vrl/models/steps/token/base.py`. Constructors bypass generation construction,
while the method-resolution order still exposes generation behavior.

Wan demonstrates a real state consequence, not just an unattractive class name:
`load_trainable_state()` reads `uses_pipeline_cpu_offload`, which reads
`_pipeline_offload`. Replay initializes that field to `None` despite owning no
pipeline. Deleting the initializer alone breaks inherited weight loading.

Migration inventory:

- Generic diffusers replay pattern: SD3.5, Flux, Sana, Qwen Image, CogVideoX,
  Hunyuan Image/Video, Mochi, Lumina2, PixArt Sigma, Cosmos Predict2,
  Predict2.5, and Cosmos3.
- Custom denoise replay: Wan T2V/I2V and dual experts, Echo, Anima,
  MiniMax H3 and VDN-H3.
- Token replay: Janus (including R1 use), Emu3, GLM Image, LlamaGen,
  and NextStep. Keep discrete/continuous and multi-segment differences.
- CausVid already uses sibling generation/replay classes over
  `_CausVidPolicyModel`; that is useful prior art, not a finished template.
  Its base still has generation-disabled methods and `generation_enabled`
  branches. Generation-only registry entries must not acquire fake replay.

Do NOT turn this into a blanket ban on encoders or metadata in replay:
LlamaGen's `replay_forward()` calls `encode_caption()` with recorded prompt
IDs/masks and legitimately needs frozen T5. Cosmos3 retains a VAE temporal
scale value, not VAE weights. H3 needs both audio and video schedulers; its
current component shell has unused generation slots set to `None`.

### B. Residency ownership is already partly separated

`WorkerMemoryParking` owns generation-worker park/wake and physical release
evidence. `Strategy` / `TrainingStateParking` own training-state parking.
Keep this separation and the shared-GPU handoff checks.

The leak is in `vrl/generation/execution/memory_parking.py`: generic worker
policy probes `uses_pipeline_cpu_offload`, `pipeline_cpu_offload_healthy` and
`reset_pipeline_cpu_offload`. Wan owns those hooks, but its replay twin inherits
their state. `vrl/nn/optimization/passes.py::PipelineOffloadPass` also checks
installation health; that is a legitimate generation construction consumer.

The condition is necessary today: CuMem and pipeline hooks cannot both own
residency, and failed hooks must not be reused. Replace ownership carefully,
not the boolean with another decorative name. Do not replace measured parking
with unconditional `.cpu()` or remove fail-closed recovery.

Placement risk to verify, not yet a hardware-proven bug: generation's generic
`ModelParking.park()` path does not request the tensor-device preservation
used by `TrainingStateParking`. LlamaGen's CPU T5 and diffusion CPU prompt
encoders make heterogeneous original placement a real case. A sleep/wake test
must preserve intended devices/dtypes rather than merely move everything back
to the worker's CUDA device.

### C. Artifact transport exposed a binding leak

The pre-fix worker assumed `output.video`. Commit `5f1362c6` introduces
`reward_media`, `media_off_wire`, primary-rank writing and gather propagation.
Keep its intent, but do not mark it accepted yet:

- Token gather still lists `output` among required concatenated tensor fields
  (`bindings/token_autoregressive/executor.py::merge_generation_batches`).
- Chunk gather still concatenates `batch.output` unconditionally
  (`bindings/chunk_autoregressive_denoise/gather.py`).
- With files-only scoring, setting that media to `None` must survive gather;
  adding an accessor at the producer alone is insufficient.
- NextStep owns its own batch result/gather in
  `vrl/models/families/nextstep_1/runtime.py`; a discrete-AR fix does not cover it.
- `GenerationWorkerCore.execute_request_pipelined()` returns the executor's
  whole-plan result without the per-batch artifact hook. The Ray executor can
  choose that path without an artifact-specific guard. Define equivalent
  finalization or explicit admission, rather than silently bypassing the
  optimization. Existing driver fallback may preserve scoring; this is not
  evidence that every such request crashes.
- File ownership begins at reward-store adoption, but worker parallel writes
  have no rollback when a later write/digest fails. Failed gather or cancellation
  before scoring can also leave produced files with no receiving owner. Closing
  the normal-path rank leak does not close this pre-adoption gap.

Sources: `vrl/generation/execution/worker.py`,
`vrl/generation/ray/executor.py`, `vrl/rollouts/collector/batch_builder.py`,
`vrl/generation/execution/reward_artifacts.py` and the bindings above.

### D. Algorithm admission needs the step semantic, not just distribution

`vrl/scripts/common/factory.py::AlgorithmEvaluatorPair` chooses
`ContinuousTokenLogProbEvaluator` for `token_grpo` whenever the distribution is
continuous. Diffusion is also continuous. A CPU configuration reproduction
successfully builds SD3.5 with `token_grpo` and that evaluator:

```python
load_config("experiment/sd3_5/online_grpo_ocr",
            overrides=["/recipe/online=categorical_token_grpo"])
```

The checked path includes `build_configs`, collector configuration and factory
construction. This is a real composable-config input, not a made-up token type.
Reject the incompatible step/layout before model loading. Preserve legitimate
continuous-token NextStep and causal-chunk denoise paths; do not derive support
from model-name substrings or maintain a second model-by-algorithm matrix.

### E. Training knobs do not have consistent role ownership

`scripts/common/online.py` applies `actor.gradient_checkpointing` only when
`step_kind == "denoise"`. NextStep separately carries a model-level
`gradient_checkpointing` default through `families/nextstep_1/{config,runtime,model}.py`.
Establish one training-role source of truth; family installers may differ, but
the same public request must not be silently ignored by a token family.

`TrainerConfig.timestep_fraction` is also required in the common configuration,
while trajectory replay returns a single logical index before consuming the
denoise subset settings. This is a misleading applicability issue, not proof
that token gradients are wrong. Resolve applicability without another full
configuration hierarchy.

### F. Public import boundaries need an explicit ownership decision

The existing architecture check flags generation imports of
`models.steps.denoise.common.tensors.expand_tensor_to_batch` and
`models.parking.{CumemPool,ModelParking}`. These predate the media fix.
The former is neutral tensor expansion with a misplaced domain dependency;
the latter is a real shared storage substrate. Place the tensor primitive with
its genuine cross-domain consumers; explicitly expose or relocate the parking
substrate rather than scattering exemptions or wrapping it solely to satisfy
a filename check. Preserve the policy/physical-storage ownership distinction.

## Target ownership

| Owner | Owns | Must not own |
| --- | --- | --- |
| Family policy implementation | Backbone math, conditioning transforms, replay, adapter/checkpoint roots | Pipeline hook state merely to satisfy another role |
| Generation model/binding | Prompt/media encoding and decoding, sampling state, backend-specific residency hooks | Optimizer/EMA/RNG restoration |
| Generation runtime/worker | Scheduling, lifecycle transitions, failure propagation, resource-release evidence | Wan `.pipeline` or AR decoder internals |
| Trainer and strategy | Replay scheduling, accumulation, optimizer, EMA, distributed state and resume | Rollout pipeline construction or reward media encoding |
| Reward runtime/store | Scoring transport, ownership transfer, release/retain | Policy action math or family-specific output attributes |

Do not add all these rows as new classes. Reuse existing owners. A narrow
residency adapter is warranted only where it removes the current worker's
backend-specific probes; it must have concrete implementations and consumers.

## Loading and residency: what actually changes

### Current behavior, not an assumed cold-reload loop

`vrl/generation/execution/worker.py::load_policy()` constructs an executor only
when one is absent. `sleep()` preserves it; `wake()` restores residency without
a disk reload. `release_policy()` is a different, cold-eviction operation and
resets the local version to the launch checkpoint. This behavior must survive.

The outer API already exists:

```text
schedule -> GenerationRuntime.activate / offload / shutdown
                    -> rank actor load_policy / sleep / wake / release_policy
```

Sources: `vrl/generation/protocols.py`, `vrl/generation/ray/runtime.py` and
`vrl/generation/execution/worker.py`. Do not build another manager around these
just to resemble Miles method names.

The needed internal change is below the actor lifecycle: shared worker logic
must not inspect Wan properties to decide how to detach or reinstall pipeline
hooks. Generation construction binds the applicable existing memory mechanism
once. Its concrete implementation handles native module movement, CuMem, or
pipeline hooks; the worker retains admission, state transitions, residual
measurement and quarantine. Do not choose the mechanism again at every layer.

### Three distinct operations

| Operation | Trigger and owner | Required behavior |
| --- | --- | --- |
| Cold load | Generation backend construction; trainer replay construction separately | Read checkpoint, establish precision/adapters/roots and hooks once per live instance. Preserve meaningful family loaders. |
| Phase sleep/wake | Schedule requests a runtime handoff; backend implements it | Yield and restore residency while preserving installed policy version, adapter values and valid cache/state rules. No checkpoint reload. |
| Within-forward streaming | Capacity policy of a generation backend | Move blocks/components during generation when the working set cannot fit. Not replaced by phase sleep/wake. |

A persistent worker avoids repeated construction, not the memory needed for a
model. Sleep may consume host RAM; four replicas can still exceed host capacity.
A model that fits only with block streaming cannot become fully resident merely
because the framework now calls a clean engine API. This sprint does not disable
Ray memory protection, discard precision exceptions, or promise offload-free
Wan 14B operation.

### Native backend target

1. Keep checkpoint/deserialization, pipeline construction, optimization order
   and hook attachment in the generation build/backend path.
2. Move hook-specific reset, verification and repair behind the concrete
   generation residency implementation. The shared lifecycle calls behavior,
   not `getattr(model, "uses_pipeline_cpu_offload", False)` at every transition.
3. Keep generic physical storage primitives in `models/parking.py` available to
   their real consumers through an intentional public boundary. Do not copy
   them into diffusion and AR implementations.
4. Keep trainer state parking separate: optimizer, EMA, gradients/scaler where
   owned, rank RNG and FSDP collective behavior cannot be delegated to rollout.
5. No full rollout model is built merely to produce a replay instance. Load the
   trainable policy and whatever frozen conditioning replay actually consumes.
6. On cold rebuild, invalidate retained slots/cache and reinstall the requested
   policy before serving. On warm wake, retain the installed policy; never
   advertise a version number whose weights have reverted to the base checkpoint.

The original `_pipeline_offload` state can remain internally useful to the
native pipeline backend. Success is correct ownership, not eliminating its
spelling or replacing it with several booleans.

### Optional SGLang-Diffusion backend, after native ownership is sound

This is how VRL could delegate actual generation loading like Miles: an engine
process loads the rollout model, and VRL sends generation, weight-install and
memory-release/resume requests. The engine still performs the loading/offload;
it is not removed from the system. Trainer replay remains local to its owner.

Reuse `docs/sprints/parked/SPRINT_sglang_diffusion_execution_provider.md` for
implementation; do not create a competing provider registry here. Its older
upstream pin and Qwen-first capability assumptions must be rechecked against an
explicit pinned engine revision. The inspected Miles engine adapter is evidence
of an ownership pattern, not proof that every VRL family is supported upstream.

Before admitting one real family through the optional backend, require:

- Actual trajectory actions, timesteps, masks, conditioning and old log-probs
  map into the existing VRL replay contract, not a second trainer schema.
- Initial and updated LoRA/full-policy delivery have named-tensor identity and
  version evidence on the receiving engine, including both Wan experts if used.
- Sample/replay parity and one real optimizer update meet unchanged numerical
  gates; sampling options and precision match the tested native workload.
- Engine sleep/resume demonstrates physical release, policy preservation and
  correct generation after wake. Child-process memory must be attributed too;
  measuring an empty wrapper process is not proof of released engine memory.
- Process failure, request drain/cancellation and cold restart have one owner.
  Do not launch an independently scheduled fleet outside existing reservations.
- Compare equal-work end-to-end timing, cold startup, warm wake, peak host RAM
  and per-rank GPU memory. A fast image endpoint alone is not RL acceptance.

If the engine lacks these capabilities, keep the native backend and record the
specific gap. Do not widen parity tolerance, mask missing data, or make switching
engines the default as part of an ownership refactor.

## Work packages and reviewable commit boundaries

### 1. Close admission and media contracts first

- Add algorithm/step/layout admission at the existing configuration composition
  boundary, before model construction. Reuse registry semantics and objective
  requirements; retain useful errors for direct public factory callers.
- Complete the in-flight media fix through producer, gather, collector, scorer
  and cleanup. Cover full-sequence denoise, discrete token, continuous token,
  multisegment token and causal chunk bindings without unifying their tensors.
- Specify primary-rank ownership and failure cleanup, including lost delivery,
  cancellation and retry. Unknown remote scoring outcomes keep the existing
  retain policy; never delete files a scorer may still read.
- Keep admission and media lifecycle in separate commits. Reuse existing tests;
  do not add a combinatorial family/reward fixture framework.

### 2. Isolate Wan policy and generation residency

- Move shared expert routing, backbone/replay and trainable-state operations
  behind a Wan policy implementation shared by sibling role classes.
- Keep `_pipeline_offload` and hook installation/recovery only on generation.
  Replay loads/verifies weights without inspecting pipeline state.
- Let the existing generation lifecycle owner invoke the backend's residency
  behavior. Preserve installation ordering, CuMem exclusion, health checks,
  mixed-dtype placement, failed-swap recovery and physical release evidence.
- Preserve public class import paths, state-dict keys, checkpoint roots, model
  identity, LoRA dtype and dual-expert ownership. No checkpoint-format migration.
- Do not use this task to move training into a new Ray actor or replace the
  rollout engine with SGLang. Those are separately measurable projects.
- Implement the native loading/residency target above through existing owners.
  Separate the role-inheritance change from the worker residency migration in
  reviewable commits, preserving an executable state after each one.

### 3. Apply the role boundary across families in small groups

- Start with SD3.5 and Janus as one diffusers and one discrete-token example.
  Factor real policy behavior, not whole constructors or empty interface stubs.
- Migrate the remaining inventory above only after checking each family's
  conditioning, scheduler, parameter names and current replay implementation.
- Preserve LlamaGen's needed T5, NextStep flow-head/noise replay, Janus R1 segment
  selection, Wan dual experts, Cosmos conditioning, H3 audio/video and VDN's
  hybrid backbone. These are not expendable exceptions to a universal schema.
- Remove obsolete replay-rollout stubs only after consumers no longer use the
  oversized generation surface. Keep narrow structural protocols; concrete
  classes must not inherit Protocol stubs as implementation.
- CausVid and generation-only families get boundary review, not forced
  conversion to a full-sequence or trainable interface.

### 4. Contain semantic trainer logic, retain shared infrastructure

- Resolve activation-checkpointing settings through the training role once;
  preserve family-specific installers and reject unsupported requests early.
  Do not change NextStep's effective default silently during migration.
- Keep one owner for checkpointing, rank-local RNG, weight delivery, optimizer,
  EMA, accumulation and resource plans.
- Review denoise timestep/SDE-window selection and denoise SFT augmentation in
  `vrl/trainers/online/trainer.py`; keep their logic in existing objective or
  replay-domain owners when extracting actually reduces mixed concerns.
  Do not introduce a new trainer subclass for every policy flavor.
- Trace configuration consumers for activation checkpointing and KL reference
  behavior on both paths. The comment that only denoise evaluators consume KL
  in `scripts/common/online.py` is not a valid rule: token evaluators support
  reference passes too. Correct documentation without inventing a KL regression.
- Separate default changes and numerical optimizations from ownership commits.
  Keep rollout/replay parity, clipping, off-policy and precision gates unchanged.

### 5. Close native lifecycle acceptance; decide the engine pilot separately

- Demonstrate cold load -> update delivery -> generate -> sleep -> wake ->
  generate with preserved policy state; then cold eviction -> reload -> explicit
  weight reinstall. Distinguish construction counters from physical GPU release.
- Run one representative discrete-token and one denoise lifecycle end to end;
  exercise continuous-token and chunk semantics in their focused tests. Do not
  manufacture support for every backend/family combination.
- Measure host-memory and GPU peaks with the actual intended topology. No
  repeated checkpoint loads in healthy warm transitions; no hidden duplicate
  GPU owner introduced by the new boundary.
- Record the optional engine's go/no-go against the gates above. Shipping that
  backend, obtaining speedup, splitting the control/trainer processes and long
  learning experiments remain separate work, not implicit sprint deliverables.

## Keep/change rules for small helpers and constants

- Keep `GenerationRuntime`, executor/gatherer protocols and lazy family builders:
  they cross process/model-loading boundaries, not arbitrary line-count splits.
- Keep `ReplayModel`, `RuntimeModel`, family-local policy math helpers and
  deliberately isolated registry/task taxonomies. Do not flatten uniform sibling
  adapters when that would scatter identity, checkpoint or error handling.
- Keep schema keys, checkpoint filenames, architecture dimensions and ordered
  optimization-pass tables. Derive validation vocabularies from their declared
  types instead of duplicating ALL_CAPS key sets.
- Keep `policy_cores` separate from `trainable_modules`: Wan can execute both
  experts while training a subset. Keep module-root setters for wrapper alias
  coherence, `_lm_trunk` for real upstream nesting, and `prepare_replay` for
  scheduler/topology setup. None is dead merely because its body is short.
- Janus R1 prompt templates mixed into model workflow are candidates for a
  family-owned prompt asset when that area is migrated; neighboring vocabulary
  dimensions and structural token constants are real architecture boundaries.
  Do not add a generic prompt registry for two family templates.
- Keep per-rank memory measurements explicitly display/provenance-only.
  `_peak_memory_by_rank` is a local report reducer, not an offload policy.
- Change inherited generation lifecycle in replay, family attributes assumed
  by shared workers, duplicated role setup and test-only/dead derived fields.
- Do not merge categorical and Gaussian probability math, family conditioning,
  public facades, or meaningful lazy-import seams to reduce LOC.

## Verification and completion criteria

Use the smallest behavioral test for each real risk. No float-token fiction,
source-string bans replacing runtime tests, giant mock graphs or tolerance
relaxation. Mock downloads/expensive services, not the replay or update under test.

1. Admission: incompatible SD3.5/token objective fails before model load;
   supported token, continuous-token, multi-segment and chunk choices still build.
2. Media: real tiny outputs pass worker -> gather -> collector -> scorer for
   files-only and mixed input needs. Primary-only files, exact sample mapping,
   successful release, partial-write cleanup and retry ownership are verified.
   Include the pipelined route if enabled; do not treat a helper-only test as it.
3. Models: rollout and replay share identical policy weights and conditioning;
   replay does not construct generation-only modules or pipeline hook state.
   Exceptions are justified by actual replay reads, not class names.
4. Math: fixed trajectories retain log-probs, masks, segment selection, gradient
   direction and parameter updates. Preserve existing reference and parity limits.
5. Persistence: state-dict names/ownership, optimizer/EMA and rank RNG restore
   remain correct; fresh-process continuation uses the same tested recipe.
6. Residency: GPU sleep/wake preserves required per-module devices/dtypes;
   shared-GPU handoff verifies release and failure recovery; multi-expert and
   compile/offload combinations exercise actual hooks, not recorded call names.
   Verify warm transitions do not reconstruct the model, and cold restart cannot
   serve until the expected updated weights are installed. Measure host memory
   as well as process-attributed GPU memory; preserve Ray's safety threshold.
7. Run focused CPU tests per commit, then the relevant combined suite. Hardware
   gates remain explicitly pending until run on the necessary devices. A green
   CPU suite is not GPU capacity, throughput or quality acceptance.

Existing starting points: `tests/models/interfaces/`,
`tests/models/steps/token/test_training_capability.py`, family model-loading and
backbone-parity tests, `tests/rollouts/replay/`,
`tests/rollouts/collector/test_worker_reward_artifacts.py`,
`tests/generation/execution/test_worker_sleep.py`,
`tests/generation/execution/test_sample_batches_pipelined.py`,
`tests/trainers/online/test_state_restore.py`, trainer checkpoint/parking suites
and `tests/scripts/test_common_factory.py`.

Audit baseline: 93 CPU tests passed across replay/runtime contracts, token
training capability, multi-segment token and chunk replay. Separately, the
in-flight artifact test file had 4 passes and 1 failure because the old hook
test omitted the new required `primary` argument. Concurrent commit `5f1362c6`
updated that coverage; the final artifact-file rerun passed all 9 tests. This
does not cover the complete producer/gather/scorer and cancellation paths listed
above. The selected architecture suite recorded
7 passed, 1 failed and 2 deselected; its failure is the existing import-floor
violations in section F, not a numerical regression. These observations are snapshot-scoped; rerun
after concurrent changes settle. No GPU training or production edits were made
for this sprint document.

The native ownership sprint is complete only when the entire family inventory
is accounted for, the shared layer has no generation-family payload assumptions,
replay owns only its actual dependencies, backend-specific residency stays with
generation, and the required behavior/hardware gates above pass. Pending required
hardware gates mean incomplete, not accepted with a footnote. Optional engine
integration and full-weight validation beyond the declared representative gates
remain separately labeled; neither can be claimed from tiny-model tests. Do not
equate a new base class or fewer lines with completion.

## Miles reference and limits of the analogy

Inspected pinned source `ebd55fc1e597520322e0225997552c0807e9293b`:

- [Rollout engine release/resume](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/miles/backends/sglang_diffusion_utils/sglang_diffusion_engine.py#L297)
  delegates to SGLang-Diffusion endpoints.
- [Trainer sleep/wake](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/miles/backends/fsdp_utils/actor.py#L261)
  moves model and optimizer separately from rollout ownership.
- [Project README](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/README.md)
  describes the independent diffusion repository and engine/trainer split.

There is no same-named `uses_pipeline_cpu_offload` in that inspected tree.
This does not mean its underlying engine has no offload machinery. Adopt the
ownership principle, not an unverified claim that its memory policy can replace
VRL's mixed-device, four-card time-sharing and failure-recovery requirements.

Related plans: `docs/sprints/SPRINT_reward_transport_ownership.md`,
`docs/sprints/SPRINT_miles_diffusion_parity_program.md`,
`docs/sprints/planned/SPRINT_continuous_three_stage_pipeline_program.md`.
This sprint neither restarts their experiment queues nor makes a two-repository
split, a new serving backend, or a control/trainer process split a prerequisite.
