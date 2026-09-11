# Repository clarity audit

Status: in progress. This audit covers the whole repository, with priority on
core generation, scheduling, and external helpers. Function counts identify
inspection candidates; they do not establish that a function needs relocation.
No repository-wide completion claim is supported yet.

## Criteria

- Put construction and state-dependent operations on their existing owner.
- Remove redundant forwarding only after inspecting callers and tests.
- Preserve protocol boundaries, lazy imports, framework adapters, and useful
  cross-family consistency. Do not create classes merely to namespace functions.
- Inspect module-level ALL_CAPS data. Keep real schema, environment, file,
  protocol, architecture, fixture, or deliberately isolated taxonomy boundaries.
  Move workflow-local business vocabularies to their domain owner or config.
- Preserve numerical behavior, admission/backpressure, cancellation, isolation,
  checkpoint compatibility, and distributed error propagation.

## Completed changes

| Area | Change | Validation |
| --- | --- | --- |
| Metrics | `MetricsCSV` owns initialization, resume alignment, and append; `OnlineMetricsCSV` owns both output formats | 151 relevant tests passed |
| Denoise trajectory | `DenoiseTrajectoryBuffers.allocate` and `record_step` own allocation and writes; timestep helpers belong to that owner | 49 denoise and full/chunk binding tests passed |
| Continuous schedule | Concrete construction moved to `ContinuousRolloutSchedule.from_config` | 163 orchestration tests passed |
| Collector | Concrete construction moved to `RolloutCollector.from_family`; production, unit, and e2e call sites migrated | 353 rollout and online lifecycle tests passed; real checkpoint e2e was not executed |

The denoise refactor preserves existing partial-step behavior; this cleanup does
not establish new guarantees about partially populated trajectory buffers.

Additional execution/capacity changes:

- `AffinePeakFit.max_samples_within` now accepts the actual request ceiling.
  Removed `_FLAT_FIT_UNBOUNDED`, whose arbitrary integer pretended to represent
  infinity. Confirmation trials, OOM bisection, and throughput checks remain.
  Execution tests: 120 passed, including flat/negative slopes and large ceilings.
- `RolloutBatch.estimated_payload_bytes` owns the queue's payload estimate;
  removed `estimate_batch_bytes` from continuous scheduling types. Counts remain
  unchanged, and documentation distinguishes object deduplication from shared
  storage and allocator/RSS measurements. Continuous/collector tests: 149 passed;
  torch-free config parsing: 1 passed.

Further owner consolidation:

- Removed `_batch_output_debug_metrics`: its only production caller was the
  worker's `_batch_metrics` wrapper. Debug gating and output assembly now live
  in that existing method. Retained `_debug_metric_value` as pure recursive
  serialization and `_require_chunked_executor` as the construction-time
  protocol check. Execution tests: 121 passed, including disabled-debug access.
- Removed `infer_next_epoch`: its only production caller was the owning
  `TrainingCheckpoint.next_epoch` property. The fallback now lives there with
  unchanged payload/meta/trainer-step/directory precedence. Checkpoint tests:
  94 passed; publication and restore collectives were not reorganized.

## Inspected boundaries retained

- `run_denoise_loop`: shared numerical execution entry across model bindings;
  its replay-buffer manipulation now delegates to the buffer owner.
- `build_rollout_schedule`: selects between concrete scheduling implementations
  and wires the common runtime coordinator. It is not a single-type constructor.
- `validate_rollout_schedule_topology`: compares resolved resources with schedule
  configuration, with no single input object owning both.
- `ContinuousRolloutSchedule` forwarding methods: cross-thread handoff is the
  facade's responsibility; reducing lines would obscure execution ownership.
- `_interval_overlap_seconds`: pure interval arithmetic without collector state.
- `_csv_field`: dataclass field adapter keeping serialization metadata alongside
  the schema instead of maintaining a second column definition.
- `_DENOISE_OPTION_FIELDS` in collector config: schema-derived projection keys,
  not a manually duplicated algorithm vocabulary. Projection helpers remain
  reviewed below.
- `HEALTH_CONCURRENCY_GROUP`: Ray concurrency-group protocol name, not business
  routing data.

Further inspected execution boundaries:

- `sample_batches.py`: strict replay merge, sample coverage, and OOM splitting
  serve driver/executor/gatherer callers. These are shared algorithms and tensor
  shape rules; a helper-container class would add no ownership.
- `EnginePlan.from_request`: already owns one batch-width fallback shared by
  direct and distributed execution. Preserve that single resolution path.
- `pipeline.py`: copy-stream/event lifetimes and exception cleanup form a real
  CUDA execution boundary. Keep it separate from family-specific denoise logic.
  No CUDA implementation change was made or newly GPU-validated in this audit.
- `rank_group.py`: explicit torch.distributed init/destroy framework boundary;
  `RankGroupSpec` remains the serializable rendezvous value, not a runtime owner.
- `_await_owner_future`: asyncio/concurrent-future cancellation adapter, retained
  because shielding prevents caller cancellation from cancelling owner cleanup.
- Ready queue, generated capacity, and staleness policy: separate state and
  invariants (ready payload ownership, pre-reward reservations, version bounds).
  Do not merge them solely because they are used by one scheduling subsystem.

Binding inspection notes:

- Full-sequence and chunk-denoise gatherers deliberately remain separate from
  executors: they operate driver-side and implement the shared gather protocol.
  Shared replay concatenation and context checks remain in `sample_batches`.
- Chunk gather helpers validate consistency across several results and perform
  optional-field concatenation. They are cohesive with the gather module; moving
  all of them to static methods alone would not remove state plumbing or a
  redundant forwarding layer. No new layout/helper class is warranted by this
  inspection. Result-local shape validation merits further review alongside the
  executor payload contract before changing validation timing.
- `call_with_supported_kwargs` serves the token loop and Janus runtime as a
  signature compatibility adapter. Retain pending a separate audit of supported
  hook signatures; deleting it merely as a free function could change accepted
  family hooks.
- Checkpoint schema/version/file constants are persistence protocol boundaries;
  retain them. `DEFAULT_CHECKPOINT_STRICT` is the explicit restore protocol
  default. No business vocabulary relocation is indicated for these constants.

Config and utility review:

- Removed `_denoise_options` from collector config; the construction now lives
  on `DenoiseRequestOptions.from_sections`. Public config types are imported only
  for type checking. Collector projection, denoise, and config tests: 148 passed,
  including the torch-free config parsing check. Generic section flattening and
  duplicate-key checks remain shared schema-adapter logic.
- `HostMemorySnapshot.capture` and `__str__` replace the external capture and
  formatting helpers. `log_host_memory` remains the convenient capture-and-log
  facade; `_read_proc_field_mb` remains the operating-system table adapter.
  Memory guard, trainer flow, and online lifecycle tests: 44 passed.
- Corrected the host-memory guard documentation: the threshold measures system
  available/total memory; process RSS is diagnostic context. Behavior unchanged.
- `utils/artifacts`: hashing/path helpers serve multiple domains. Environment
  and metadata field constants are public contracts; image suffixes are a small
  deliberately isolated file taxonomy. No class wrapper is justified.
- `utils/deadline`: `require_timeout` is a shared public/direct-constructor
  validation boundary; `OperationDeadline` already owns expiry and waiting.
- Online config reflection helpers operate on varying dataclass/section types;
  preserve framework-adapter ownership rather than make them instance methods.

Reward runtime and asset review:

- Kling `_DataConfig.build_chat_payload` now owns frame sampling, pixel limits,
  and prompt selection when assembling a scoring request. Removed the external
  function's config plumbing. Kling tests: 33 passed; the optional real-processor
  decode test was excluded, and no production reward checkpoint was run.
- Moved the two image-QA default prompt templates from execution code into
  `rewards/assets/codex_image_qa_prompts.py`. AST value comparison against the
  pre-change source confirms both strings are unchanged. The public model-module
  export remains available. Image-QA tests: 27 passed without external judging.
- Removed a duplicate `HuggingFaceRepoRevision` entry in `hub.__all__`.
- Reward service `wire.py` remains a single client/server protocol codec; its
  schema-derived field lists and version envelope are real wire boundaries.
- `_build_prepared_model_in_pool` remains a separate stack frame: the caller
  clears failed construction tracebacks before closing CuMem. Inlining it could
  retain partial model tensors during cleanup.
- Kling special tokens and public score aliases represent checkpoint/output
  protocols. The score conversion helper remains consistent with other video
  reward adapters. Shared Hugging Face resolution remains a cross-model helper.
- Further image-QA output parsing/schema helpers and other reward model loaders
  still require review; the prompt extraction does not complete the reward audit.

Checkpoint identity and trajectory review:

- `LocalCheckpointContent.from_path` replaces the external content constructor.
  Trace sealing/verification and the injectable identity resolver use the class
  entry. Identity and trace tests: 63 passed, covering relocated trees, symlinks,
  rejected special files/cycles, mutation detection, and trace content checks.
- Hash traversal and stat-signature checks remain shared integrity algorithms.
  Identity metadata helpers remain dataclass/schema adapters; `_IDENTITY_KINDS`
  is derived from the public Literal, not a second business vocabulary.
- Removed `_reward_modality_for_task`, a pure pass-through to the already
  imported `task_modality`. Trajectory/binding tests: 59 passed, 2 skipped.
- Trajectory builders remain explicit adapters from different family payloads
  to the neutral trajectory schema. Do not put every regime's tensors into
  constructors on `TrajectoryBatch` merely to eliminate module functions.
- Completed the storage-policy follow-up: `TrajectoryStoragePolicy.from_config`,
  `apply_to_value`, and `apply_to_trajectory_` own parsing, tree conversion, and
  in-place trajectory mutation. Removed three public free functions, their lazy
  facade exports, and the internal application forwarder. Migrated worker,
  collector, tests, and Sana report comparison. No-op identity, integer dtype,
  conversion scope, and metadata handling remain unchanged.
- Storage/trajectory, full-sequence binding, collector, and config tests:
  117 passed (including torch-free parsing). Sana report tests: 32 passed.
  Tensor-tree traversal and byte estimates remain independent algorithms;
  derived Literal validation sets remain schema data. No new wrapper class.

Trainer diagnostic and parking identity review:

- `OnlineTrainer._precision_metadata` replaces `_trainer_precision_metadata`.
  Both callers already owned config/model/evaluator; they now read through one
  trainer method instead of unpacking the same state into an external helper.
  Online trainer tests: 142 passed.
- `TrainingMemoryState.identity_key` replaces `_training_state_key`. The key
  compares owner identities and target device, never tensor equality. Strategy
  tests: 12 passed, 2 skipped; frozen-offload and strategy-MRO tests: 23 passed.
  This is not a new real multi-GPU parking validation.
- EMA checkpoint shape validation remains a cross-type comparison. Scalar dtype
  labeling remains formatting. Strategy selection and FSDP tensor/collective
  adapters are not candidates for a blanket static-method conversion; their
  execution and distributed ownership require preserving call order.

Segment lookup and weight-sync boundary review:

- `TrajectorySegment.role_tensor` and `named_tensor` now own lookup into their
  tensor collection. Removed the standalone view helpers and facade exports;
  migrated collector, evaluators, Janus, and resolver. Resolver still owns
  cross-segment addressing and delegates role selection to the same method.
- Validation: trajectory/rollout/binding tests 385 passed, 2 skipped; Janus replay
  and R1 model tests 6 passed; torch-free config parsing 1 passed.
- `_require_installed_policy_version` remains the shared untyped Ray ACK boundary
  for direct and bucketed weight installs. `_validate_prepared_weight_snapshot`
  remains a recursive CPU/detachment guard before cross-thread handoff. Neither
  should be deleted or hidden by a helper-container class merely to reduce the
  free-function count. No weight protocol behavior changed in this pass.

Profiling ownership review:

- `TorchProfilerConfig.should_capture` owns enabled/skip/window selection;
  `ResolvedActivities.from_config` owns construction against supported Torch
  activities. Removed the two external owner-specific helpers.
- Profiling tests: 24 passed, including real CPU trace/summary/manifest output,
  unsupported activity handling, and finite/unlimited capture windows. No new
  CUDA profiling validation or output schema change.
- Retain `capture_torch_trace` and `profile_range` as context-manager framework
  boundaries. Enum discovery stays derived from Torch's enum; file discovery,
  filename sanitization, and summary/manifest I/O stay cohesive in the module.
- Logging's handler/init helpers manage a process-wide logging namespace and
  redirected stdout; `kv` is pure formatting. They should not be relocated to
  arbitrary consumer classes. Logging names/format strings are output protocol
  data rather than model-specific workflow vocabularies.

Dataset and evaluation helper review:

- `PickAPicPreferenceDataset.from_hub` replaces `load_pickapic`; the dataset
  owns its remote construction and existing preprocessing. Migrated the DPO
  entrypoint and removed the lazy export. Dataset loading semantics are unchanged.
- DPO identity/config tests: 10 passed. Isolated execution exposed existing fake
  registry import-order dependence; tests now bind/parse the real schema before
  replacing model-construction registry calls. No production dataset download
  or full DPO training was run.
- `collate_preference` remains the DataLoader adapter. Prompt manifest parsing,
  mixture sampling, and clean-latent shard I/O operate on collections or storage
  contracts, so they do not belong to an individual prompt instance.
- Evaluation `seed_for` is pure checkpoint-independent sample indexing; image
  and video generation adapters remain shared evaluator/model boundaries.
  Sampling/runtime identity types already own record parsing/serialization.

Model memory adapter and gate recheck:

- Removed `_call_required` from VAE memory configuration: it discarded `owner`
  and only invoked `getattr`. Tiling/slicing now call the target methods directly;
  removed the unused lower-level owner argument. The upper generation-memory
  adapter retains owner context for unsupported-target errors. Tests: 17 passed,
  including real VAE toggle behavior. Policy values and dispatch stay unchanged.
- Retain the VAE application module as a framework adapter, separate from the
  torch-free policy values. Shared denoise CFG/timestep/replay tensor helpers
  likewise represent cross-family numerical/layout operations, not candidates
  for a single monolithic model utility class.
- Rechecked config gates with unused `precision`: their uniform registry
  signature is deliberate, and they perform actual checks. Compile gate now
  reads `compile_conflicts` directly. Preserve uniform gate signatures rather
  than special-case dispatch merely to eliminate an unused parameter.
- Canonical dtype aliases remain isolated in `models/dtypes.py`; plain-precision
  validation and checkpoint dtype parsing have different acceptance contracts.
  Do not merge those semantics based solely on similar names.

Combined regression and export scan:

- Ran generation, rollout, trajectory, utils, and config suites together after
  the accumulated ownership migrations: 1130 passed, 2 skipped. This included
  real Ray worker failure/cleanup tests and CPU profiler artifact generation.
  It does not establish full model-checkpoint or multi-GPU training acceptance.
- Parsed 499 repository Python modules for duplicate literal `__all__` entries.
  Removed duplicate `RewardInferenceConfig` and `LogprobMismatchStats` exports.
  Mismatch algorithm tests: 24 passed. Computed export maps and broader public
  API semantics are not certified by this narrow static scan.
- Reviewed Ray launch-input serialization, driver-device discovery, and batch
  placement: existing classes own resolved settings/plans; device probing is
  a cross-object runtime adapter. Optimization pass sequencing remains an
  explicit shared pipeline, independent of family-specific policy roots.

## Remaining review

These are inspection candidates, not approved mechanical transformations.

1. Generation execution: worker/planner, sample batching and OOM retry, pipeline
   transfers, rank groups, memory sizing, bindings and gatherers, token loop.
   The flat-fit capacity sentinel is now removed; worker lifecycle and binding
   helper ownership still require further review.
2. Continuous scheduling: producer, consumer, owner shutdown, ready/generated
   capacity, staleness, weight synchronization; inspect ownership without
   collapsing independent execution or resource lifetimes.
3. Collector config projection and batch operations; distinguish schema adapters
   and tensor-tree utilities from operations on a single owning type.
4. Trainer checkpointing, online training helpers, strategy/FSDP, metrics and
   trace consumers. Checkpoint publication and distributed collectives require
   behavioral review before regrouping functions.
5. Model construction and checkpoint identity, family adapters, rewards and
   service wire code, resource resolution, config, trajectory builders, utils.
6. CLI/data/eval/performance helpers, docs, and tests. Preserve standalone entry
   points, vendor boundaries, and useful test fakes; do not rewrite historical
   completed sprint records solely for renamed APIs.

Completion requires inspecting the remaining areas, recording justified keeps,
fixing confirmed ownership problems, and running checks appropriate to each
change. Passing the tests above only verifies the listed changes.

## Explicit progress and concrete ownership follow-up

- `TrainingCheckpoint.load` owns payload construction; all Python callers and
  evaluator test patches use that API. Resume positions now require the exact
  `progress.next_epoch` / `progress.next_step` integer fields. Trainer counters,
  metadata and directory names are no longer fallback sources. Online saves now
  write `next_step` explicitly; offline DPO already does. Older checkpoints
  without the required progress fields cannot resume via these properties;
  model-only loading remains available. Invalid floats, strings and booleans
  are rejected rather than coerced. The shared private position reader owns
  the identical validation for both fields.
- `PreferenceBatch.collate` is passed directly to DataLoader. The framework
  callback remains, but its independent function and lazy export are removed.
- `TrajectorySegment.named_tensor` is removed: the sole caller reads the tensor
  dictionary directly. `role_tensor` stays because a semantic role must identify
  exactly one tensor; ordinary dictionary access cannot enforce that invariant.
- `ProfilerActivitySelection` replaces the vague `ResolvedActivities` name.
  Requested/effective activity tracking and unsupported-device failures stay.
- Trajectory storage directly checks `torch.Tensor` and uses the canonical dtype
  parser once per conversion, removing two forwarding helpers. Runtime imports
  stay lazy so parsing configuration does not import Torch. Storage placement,
  integer tensor preservation and byte accounting remain distinct operations;
  no generic utility class is introduced. The shared tree walker and recursive
  byte estimator retain real traversal/deduplication responsibilities.
- Storage's `_VALID_DEVICES` / `_VALID_DTYPES` stay as schema validation sets
  derived from Literal definitions; they contain no independent business facts.
  Checkpoint filename/schema constants likewise remain real format boundaries.

Validation for this follow-up: 752 tests passed across checkpointing, FSDP,
DPO timestep handling, trajectory, profiler, online lifecycle, evaluation,
rollouts, storage adoption and the Torch-free config parsing check. Touched
Python files pass Ruff; `git diff --check` passes. This validates the changes
above, not completion of the outstanding repository-wide audit.

## Generation memory telemetry review

- Removed `build_batch_memory_shadow`: its sole production consumer converted
  typed readings into dictionaries solely to read them back for logging. The
  executor now logs `BatchMemoryReading` directly, preserving log text and
  skipping absent readings. No intermediate list or new class is needed.
- Kept CUDA occupancy capture as a lazy runtime sampling boundary: it records
  pre-loop quantities that cannot be reconstructed from the completed batch.
  Kept `AffinePeakFit` as the existing owner of startup sizing math. Neither
  measurement nor the probe/confirmation algorithm changes.
- Rechecked `_require_chunked_executor` (plugin protocol validation), recursive
  debug value formatting (serialization boundary), and `_is_oom_error` (rank
  errors arrive as text). These short functions retain concrete responsibilities.
- Inspected coordinator policy-version fallback and its collector initialization
  dependency. No change made: removing it requires resolving version ownership
  across initial attachment and weight-sync acknowledgement, not just deleting
  the fallback. This lifecycle review remains outstanding.
- Validation: 139 execution, OOM-split and Torch-free parsing tests passed.
  The removed dictionary-construction test is replaced by executor-path log
  coverage with and without a memory reading. Ruff passes on all touched files.

## Policy-version ownership review

- Removed the coordinator's `_last_policy_version` cache and push-count fallback.
  `current_policy_version` now reads published provider state directly, returning
  `None` when neither provider reports a version. An unversioned syncer no longer
  acquires invented versions 1, 2, ... merely because pushes completed.
- The syncer remains the version allocator. Its one-caller
  `_resolve_next_policy_version` helper is folded into its constructor, retaining
  the explicit initial-version override and first-version semantics.
- Ray runtime publication means accepted target version: active sessions install
  before publication, while inactive sessions stage the target for activation.
  This cleanup does not conflate accepted state with installed worker state.
- Retained provider precedence (collector runtime, then syncer during attachment),
  lock-protected allocation, runtime publication, and continuous staleness logic.
  The collector's exception-based pre-attachment access remains a separate
  lifecycle interface issue; no broad RuntimeError handling changes are bundled.
- Validation: 230 orchestration, trainer sync, Ray weight-sync and runtime-config
  tests passed, including real Ray shared-object transfer. New coordinator tests
  cover unversioned pushes, clearing a reported version, and pre-attachment
  syncer reads. The subsequent constructor-only consolidation passed the trainer
  weight-sync suite separately. Touched-file Ruff checks pass.

## Collector attachment boundary

- `RolloutCollector.generation_runtime` now explicitly returns an optional runtime
  during setup, matching the collector's constructor state. The scheduling
  protocol declares the same optional return type.
- Removed coordinator `_collector_generation_runtime`, which caught every
  `RuntimeError` and interpreted it as missing attachment. Coordinator queries
  now read the property directly; provider failures propagate instead of being
  hidden behind a syncer fallback or a false offload requirement.
- Collector execution uses `_require_generation_runtime` for the shared
  initialization precondition before generate/activate/offload. This private
  method stays because four execution sites require the same actionable failure;
  state queries and execution preconditions are intentionally different APIs.
- No new exception/state classes, constants, scheduling policy or transport
  changes. Runtime attachment and the trainer's capability-based syncer factory
  continue to work with optional setup state.
- Validation: 356 rollout and online lifecycle tests passed; expanded direct
  collector attachment/activation test passed separately. Provider-error tests
  verify propagation through both version lookup and driver offload decisions.
  Ruff and diff whitespace checks pass for the touched files.

## Batch planning and replay alignment

- Removed planner `max(1, int(...))` coercion of explicit batch widths. The
  existing `GenerationSampleBatch.plan` entry now requires positive Python
  integers for prompt count, group size and batch width. Invalid overrides fail
  explicitly instead of being truncated or clamped into a different plan.
- Replay tensor concatenation now checks each tensor's leading dimension against
  that source batch's sample count, using the existing `_require_rows` helper.
  Ragged sample wrappers already enforced this invariant; tensors previously
  bypassed it. A short batch and an oversized batch can have the correct total
  row count while assigning replay values to the wrong samples.
- Kept shared coverage/order checks, static context comparison and replay merge
  functions. They enforce cross-binding consistency and have no natural single
  object owner; a new utility class would not simplify the contract.
- Kept per-family gather/layout boundaries: full-sequence and chunked denoise
  gatherers share replay rules while retaining their distinct payload schemas.
  No ALL_CAPS workflow vocabularies were introduced or relocated in this slice.
- Validation: 189 execution, binding, OOM-split and Torch-free config tests
  passed; two binding tests skipped. Regression cases reject mismatched per-batch
  tensor rows even when their total matches, scalar replay tensors, and invalid
  explicit widths. Touched-file Ruff and diff whitespace checks pass.

## Exact sample identity at generation boundaries

- `GenerationSampleBatch` now rejects non-integer prompt indices, sample starts
  and counts at construction. Previously fractional values passed range checks,
  and `ordered_covering_batches` silently truncated them with `int()`.
- Removed those ordering/coverage casts. Gatherers now consume the identity
  established by the batch rather than manufacturing a corrected identity.
- `GenerationRequest` enforces integer sample counts/widths and its standalone
  batch-range validator rejects non-integer arguments. These are distinct public
  boundaries: token and chunked-denoise execution call the range validator
  directly, while planned work carries a GenerationSampleBatch.
- Kept the existing request/plan/batch types, per-family layout interfaces,
  coverage checks and OOM split algorithm. No integer utility class, new module
  or workflow constant is introduced. Malformed numeric inputs now fail rather
  than changing which sample they identify.
- Validation: complete `tests/generation` and `tests/rollouts` selection passed
  (778 passed, two skipped), including live Ray timeout/cleanup coverage. Added
  cases reject fractional, boolean and string sample identity/count inputs at
  both construction and direct range-validation boundaries. Touched-file Ruff
  and whitespace checks pass.

## Data schema placement and external helper review

- Moved `SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS` from generic path utilities
  into the existing data artifact-validation module. All current consumers are
  data provenance validation or dataset derivation; no reward runtime depends
  on this schema. Retained its eight ordered keys and existing data-module
  export, and updated direct imports without a compatibility forwarding alias.
- Kept `DATA_ROOT_ENV` (environment boundary) and `IMAGE_SUFFIXES` (shared media
  extension classification) in the Torch-free utility leaf. Kept path/hash
  helpers as cross-domain operations; they have no state that would justify an
  artifact utility class.
- Reviewed token and denoise generic build modules. Retained descriptor-driven
  builders and common config projection because they remove family duplication,
  maintain lazy import boundaries, and enforce quantize/device/compile ordering.
  Their short validation helpers are shared within the build sequence, not
  arbitrary forwarding layers.
- Reviewed JSON/JSONL read/write helpers and shared atomic publication. Retained
  them as the common filesystem boundary. Corrected the module docstring:
  finally cleanup cannot promise removal after abrupt process termination, and
  exclusive publication uses a hard link rather than a rename.
- Validation: 70 data and Torch-free config tests passed; touched-file Ruff and
  whitespace checks pass. Schema order/content and atomic-write implementation
  are unchanged. This review does not establish full model-family audit coverage.

## Continuous consumer wait budget

- Consumer polling now sleeps for the smaller of the poll interval and remaining
  wait budget. A ten-second poll interval no longer delays a ten-millisecond
  consumer timeout until the full interval elapses.
- Reused `require_timeout` at the public collect boundary to reject non-finite or
  non-positive wait/poll inputs. Kept the existing TimeoutError and diagnostic
  message, including producer health counters; no new deadline wrapper added.
- Reviewed generated capacity transitions: reservation persists through reward,
  size reports replace the ceiling once, scoring does not free admission, and
  terminal cleanup releases the reservation. Retained this state owner and the
  distinct ready queue; combining them would conflate independent lifetimes.
- Retained staleness policy as the shared version-window comparison used by
  producer/consumer, and kept consumer selection separate from iteration
  materialization. The deadline fix changes neither policy selection nor the
  prompt-batch identity/coverage checks. No new ALL_CAPS data introduced.
- Validation: all 124 continuous orchestration tests passed, including real
  event-loop timeout coverage with a poll interval much longer than its budget
  and invalid-setting rejection. Touched-file Ruff and whitespace checks pass.

## Config rule and launch-gate review closure

Reviewed current `vrl/config/rules.py`, `vrl/config/algorithm.py`,
`vrl/config/validation.py`, `vrl/algorithms/config_contract.py`, and their rule,
contract and validation-tier tests. No further structural change is warranted
for these four modules in this audit:

- Algorithm SDE, step-KL, SFT and consumed-section facts are already owned by
  algorithm configs. Tests enumerate the schema's supported kinds and verify
  that changing a declaration changes validation without changing the kind.
- Keep the remaining explicit Janus-R1/NextStep pairing checks in the single
  cross-section entrypoint. The user rejected a separate FamilyTrainingContract;
  rebuilding that abstraction would violate the requested direction.
- Keep `algorithm_config_class`: this is the lazy import dispatch boundary,
  not a duplicate algorithm-fact table. Its unsupported-kind error is explicit.
- Keep `compile_conflicts` as the shared matrix queried both by launch and
  narrower runtime checks. `gate_compile_compatible` aggregates its conflicts
  into a configuration error; it is not just a forwarding alias.
- Keep the common gate signatures and `TRAINING_GATES` tuple. This is an ordered
  validation registry, a legitimate protocol/config table rather than workflow
  business vocabulary. Production reward contracts and dataset provenance stay
  with their owners, and the gate performs only orchestration.
- Keep tier separation: section shape, cross-section relationships, then launch
  checks requiring precision/runtime imports or files. Do not create per-rule
  classes, recreate deleted family contracts, or inline gates to reduce LOC.

This closes the above module slice, not schema/loading/precision or the entire
configuration package's remaining source review. The broader remaining list
should be read with these explicit module closures to avoid repeated cleanup.

Validation: all 320 `tests/config` tests passed against the current worktree,
including all experiment parsing, declared algorithm facts, launch tiers,
precision and Torch-free parsing. No production code changed in this review.

## Config loading review closure

Reviewed `vrl/config/loading.py` and its resource/composition tests. Replaced
private-attribute probing (`hasattr(entry, '_content')`) with the public
`OmegaConf.is_config` check. Defaults entries now reject numeric, boolean, null
and list values before path resolution, naming the source file and entry index;
previously these values could become incidental filenames or opaque errors.
The existing single-key mapping check and valid string/mapping semantics remain.

Retained the module's free functions: package-resource traversal, path joining,
default selection and recursive YAML composition are stateless operations.
`compose_config` deliberately preserves mandatory values for inspection, while
`load_config` resolves and validates them for execution. `_SELF_` is a YAML
composition marker and `_BUNDLED_CONFIGS` is the package-resource boundary;
neither is an unexplained business vocabulary. Cycle detection tracks the
active recursion ancestry rather than globally rejecting reused presets.
Additive layers remain independent of base default replacements, and scalar
values apply last. No ConfigLoader class or generic parser framework is added.

This closes the loading module's current clarity review alongside the already
closed rule/algorithm-dispatch/gate modules; precision and schema source review
remain outside that closure.

Validation: 324 config tests passed, including four malformed-default regression
cases and all bundled experiment/resource composition checks. Touched-file Ruff
and diff whitespace checks pass.

## Precision policy and dtype review closure

Reviewed `vrl/config/precision.py` and `vrl/models/dtypes.py` with their callers
and precision tests. Required policy dtype values now reject null or blank input
instead of inheriting the tool helper's fp32 default. Rollout's optional null
remains the declared inheritance signal; an empty string is no longer an
accidental fp32 override of bf16 training.

Retained the two parsing boundaries: `normalize_precision` supplies defaults for
optional tool arguments, while policy fields require an explicit plain token.
Torch dtype parsing accepts checkpoint aliases and materializes actual Torch
types lazily; wire naming deliberately preserves unknown future tokens. These
are different contracts, not duplicate implementations to merge.

Retained `_QUANTIZATION_FORMAT_RULES` as the isolated format/recipe vocabulary,
`_PLAIN_DTYPES` as public schema tokens, and the derived Torch alias lookup.
Quantization constructors already own validation and construction. Plain dtype
and float32-policy helpers are shared by the schema validators and immutable
runtime policies. A utility class or a second alias/config registry would add
indirection without removing complexity. This closes this two-module source
review; schema and model runtime precision execution remain separate slices.

Validation: 336 config tests passed, including null/blank policy dtype rejection,
optional rollout inheritance, all bundled experiment parsing, quantization and
Torch-free import checks. Touched-file Ruff and diff whitespace checks pass.

## Reward disk-write helper ownership

- Folded `_validate_media_shape` and `_fps` into their sole caller,
  `DiskRewardArtifactStore._write_one`. Shape checks read the store's media type
  directly, and FPS is interpreted only inside the MP4 branch. Tensor writes
  still ignore irrelevant FPS metadata. Image type/empty-payload errors now name
  image rather than incorrectly describing every payload as video.
- Retained `_artifact_provenance`: it is the explicit scalar-metadata wire
  boundary, distinct from file creation. Retained artifact store materialize,
  release and retain methods, including in-memory no-op implementations, because
  these preserve uniform ownership semantics across transport implementations.
- MediaType/ArtifactFormat remain schema Literals; no backend vocabulary or
  utility class introduced. The dtype/device transfer, hashing, file publication
  and cleanup implementation are unchanged.
- Validation: 14 artifact-store, disk-reward/default and Torch-free config tests
  passed, including tensor and MP4 output. Touched-file Ruff and whitespace
  checks pass. Reward service lifecycle and base scoring execution remain
  separate review slices.

## Reward service acknowledgement and error decoding

- Client cancellation now requires the server's explicit `cancelled` status.
  Previously any string status was treated as a successful cancellation, so
  `running` could incorrectly settle an ambiguous request and allow its shared
  artifacts to be released. Unknown statuses become transport errors and the
  existing ambiguous-request path continues to retain ownership conservatively.
  Explicit REQUEST_COMPLETED/CANCELLED error acknowledgements still count as
  terminal, as before.
- Error decoding defaults `details` only when absent. Explicit false, zero, empty
  string/list or null now fails the JSON-object requirement instead of becoming
  `{}` through a truthiness fallback.
- Kept the shared wire functions and versioned envelope helpers: both endpoints
  consume them, and inference dataclasses define the field vocabulary. Kept
  WIRE_VERSION/error codes and derived artifact-field names as real protocol
  boundaries. No wire utility class or extra cancellation state class added.
- Reviewed cancellation endpoint semantics: it waits for the request task before
  returning `cancelled`; already completed and unknown requests have distinct
  errors. Client uncertainty cannot be resolved merely from a successful HTTP
  code. This review does not close all service-owner/server execution paths.
- Validation: all 39 reward service tests passed, including client/server scoring
  and lifecycle tests plus new unknown-cancellation-status and malformed-details
  regressions. Touched-file Ruff and diff whitespace checks pass.

## Reward owner cancellation before task startup

- Fixed a completion-signalling race in `RewardScorerOwner.score_batch`.
  Cancelling the cross-thread submission could cancel its asyncio task before
  the coroutine entered its try/finally, leaving the completion event unset.
  The caller then waited indefinitely for an acknowledgement that could not run.
- Shield the submission from caller cancellation and send cancellation to the
  owner loop explicitly. Owner-local state records cancellation before startup;
  an executing task is cancelled normally. In both cases the execution wrapper
  reaches its finally and acknowledges completion. No new class or shared lock
  is needed; the cancellation helper is a real cross-thread callback boundary.
- Kept shutdown's single-owner lock/event protocol and non-cooperative work
  semantics: an in-flight synchronous model call must actually return before
  its caller can declare cancellation complete. This change does not claim GPU
  preemption or change the service's terminal artifact ownership contract.
- Validation: all 40 reward service tests passed, including a deterministic
  queued-before-start cancellation test using an actual blocked owner thread,
  and existing non-cooperative cancellation coverage. Touched-file Ruff and
  whitespace checks pass.

## Reward service construction ownership

- Replaced private free constructor `_load_service` with public
  `RewardService.from_yaml`. YAML loading, launch-policy validation, relative
  artifact-root resolution and runtime assembly now belong to the constructed
  service. Path expansion/resolution is owned by this entry, not its CLI caller.
- Migrated both consumers: the service CLI and CountGD installer's generated
  health-check script. The latter previously imported the private helper across
  modules. No forwarding alias or new loader class remains.
- Retained `_run_cli` as the signal/event-loop adapter and HTTP handler methods
  as framework boundaries. Kept fresh request execution separate from cached
  response revalidation: both settle admission but only one runs the model.
  Shared try/finally shape alone does not justify a generic executor abstraction.
- This constructor preserves standalone-service parking rejection, CPU overlap
  semantics and launch defaults. No new ALL_CAPS data introduced.
- Validation: 40 reward service tests and eight CountGD installer tests passed;
  source search finds no remaining Python `_load_service` references. Ruff on
  touched files and diff whitespace checks pass.

## Explicit captions for video reward judges

- `RewardInferenceArtifact.require_prompt_and_video_path` now reads only the
  declared `artifact.prompt`. Removed its fallback to arbitrary metadata and
  string coercion, which could turn null metadata into the literal caption
  `None` or choose a second caption source when the actual field was absent.
- Caption-conditioned judges reject missing, non-string or whitespace-only
  prompts at their shared boundary. General artifacts still permit absent
  captions for scorers that do not consume them. Existing artifact stores
  already populate the explicit prompt field.
- Retained the method as cross-family consistency: Kling, HPSv3, UnifiedReward,
  Qwen-VL and VideoCon share its prompt/path precondition. No per-judge duplicate
  validators, prompt registry or new class added. Metadata-only captions are
  intentionally no longer accepted by these judges.
- Validation: 156 reward inference/service/model/Kling tests passed, two skipped.
  New regressions cover malformed explicit captions despite available metadata
  and preserve the explicit field when metadata disagrees. Touched-file Ruff and
  whitespace checks pass. Full reward numerical/model execution is not claimed.

## Prompt sampler epoch semantics

- Sequential-window sampling no longer clamps negative epochs to zero or
  truncates fractional/string values. It uses the existing exact-integer
  boundary validator before calculating the requested window, shared by sample
  and preview through `_sample_with`.
- Retained PromptBatchSampler as the real RNG/configuration owner. Preview clones
  generator state, every rank consumes the same global draw, and rank slicing
  happens afterward. No extra sampler class, helper or strategy table added.
  Random-without-replacement continues to consume RNG rather than use epoch.
- Reviewed prompt loading/projection seams: keep config-based loader dispatch
  separate from dataset indexing, and keep PromptExample's generation_input /
  reward_metadata projections as the engine-versus-reward boundary. Full prompt
  manifest parser validation remains a separate review slice.
- Validation: 35 prompt sampler, continuous owner and Torch-free parsing tests
  passed. Added invalid sequential epoch coverage for both sample and preview,
  asserting the rejection leaves RNG state unchanged. Ruff and whitespace checks
  pass on touched files.

## Native prompt JSONL boundary

- Native JSONL parsing now requires a string prompt and object-valued metadata /
  request_overrides, with errors identifying source and physical line number.
  Removed the truthiness fallback that silently accepted false, zero, empty
  strings/lists, and dict-convertible pair lists as metadata.
- Missing/null optional object fields normalize explicitly to empty objects.
  Empty string prompts remain legal for unconditional tasks. Unknown row fields
  still merge into metadata with their established precedence.
- Kept path loading and immutable-byte parsing as distinct entrypoints: callers
  with authenticated snapshots must not reopen potentially changed files.
  Retained dataset adapters and PromptExample projections; no parser utility
  class or new schema-key table is introduced. Image-caption-specific parsing
  remains a separate source-review slice.
- Validation: 109 trainer-data, dataset/provenance and prompt-config tests passed,
  including malformed field diagnostics and preserved empty-prompt/null-mapping
  behavior. Touched-file Ruff and diff whitespace checks pass.

## Image-caption manifest input semantics

- Removed `_required_string_field`, whose name promised validation but whose
  implementation stringified numbers, booleans and containers. Its only consumer,
  ImageCaptionPromptDataset construction, now checks image/caption fields
  directly without changing valid strings or the existing missing-field errors.
- Optional metadata/request_overrides now accept objects or explicit null/absence,
  rejecting other types before they can be converted or silently discarded.
  This matches native manifest object semantics while retaining the distinct
  image/caption field names and task-type default.
- Retained dataset indexing adapters and loader selection; no new validator
  class or configurable schema vocabulary introduced. The two format-specific
  parsing loops remain distinct rather than introducing an inheritance framework
  just to share a few validation lines.
- Validation: 127 trainer-data, dataset and prompt-config tests passed, including
  malformed string/object field regressions. Touched-file Ruff and diff
  whitespace checks pass.

## User-requested removal of remaining checkpoint progress guesses

Follow-up to a9a2c4726: next_epoch/next_step loading properties already require
explicit progress fields, but publication and latest-checkpoint selection still
contained guesses. Removed both:

- Metadata mirrors only provided progress fields, retaining step/epoch units.
  next_step no longer populates next_epoch, next_epoch no longer supplies
  completed_epoch, and absent counters no longer become zero. Explicit fields
  are checked as non-negative integers. Offline checkpoints retain their actual
  completed_step/next_step fields.
- Latest-checkpoint discovery orders by the recorded global_step only. It no
  longer parses a numeric directory suffix or supplies zero for missing steps.
  Complete checkpoints without valid global_step fail explicitly instead of
  silently selecting an inferred position or restarting fresh. Equal-step copies
  use stable path order without claiming either represents a later step.
- Retained strict next_epoch/next_step accessors and their shared private reader:
  these read exactly one named field and validate it; they do not infer progress.
  Retained completeness/publication boundaries and checkpoint schema/file names.
- Validation: 303 checkpoint, evaluation, online lifecycle and DPO identity tests
  passed. Regression tests cover metadata field absence and discovery without
  explicit global_step. Existing older checkpoints that relied on missing
  progress fields no longer qualify for automatic resume by directory name.
- Supervisor regression suite also passed (59 tests). Touched-file Ruff and
  diff whitespace checks pass. SFT latent shard review remains pending after
  this user-steered checkpoint follow-up.

## SFT latent shard version and target identities

- Schema versions must be exact integers matching the supported version. Removed
  int coercion that accepted a fractional version such as 2.9 as version 2.
- Save and load require non-empty string target keys. Saving no longer stringifies
  keys, which could overwrite distinct entries such as integer 1 and string "1".
  Valid target strings retain their exact identity.
- Retained save_sft_latents/load_sft_latents as tensor file-format boundaries and
  SFT_LATENTS_SCHEMA_VERSION as the actual on-disk version. No shard manager class
  or shared validation wrapper is needed for these checks. This slice does not
  claim comprehensive tensor-shape or provenance-type validation; those are
  separate from the corrected version and key semantics.
- Validation: 16 shard, script-loading and torch-free config-import tests passed.
  Touched-file Ruff lint/format and diff whitespace checks pass.

## Continuous policy-version comparison semantics

- Removed integer coercion of policy versions in StalenessPolicy. Known versions
  must be non-negative integers; absent versions retain the existing None result.
  Fractional or string versions cannot silently become a different policy identity.
- The stale-version window is validated as an integer at configuration and direct
  settings construction. Schedule and owner forward it unchanged, so those adapters
  cannot truncate an invalid value before the policy sees it. The isolated policy
  still supports a zero window; production continuous settings require at least one.
- Retained StalenessPolicy and its small predicates: producer receipt checks and
  consumer admission share this rule, and their names express different decisions.
  No new helper/class or taxonomy constant is introduced. Other continuous capacity
  and timing settings are not covered by this version-specific change.
- Validation: 197 continuous orchestration, schedule and experiment-config tests
  passed, including malformed window rejection through the schedule factory and
  malformed version rejection through freshness predicates. Touched-file Ruff and
  diff whitespace checks pass.

## Chunk-denoise result validation ownership

- Moved result-local trajectory shape validation from the gather module's
  _validate_trainable_chunk function to
  ChunkAutoregressiveDenoiseResult.validate_trainable_trajectory. Tensor fields,
  temporal chunk counts and transition counts now have one visible owner.
- The gatherer still invokes validation at the same point after cross-result
  consistency checks. Construction timing, error text and generation-only handling
  remain unchanged; no eager validation or new wrapper class was added.
- Retained ordered-batch and concatenation helpers: they operate across multiple
  results, including all-or-none optional fields, and belong to gathering rather
  than any individual result. Retained the separate gather module as the driver
  protocol boundary, with executor types imported only for type checking. This
  closes the result-local ownership question noted in the earlier binding review;
  moving every gather helper to a static method remains a non-goal.
- Validation: binding/replay/config-import selection passed (45 tests, 2 skipped).
  The chunk binding suite then passed all 8 tests with added malformed actions,
  optional KL and finalized-latent axis regressions. Touched-file Ruff and diff
  whitespace checks pass.

## Pipeline dispatch helper review

- Removed the nested _teardown forwarding function: both callers already branch
  on copy_stream, so its CPU/CUDA decision repeated an established condition.
  CUDA branches now call _move_tree_to_cpu_async directly; CPU branches retain
  the existing result directly. No lifecycle state or synchronization changed.
- Reworded pipeline documentation to distinguish submitting a copy from waiting
  for its completion, and to identify indexed slots as the batch-order guarantee.
- Retained the tensor-tree copy helper and pipeline module: pinned storage,
  source record_stream, produce events and failure barriers manage actual resource
  lifetimes. Combining synchronous worker copies with this stream-scoped path is
  a non-goal because their synchronization contracts differ.
- Rechecked worker's _require_chunked_executor and _debug_metric_value: retained
  the former as plugin protocol validation and the latter as recursive diagnostic
  serialization. Neither benefits from a new owner class or a static-method move.
- Validation: 40 pipeline execution, Ray progress, binding equivalence and real
  CUDA tests passed, including copy completion and failure cleanup. Touched-file
  Ruff and diff whitespace checks pass.

## Token hook signature compatibility review closure

- Audited runtime-provided init arguments against Janus, NextStep, Emu3, GLM-Image
  and LlamaGen runner signatures, plus Janus R1 generate_with_refine. These are
  repository-owned interfaces and accept the arguments supplied by their callers.
- Removed call_with_supported_kwargs and inspect-based argument filtering. Token
  initialization and R1 refinement now call their declared methods directly;
  misspelled or unimplemented keywords raise instead of silently changing sampling
  behavior. This supersedes the earlier provisional keep pending signature review.
- Updated the fixed-output R1 test model to accept the actual image_sampler
  argument. Replaced the loop test's implicit tolerance of an unsupported keyword
  with explicit rejection before any token step runs.
- Retained family-specific runner signatures, ARBatchInputs projections and the
  shared token-step protocol. No universal kwargs schema, adapter class or extra
  compatibility layer was introduced. External old hooks that depended on silently
  dropped keywords must now implement the invoked interface.
- Validation: 148 token composition/binding, five model-family and torch-free
  config-import tests passed; 2 tests skipped. Touched-file Ruff and whitespace
  checks pass.

## TeaCache default ownership

- Moved threshold and warmup defaults onto TeaCacheConfig fields and removed the
  two private module constants. They were constructor defaults, not independent
  protocol/schema identifiers; the dataclass now declares them directly.
- from_sampling is a classmethod: true constructs cls(), and mappings pass only
  explicit overrides. Missing fields use the same constructor defaults. Existing
  off forms, conversion rules and runtime decisions remain unchanged.
- Retained rel_l1 as the shared runtime/offline-probe numerical formula, and
  TeaCacheState as the owner of cached predictions and accumulated step changes.
  No helper namespace or new configuration class is introduced; this is not a
  change to cache approximation policy or its drift requirements.
- Validation: 72 denoise, request-layout and experiment-config tests passed.
  Direct checks confirmed true/empty/partial mappings match constructor defaults
  and overrides. Touched-file Ruff and diff whitespace checks pass.

## Exact SDE window declarations

- DenoiseRequestOptions now requires a two-element list/tuple of integer window
  bounds. Removed indexing plus int coercion that silently ignored a third bound,
  accepted a two-character string as a range, or truncated fractional boundaries.
  The window size likewise requires a non-negative integer.
- SdeConfig uses StrictInt for range elements so YAML parsing cannot coerce values
  before the request boundary checks them. Updated the stale permissiveness comment.
- Retained the existing options owner, schedule-dependent range resolution and
  seeded window selection. List-to-tuple normalization remains a representation
  conversion without changing bounds. No standalone validator class or helper was
  added; valid window selection and implicit full-schedule ranges are unchanged.
- Validation: 378 config, denoise and layout tests passed. Regression cases cover
  extra/missing bounds and malformed integer values; direct schema checks confirm
  fractional/string/bool bounds fail before projection. Touched-file Ruff and
  diff whitespace checks pass.

## Denoise timestep storage contract

- Removed DenoiseTrajectoryBuffers._timestep_dtype, which inspected the first
  element and defaulted to float32 on errors. DiffusionSamplingStateBase declares
  timesteps as a Tensor, and the execution loop already calls Tensor-only methods.
  Allocation now checks that contract explicitly and reads the actual dtype.
- Removed the unreachable non-Tensor conversion in _expand_timestep: record_step
  already calls detach before passing the timestep. Retained batch broadcasting
  and shape diagnostics, which perform necessary replay-layout work.
- Updated two allocation-only fixtures from Python lists to Tensor schedules.
  Added invalid-schedule regressions instead of preserving a partial compatibility
  path that could never execute the denoise loop successfully.
- Retained the loop entrypoint and buffer owner: numerical step composition and
  replay tensor allocation/writes have separate responsibilities. Probe execution
  still allocates full trajectory capacity while executing bounded steps; this
  change does not alter its memory sizing or claim unexecuted entries are replay.
- Validation: 81 denoise, full-sequence binding, memory-probe and tiny real-component
  pipeline tests passed. Touched-file Ruff and diff whitespace checks pass.

## Weight-sync acknowledgement identity

- Removed int coercion from installed-policy ACK validation. Engine replies must
  be non-negative integers before comparing with the requested version; strings,
  fractional numbers and bools cannot acknowledge a different identity.
- Rank aggregation now requires matching result types as well as values. This
  prevents a secondary rank's float/bool echo being hidden by Python numeric
  equality when the primary rank returns an integer.
- Retained the shared ACK validator, rank combiner and weight manifest/chunk
  functions: they enforce protocol boundaries and bounded transport object
  lifetimes across independent senders/receivers. No namespace class introduced.
  Sender-side version conversion and staged-transfer input validation remain a
  separate review; this change makes the acknowledgement boundary explicit.
- Validation: 46 engine, weight-sync and chunk-transfer tests passed, including
  real Ray object-store dereferencing and malformed/mixed-type ACK regressions.
  Touched-file Ruff and diff whitespace checks pass.

## Weight-install input identity follow-up

- Removed policy_version coercion from runtime pending installs, worker installs,
  staging and active-weight verification. Existing require_exact_int validates
  non-negative versions at runtime/sync/worker receiver entrypoints; each layer
  then forwards the original value without reinterpretation.
- Validation occurs before model loading/installing, staged-transfer assignment
  or version publication. Invalid runtime input stays outside terminal failure
  handling, preserving the healthy runtime for a corrected request.
- Retained independent runtime, sender and worker checks because these entrypoints
  are callable independently and cross execution/process boundaries. Reused the
  existing scalar validator rather than creating a policy-version wrapper type.
  This closes the input-side follow-up recorded in the preceding ACK review.
- Validation: 119 weight-transfer, version-slot, real Ray sync, runtime sleep/wake
  and torch-free config tests passed. Regressions assert malformed versions leave
  live worker weights/current version/staging and runtime health unchanged.
  Touched-file Ruff and diff whitespace checks pass.

## Collector projection forwarding cleanup

- Removed _section_values, whose sole caller was _merge_flat_section_values.
  The merge adapter now handles absent sections and reads explicitly declared
  model_dump values directly. Duplicate ownership and nested-block filtering
  remain in the shared adapter used by rollout and sampling sections.
- GenerationRequestBuilder applies task defaults with dataclasses.replace instead
  of rebuilding every conditioning field manually. Existing input values remain
  intact without another field list to maintain when GenerationInput evolves.
- Retained schema-derived denoise field sets as real projection boundaries, the
  request builder as the config-to-request adapter, and shared merge logic. No
  new class or general-purpose serialization abstraction introduced.
- Validation: 121 request construction, family/runtime projection, video reference
  metadata and experiment-config tests passed. Touched-file Ruff and diff
  whitespace checks pass.

## Collector batch dispatch clarity

- Inlined _trainable_segments and _is_multisegment_categorical into their sole
  caller, build. The complete dispatch decision is now visible together, and
  primary trainable segment selection appears once before the branches.
- Retained primary-segment validation, reward-view selection and reference
  resolution as diagnostic boundaries. Retained group-ID construction shared by
  AR/diffusion packing; the two packers retain their different reward adjustment
  and device semantics. No new dispatcher class or distribution taxonomy added.
- Validation: 61 collector, Janus multisegment, chunk-denoise binding and trajectory
  tests passed. Touched-file Ruff and diff whitespace checks pass. This is a
  dispatch-structure cleanup, not a change to which trajectories are trainable.

## JSON, logging and deadline helper review closure

- Re-read JSON file publication, process logger initialization and monotonic
  deadline implementations alongside their tests. No production restructure is
  warranted: these are shared filesystem/framework/runtime boundaries rather than
  operations belonging to a particular trainer or generation model.
- Keep _write_atomically: JSON and JSONL writers share publication and temporary
  cleanup. Keep emit callbacks as the serializer-to-file boundary; keep read_json
  as the uniform UTF-8 public facade. Atomic publication does not claim durability
  of the destination directory after power loss.
- Keep the logging handler/init pair and kv formatter. Logger namespace/format
  constants specify logging output, not a domain business vocabulary. Existing
  tests verify repeated initialization and stdout redirection; they do not prove
  concurrent first-use initialization is thread-safe.
- Keep OperationDeadline, OperationTimeout and require_timeout: Ray subclasses
  the error/deadline contract, and reward uses the transport-neutral version.
  Expiry is monotonic and remaining wait budget cannot become negative.
- Change: close this narrowly identified structural review in the audit record.
  Non-goals: utility namespace classes, merging filesystem ownership into callers,
  or claiming the remaining CUDA/media/lifecycle utility review is complete.
- Validation: 15 JSON, logger and Ray deadline tests passed. No production edits;
  diff whitespace check passes. Artifact/config helpers were also rechecked for
  cross-domain callers, with no additional structural change selected here.

## Media singleton-batch normalization

- Moved optional leading singleton-batch handling before the Tensor/NumPy branch
  in image_to_uint8_hwc. The documented NumPy [1,...] input previously failed
  despite identical Tensor inputs being supported. Both now share the same
  one-image check; multiple-image batches fail explicitly.
- Retained tensor-specific dtype/device conversion and existing channel/range
  conventions. Retained free media converters as cross-domain numerical/file
  boundaries; no ImageConverter class or new helper introduced. Resolving all
  ambiguous CHW/HWC layouts or replacing automatic value-range handling is outside
  this correction and remains a separate API-contract question.
- Validation: 348 media/reward tests passed, 5 skipped. Added pixel-exact NumPy
  CHW/HWC singleton regressions and consistent multi-image rejection checks.
  Touched-file Ruff and diff whitespace checks pass.

## Reference-image loading owner

- Removed load_reference_image from utils.media and its export. Its sole production
  caller was ReferenceConditionedBatches._reference_image_for_chunk; that existing
  owner now loads the selected prompt's image directly instead of forwarding to
  an otherwise unused utility facade.
- String and Path inputs load to RGB through a context-managed Image.open. The
  source handle closes before returning the converted image. Previously Path
  inputs silently passed through without being loaded. Non-path loaded values
  retain the established pass-through behavior.
- Retained ReferenceConditionedBatches as the shared family hook implementation
  and left encode/prepare hooks separate: model conditioning stages consume
  different payloads. No new image loader class or cross-call cache introduced.
- Validation: 51 media, full-sequence binding, tiny pipeline and torch-free config
  tests passed. Reference selection now uses actual files instead of mocking the
  deleted helper and covers both path representations. Touched-file Ruff and
  diff whitespace checks pass.

## Clean-target source identity

- CleanTargetRef.from_source now requires target values to be strings instead of
  stringifying arbitrary metadata. Both encoder PromptExample input and trainer
  reward metadata use this constructor, so invalid target types fail at the same
  identity boundary rather than later as missing files or shard keys.
- Retained existing whitespace trimming, absent/empty-field handling and exactly
  one target requirement. Retained the constructor on the identity dataclass;
  no extra validation function or class introduced. This complements the earlier
  strict shard-key save/load checks without changing valid target naming.
- Validation: 40 shard, encode-target, script-loader and trainer regularizer tests
  passed. Regressions exercise both source representations and retained empty /
  whitespace behavior. Touched-file Ruff and diff whitespace checks pass.

## Rank-local CUDA mask interpretation

- CUDA visibility narrowing validates the complete ordinal list before selecting
  a rank. Empty entries are no longer silently removed, negative/non-ordinal
  tokens are rejected, and duplicate ordinals are compared numerically so 0 and
  00 cannot assign two local ranks to the same device.
- Removed redundant selected-device parsing after validated selection. Failure
  leaves the supplied environment unchanged; valid selection publishes one
  canonical ordinal. This path continues to support integer ordinals only.
- Retained the standalone launch-environment function/module as a pre-import
  boundary: CUDA visibility must be set before trainer/Torch/Ray initialization.
  CLI/report filename and schema constants remain real external boundaries;
  no broad relocation of evaluation fixtures or standalone probe scripts made.
- Validation: 16 online entrypoint and torch-free config tests passed, including
  malformed-mask/no-mutation regressions and existing rank-to-device mappings.
  Touched-file Ruff and diff whitespace checks pass.

## Common recipe factory structural review

- Retained build_algorithm_and_evaluator as a composition boundary: it combines
  independently owned algorithm, family replay semantics, scheduler and precision
  settings. Moving it onto the result pair would only namespace the same cross-type
  work, while moving it onto one algorithm would invert family/evaluator ownership.
- Retained reward function/runtime constructors shared by online training and
  reward preflight, including the lazy implementation imports. Retained the
  cross-type memory-parking guard that validates without loading reward models.
- The local diffusion-objective branch set selects actual constructors/evaluators;
  it is dispatch implementation, not a new config fact table. No generic factory
  registry or extra contract class introduced solely to eliminate branches.
- Corrected stale comments claiming four algorithms and referring to the old
  sampling.return_prev_sample_mean location; the field is rollout-owned.
- Validation: 52 common-factory and online lifecycle tests passed, covering grouped
  replay exclusions, algorithm/evaluator construction and recipe cleanup. Touched-
  file Ruff and diff whitespace checks pass. No runtime behavior changed.

## Continuous capacity integer accounting

- Ready queue admission requires non-negative integer item bytes. Negative
  receipts can no longer reduce total occupancy; fractional/string/bool inputs
  are rejected rather than truncated. Queue limits/resizing use exact integers.
- Generated capacity validates limits, reservations and reported bytes with the
  existing scalar validator. NaN cannot poison the byte accumulator and bypass
  subsequent capacity comparisons. Invalid reports fail before reservation state
  changes, so normal cancellation/release remains usable.
- Retained separate ready payload storage and pre-reward capacity accounting;
  their lifetimes differ. No new helper class or shared container hierarchy.
  This boundary check does not claim callers may mutate resident receipt sizes;
  queue items remain producer-owned receipts whose size is fixed after admission.
- Validation: 152 continuous orchestration tests passed. Regressions assert failed
  admission/reporting preserve occupancy and allow normal reservation release.
  Touched-file Ruff and diff whitespace checks pass.

## Continuous capacity config projection

- ContinuousRolloutConfig validates integer capacity/count fields without int
  coercion, including the unscored byte budget and fail-fast count. The previous
  checks tested converted values while storing the original values, allowing
  later projection to silently change the declared limits.
- Schedule projection and owner ready-byte calculation no longer cast those
  fields. MB-to-byte multiplication remains the actual unit conversion; the
  generated-group ceiling must still fit the total unscored budget.
- Retained config ownership and independent container admission checks. No new
  field/default taxonomy or additional settings class; timeout semantics are
  unchanged by this count-specific follow-up.
- Validation: 506 continuous orchestration and config tests passed, including
  malformed count tests across all six affected fields. Touched-file Ruff and
  diff whitespace checks pass.

## Continuous wait configuration boundary

- Replaced comparison-only timeout checks with the existing require_timeout
  validator. ContinuousRolloutConfig now rejects NaN/infinity as well as zero or
  negative waits and stores the validated float before producer construction.
- Removed duplicate float conversions from schedule projection. Consumer-owned
  direct-call validation remains because callers can invoke that boundary without
  config construction; producer/drain lifecycle and configured defaults are unchanged.
- No new timeout helper, setting or scheduler class. This unifies the declared
  configuration contract with the existing finite-positive wait semantics.
- Validation: 516 continuous orchestration/config tests passed, including both
  wait fields with all non-finite/non-positive cases. Touched-file Ruff and diff
  whitespace checks pass.

## Preference dataset loading contract

- Removed the unused streaming argument from PickAPicPreferenceDataset.from_hub.
  The returned Dataset requires column access, indexing and len; handing an
  IterableDataset directly to it was not a supported execution path. A bounded
  max_samples request still streams its prefix and materializes indexable rows.
- Forwarded cache_dir in the bounded branch, which previously ignored the caller's
  cache location. Full loading and bounded loading now honor the same setting.
- Retained PreferenceBatch.collate on its batch owner and the Dataset/DataLoader
  adapter shape. No loader class, new mode or compatibility alias introduced.
  Earlier audit wording referring to a free collate_preference is historical;
  current source already uses the classmethod at the DPO DataLoader callsite.
- Validation: 5 preference-loader and DPO checkpoint-entrypoint tests passed.
  The environment lacks the Hugging Face datasets implementation, so loader tests
  use an explicit module-boundary fake for download/materialization and exercise
  the real preference wrapper. No network dataset load was performed. Touched-file
  Ruff and diff whitespace checks pass.

## Offline DPO caption alignment

- Replaced output-length-based text repetition in OfflineDPOTrainer.step with
  explicit winner-block/loser-block duplication. Previously two captions became
  [A, A, B, B] while images were [winner A, winner B, loser A, loser B].
- Require one caption per pair, pixel encoding to preserve 2B samples, and text
  encoding to return B embeddings. Updated the encoder contract documentation;
  the production Wan adapter already returns one embedding per input caption.
- Retained PreferenceBatch stacking methods and family encoder adapters: they own
  the layout and model-specific conversion respectively. No new helper, class,
  configuration vocabulary, or changes to loss/noise scheduling are needed.
- Validation: 20 trainer, preference-loading and Wan entrypoint tests passed.
  The new two-pair regression checks both policy and reference forward inputs,
  and rejects three count mismatches before forward or optimizer progress.
  CPU tests use an identity noise adapter to expose the image ordering directly;
  no full Wan GPU training was run. Touched-file Ruff checks pass.

## Token scheduler row identity

- Removed int conversion from the token envelope and ARCacheRows index handling.
  Fractional values, booleans and numeric strings no longer silently select or
  overwrite another sample's cache row. Cache operations validate the complete
  index list before mutation; TokenStepBatch also validates integer row IDs and
  token positions for direct protocol callers.
- Retained cache gather/scatter and split/concat helpers: they implement shared
  tensor/container/Hugging Face cache layouts. Kept the model-facing protocol
  separate from scheduling, preserving one-way family dependencies. No new
  class, setting or taxonomy; cache representation redesign is outside this fix.
- Verified the production ARCacheRows owner is the token envelope, whose loop
  supplies integer ranges. Validation: 42 protocol/composition/cache tests and
  16 scheduler-batching, GLM schedule and decoder-contract tests passed. New
  regressions reject non-integer inputs and verify failed writes preserve all
  cache rows. Ruff on touched files and git diff --check pass.

## Generation launch and session version identity

- GenerationRuntimeLaunchContract now validates an optional non-negative integer
  policy_version with the existing require_exact_int boundary helper. Previously
  int() silently changed fractional, boolean and string versions before the
  worker adopted them as its initial identity. None retains its existing meaning.
- RayGenerationSession forwards the version unchanged so the weight-sync owner's
  existing validation cannot be bypassed by conversion in the resource owner.
- Kept the launch contract as a serializable process boundary and weight-sync
  Protocol as the implementation seam. No new validation class or duplicate
  session guard; parking/release timeout constants and lifecycle are unchanged.
- Validation: 80 runtime-config/session/weight-sync tests passed on the initial
  run, including existing real-Ray cases. Four new session regressions initially
  failed because the test omitted the engine helper's ID argument; after fixing
  that fixture construction, all four passed. They exercise the real sync owner
  and assert invalid versions never reach the local worker. Launch tests cover
  None, zero, positive, negative, boolean, fractional and string versions.
  Touched-file Ruff checks and git diff --check pass.

## AR sampling geometry contract

- ARRequestLayout now requires positive integer image_token_num, image_size and
  max_text_length, using require_exact_int instead of silently truncating or
  parsing values. Explicit family defaults remain supported; explicit null does
  not fall back to a default. Seed must be an integer when supplied; negative
  integer seeds retain their existing torch-compatible meaning.
- Removed the redundant samples_per_prompt conversion in the seed-offset formula;
  GenerationRequest owns that field's integer contract.
- Retained _sampling_int because three geometry fields share its required/default
  semantics. Retained right_pad and the tokenizer adapter used by Janus, NextStep
  and LlamaGen; their shared shape keeps family tokenizer handling consistent.
  No new dataclass, configuration keys or module split, and no redesign of
  generation defaults or RNG sequencing.
- Validation: 64 NextStep parsing, LlamaGen construction and AR scheduler-batching
  tests passed, including invalid dimensions, explicit nulls, seed types and
  preserved defaults. Touched-file Ruff and git diff --check pass.

## Profiler local helper consolidation and trace ownership

- Inlined the sole-use activity enum query into ProfilerActivitySelection.from_config
  and the sole-use output-directory projection into capture_torch_trace. Both
  remain lazy with respect to Torch and continue to derive activity vocabulary
  from Torch's enum, without a local backend table.
- Fixed trace discovery to require the filename separator after the full worker
  name. A step1 prefix previously also matched step10, contaminating the current
  manifest with another step's trace.
- Retained filename sanitization, summary/table error handling, manifest writing
  and context-manager APIs: these implement distinct formatting, diagnostics or
  framework boundaries. No profiler wrapper class or output schema introduced.
  This does not give repeated captures of the same worker and step unique IDs.
- Validation: all 25 profiling tests passed, including an actual CPU capture of
  step10 followed by step1 that verifies the latter manifest includes only step1
  files. No CUDA capture was added. Touched-file Ruff and git diff --check pass.

## Offline optimizer construction ownership

- Inlined the single-consumer _build_optimizer into OfflineDPOTrainer.__init__.
  Parameter selection, the empty-trainable guard and optimizer construction now
  form one initialization block. Removed the unused Iterable import and duplicate
  list materialization. No optimizer wrapper or shared factory was introduced.
- Preserved standard torch.optim.AdamW, optional lazy Adafactor import, all
  optimizer arguments, reference freezing and checkpoint/step behavior. Online
  optimizer integration is not being forced into the offline constructor shape.
- Changed the former private-helper test to construct an actual trainer; it now
  checks both AdamW options and Adafactor's fixed-learning-rate settings.
  Validation: 28 trainer, offline config builder and Wan checkpoint-entrypoint
  tests passed. Touched-file Ruff and git diff --check pass.
- Also re-read EnginePlan and DistributedExecutionPlanner: request batch-width
  resolution and fleet placement have separate owners and remain distinct.
  Host-memory proc parsing and capture/logging remain shared OS and logging
  adapters, consistent with the earlier ownership review.

## Trajectory resolver error flow

- Removed _fail, which only raised TrajectoryResolverError. Error branches now
  raise directly, making termination visible at the check and avoiding an extra
  forwarding frame. Wrapped slicing errors explicitly preserve their cause.
- Kept tensor/role accessors as the resolver API, reference parsing and recursive
  slicing as layout logic, and map_tensor_tree/byte counting as shared algorithms.
  Storage-policy Literal-derived sets are schema boundaries, not duplicated
  business vocabulary. No storage dtype, conversion, indexing or schema changes.
- Validation: 43 trajectory/replay tests plus 24 trajectory-granularity and
  Janus/Emu3/GLM replay tests passed. Touched-file Ruff and git diff --check pass.

## Trajectory device conversion failure contract

- Removed move_value_to_device's local _move helper and its TypeError fallback.
  The shared walker now invokes leaf.to(device) directly. An invalid device or
  broken tensor-like implementation no longer silently returns the unmoved leaf
  as though conversion succeeded; original exceptions propagate unchanged.
- Preserved device=None identity, metadata passthrough, duck-typed tensor support
  and the shared tree traversal used by resolver, SDE replay and signal assembly.
  No new tensor wrapper or conversion policy, and no changes to storage dtype
  rules or schema constants. Tree moves do not promise rollback of custom leaf
  side effects when a later leaf fails.
- Validation: 55 trajectory, replay and trainer granularity tests passed. New
  tests cover a real Tensor receiving an invalid device, original TypeError and
  RuntimeError propagation, and the no-device no-op. Touched-file Ruff and
  git diff --check pass.

## Trajectory selection uses one row interpretation

- Normalize the selector once at select_trajectory_batch, then use the same
  positions for sample identities, tensor leaves and list/tuple context values.
  Python boolean masks previously became integer row IDs for identities/context
  while tensor indexing treated them as masks, producing incompatible selections.
- Reject mixed boolean/integer, fractional/string and multidimensional selectors;
  boolean masks must match the sample count. Preserve integer ordering and
  Python-style negative indices. Tensor selectors are copied to CPU once rather
  than reparsed for every list-valued field.
- Retained the shared reconstruction and value-selection functions: both carry
  structural rules across trajectory consumers. No selector class, schema field
  or dtype policy introduced. Context's existing sample-alignment convention and
  non-sample metadata handling are unchanged.
- Validation: 65 trajectory, replay and trainer-granularity tests passed. New
  regressions compare sample IDs, actual tensor contents and list/tuple metadata
  for Python/Torch masks and index arrays, and reject malformed selectors.
  Touched-file Ruff and git diff --check pass.

## Trajectory reconstruction preserves its source schema

- Replaced manual reconstruction of tensor, segment and batch dataclasses with
  dataclasses.replace. Removed request_id/family/task parameters that both callers
  copied directly from the same source batch. The operation specifies only the
  fields it changes, avoiding a second field inventory that could omit additions.
- Kept explicit copies of mutable segment metadata/replay maps and reward-view
  maps, and retained validation of the rebuilt batch. Selection and movement
  remain shared operations rather than methods that import Torch into the schema.
- Reviewed the outer RolloutBatch selection contract: context is shared metadata
  and intentionally retained; extras select sample-aligned tensor leaves. No new
  blanket slicing of arbitrary metadata, wrapper class or schema constants.
- Validation: 51 trajectory, online reward-update and trajectory-granularity tests
  passed. Touched-file Ruff and git diff --check pass.

## Reward HTTP timeout boundary consistency

- RewardInferenceConfig and HttpRewardScorer now share require_timeout rather
  than implementing different checks. Direct client construction previously
  accepted NaN/infinite timeout values that config construction rejected.
  Removed the client's redundant float conversion after normalization.
- Kept service wire serialization/deserialization functions as transport adapters
  and WIRE_VERSION as protocol data. No new timeout setting, wrapper class or
  change to cancellation/identity semantics. The shared deadline module remains
  dependency-light and does not import Torch.
- Validation: 50 service tests and 47 reward/config-loading tests passed, including
  both constructors with NaN, positive/negative infinity, zero and negative values.
  Touched-file Ruff and git diff --check pass.

## Reward artifact integrity schema

- RewardInferenceArtifact validates size_bytes as a non-negative integer through
  require_exact_int. Its previous int(value) check accepted booleans, numeric
  strings and fractional sizes, including negative fractions truncated to zero.
  Both direct construction and wire decoding now use the same schema rule.
- Kept wire envelope/request/result/info/status encode/decode functions as paired
  protocol adapters. The artifact field set derives from dataclass schema and
  WIRE_VERSION is a compatibility boundary. Server-side actual file size/hash
  verification remains separate from structural input validation.
- No transport class, extra schema field or new integrity protocol. Validation:
  89 inference/service tests passed, one skipped. New malformed wire-size tests
  initially lacked their exception import; after correction the complete group
  was rerun successfully. Touched-file Ruff and git diff --check pass.

## WD tagger input ownership

- Inlined the sole-use _artifact_image into WDTaggerRewardModel.score_batch,
  keeping the PIL fast path and existing decode/frame-selection behavior.
  No new adapter object or media-loading mode introduced.
- _wanted_tags now requires string elements instead of manufacturing tag names
  with str(). Case normalization, whitespace handling, deduplication, threshold
  and recall semantics remain unchanged. Invalid tags fail before inference.
- Kept WD14_INPUT_SIZE as the checkpoint architecture dimension and category ID
  as selected_tags.csv protocol data. prepare_wd14_input remains a public,
  independently verifiable preprocessing boundary (white padding/BGR/raw scale).
- Validation: all 12 WD tagger tests passed, covering injected batch scoring,
  preprocessing pixels and invalid tag types without tagger execution. No ONNX
  model download or inference performed. Ruff and git diff --check pass.

## Robotics reward optional-section semantics

- Replaced truthiness fallbacks in RoboticsRewardWeights.from_mapping and
  _child_config with explicit None handling and Mapping validation. False, zero,
  empty strings/lists and pair sequences no longer masquerade as missing config.
- Kept _child_config because three child configurations share its copy/validation
  rule. Weight names remain dataclass-derived; the thin reward-function binding
  remains a registry/factory protocol adapter consistent with other rewards.
  No new class/table, changes to weights, device inheritance or model selection.
- Validation: 25 robotics tests passed. New cases verify all malformed sections
  fail before child construction and None/empty mappings retain defaults. Child
  models are test doubles; no real video model was loaded. Touched-file Ruff and
  git diff --check pass.

## UnifiedReward video reader lifetime and defaults

- Put VideoCapture release in _sample_frames' finally block. Metadata reads,
  frame reads and RGB/PIL conversion failures now release the capture as normal
  completion does, while propagating the original error.
- Moved the single-use num_frames/max_new_tokens defaults into their constructor
  lookups. Kept the checkpoint identifier as model identity, prompt/range grammar
  in the named asset module, and parsing/rubric/media functions as format adapters.
  No new reader class, frame algorithm, prompt, score or config key change.
- Validation: 13 UnifiedReward tests passed. Added capture-boundary tests for
  successful RGB extraction and three decoder/conversion failure points; no
  real model loading or video decoder integration run. Ruff and diff checks pass.

## Shared media resource lifetime follow-through

- Followed the UnifiedReward capture fix through OpenCV/imageio/PIL open sites
  in rewards, generation, trajectory and shared media utilities. The remaining
  shared read_video_frames closes its imageio reader in finally; write_mp4 owns
  its writer with a context manager; read_image_as_frames, generation reference
  loading and Codex image-QA reference loading close PIL sources after RGB copy.
- No implementation change justified at these sites. Keep shared media functions
  as I/O/layout boundaries and keep reference loading on its consuming owner.
  No generic resource wrapper class or flattening of meaningful adapters.
- Validation: 8 existing pixel-conversion tests passed. Separately performed real
  PNG and MP4 write/read round trips in a TemporaryDirectory: RGB dimensions,
  two-frame order and dark/light pixel ranges matched; artifacts were removed.
  These successful round trips do not fault-inject decoder failures; exception
  cleanup evidence here is the inspected finally/context-manager structure.
- Non-goal: changing the existing ambiguous 4-D reward tensor layout compatibility
  convention in pil_frames_from_media. That convention requires caller/layout
  review before any migration and is not resolved by resource-lifetime checks.

## Reward video tensor layout ambiguity removed

- Traced pil_frames_from_media through its Aesthetic/PickScore/AnimeReward users,
  TorchRewardModel artifact handoff and collector reward-output selection. The
  generated video contract is CTHW (BCTHW for a batch). The helper additionally
  guessed TCHW only when dimension zero was not 1/3/4, making frame counts alter
  the meaning of the same caller's layout.
- Removed that guess. Torch videos always follow the declared channel-first
  contract; external TCHW callers must permute explicitly. Migrated the AnimeReward
  frame-window and empty-video fixtures to CTHW; empty video tensors raise clearly.
- Kept the shared conversion boundary, PIL frame lists and NumPy HWC/THWC paths.
  No layout flag or wrapper class. This resolves the specific deferred 4-D tensor
  compatibility branch, not every image-layout/range heuristic in media utilities.
- Validation: 24 shared-layout and Aesthetic/PickScore/AnimeReward tests passed,
  including pixel/frame-order checks at 1, 3, 4 and 10 frames and rejection of
  unambiguous TCHW input. Touched-file Ruff and git diff --check pass.

## Combined validation after media and trajectory cleanup

Validated the current worktree at 7ea80b2b0 across affected producers/consumers,
not merely the files edited in individual slices:

- `.venv/bin/pytest -q tests/rewards tests/trajectory tests/generation/bindings tests/generation/composition tests/generation/steps`
  completed: 527 passed, 7 skipped (4.99s).
- `.venv/bin/pytest -q tests/rollouts tests/trainers tests/generation/execution tests/config -ra`
  completed: 1,392 passed, 7 skipped (61.81s). Four skips require bitsandbytes;
  three require explicit distributed-test enablement.

Total: 1,919 passed, 14 skipped. This supports compatibility of the checked
reward/trajectory changes with scheduling, worker execution, trainer consumption
and configuration. It does not establish full-model quality, all optional
backends, multi-node training or repository-wide architectural completion.
No implementation changes were needed in this validation pass. The earlier
recorded keeps remain intentional protocol/framework/shared-algorithm boundaries,
not a mandate to eliminate every free function. Remaining model-family and script
ownership coverage still needs source-level review; passing tests alone cannot
close that architectural scope.

## Wan DPO encoder boundary readability

- Added explicit device/dtype and callable return annotations to _build_encoders,
  and documented the 2B image / B caption contract at the adapter construction
  site. This complements OfflineDPOTrainer's explicit block duplication.
- Kept the two closures because they share a loaded pipeline and precomputed VAE
  statistics. Kept wan_forward as the family-specific ForwardFn adapter and
  _required as the recipe's repeated config guard. No encoder container class,
  function relocation, sampling defaults or numerical changes.
- Validation: 13 encoder, genuine tiny-Wan forward, config and checkpoint-entry
  tests passed. An isolated Python import confirmed the recipe still leaves
  torch absent from sys.modules. Touched-file Ruff and diff checks pass.

## PickScore invalid media is not a reward value

- Removed PickScoreRewardModel.score_media's TypeError-to-zero fallback. Shared
  media parsing errors now propagate, matching the error behavior of the other
  consumers instead of manufacturing a successful low reward for malformed input.
- Kept valid-image scoring, middle-frame video selection and normalization;
  retained the shared media converter and registry binding as real boundaries.
  No error wrapper, new score key or changes to model inference arithmetic.
- Validation: 36 CLIP/media/AnimeReward/in-process runtime tests passed, one
  skipped. Existing tiny real-CLIP tests still compare valid scores with an
  independent oracle; the invalid-media assertion now requires the parser error.
  Touched-file Ruff and git diff --check pass.

## Empty reward media is rejected before model-specific consumption

- pil_frames_from_media now rejects zero-size PIL images/frame lists, empty
  NumPy media and empty Torch media, including the image branch that previously
  preceded the tensor emptiness check. Its successful result supplies real frames
  for every sample, avoiding PickScore indexing failures or empty Aesthetic input.
- Removed AnimeReward's now-redundant empty-frame check. Kept the shared converter
  and existing type/layout dispatch; no validation class, new config or numerical
  scoring changes. Unsupported types still fail rather than become empty results.
- Validation: 33 media-layout/CLIP/AnimeReward tests passed. Nine new empty-input
  cases cover PIL, sequences, arrays and image/video/batched tensors; valid score
  oracle tests remain green. Touched-file Ruff and git diff --check pass.

## Diffusion sampling matches explicit integer semantics

- DiffusionRequestLayout now uses require_exact_int for request geometry, step
  count, optional fps/text length and seed. Fractional/string/boolean values no
  longer get converted before entering model requests and SDE window resolution.
  Geometry/schedule lengths are positive; integer seeds retain existing semantics.
- Kept executor-provided defaults, the existing num_frames/frame_count precedence,
  optional fields and family-specific parse overrides. No new config dataclass,
  helper table or changes to guidance/SDE mathematics. DenoiseRequest direct
  construction is a separate boundary and is not claimed validated by this change.
- Validation: 120 full-sequence binding/composition/step tests passed, including
  33 new invalid-input cases and existing request-window tests. Touched-file Ruff
  and git diff --check pass.

## Denoise request owns geometry validation

- Moved geometry/step/fps/seed integer validation from DiffusionRequestLayout to
  the existing DenoiseRequest.__post_init__. Eval and probe scripts construct that
  request directly, so the parser alone was not the owner of the contract.
  The parser now constructs it before resolving the SDE window and retains only
  text-length validation, which is not a DenoiseRequest field.
- Defaults and field mapping stay with the executor/parser; no new contract class
  or duplicated geometry guard. Direct construction gets the same rules without
  changing valid numerical execution. Payload-module import remains Torch-free.
- Validation: the expanded generation binding/composition/steps, model-family and
  script run reached 875 passed/3 skipped before stopping on a pre-existing stale
  online checkpoint assertion. Inspected save/resume code and confirmed next_step
  is explicitly required; updated the assertion in a separate test-only commit.
  The entire script suite then passed (559 tests). Counts overlap and are not an
  additive total. All generation/model-family tests preceding scripts had passed.
  Added 25 direct-constructor geometry cases; touched-file Ruff/diff checks pass.

## Direct diffusion request construction

- Replaced the fixed-key model_request_kwargs dictionary plus conditional writes
  with one explicit DenoiseRequest construction. Field mapping is visible where
  the request is created, with no new constructor helper or intermediate type.
- Preserved absent/null negative-prompt normalization, optional fps/seed and
  frame-count precedence. Kept the SDE parameter owner and family parse override
  surface; this changes neither numerical behavior nor configuration defaults.
- Validation: 109 full-sequence binding and Echo flow-policy tests passed.
  Touched-file Ruff and git diff --check pass.

## Generic diffusion defaults do not bypass request validation

- Removed int coercion of num_frames/fps/max_sequence_length from the generic
  executor constructor. Original default values reach the existing parser/request
  owner, so malformed values cannot be normalized into accepted integers first.
- Kept validation on consumption rather than adding duplicate constructor guards;
  request overrides and optional defaults retain their precedence. Kept the generic
  executor as the shared implementation replacing behavior-free family subclasses.
- Validation: 118 full-sequence/Echo tests passed. Nine new tests follow malformed
  defaults through the actual executor parser and require the existing error.
  Touched-file Ruff and git diff --check pass.

## Thin-function candidate rescan and deliberate keeps

Rescanned top-level Python functions with AST, selecting a single return/raise
statement after an optional docstring. This is candidate discovery, not a quality
metric or proof that every returned candidate has been reviewed.

Inspected current implementations and callers; retain:

- generation/ray/executor._is_oom_error: text classification at the rank-result
  boundary. It is distinct from local exception-type checks and has CUDA/HIP
  and non-OOM tests. Do not collapse transport and local error semantics.
- Cosmos/Anima script _resolve_sampling adapters: translate CLI flag names into
  shared configuration projections. Their tests verify override semantics;
  these are framework adapters rather than duplicate sampling default owners.
- token paged-attention prefill and mask helpers: typed backend prefill input and
  conditional/unconditional mask extension shared through PagedCFGTokenRunner.
- MAGI source-relative/optional-path helpers: reused for checkpoint, T5 and VAE
  paths. Source normalization and optional-value handling are distinct operations.
- Task semantic lookups: named access to an isolated family/task taxonomy, not
  business vocabulary mixed into runtime control flow.

No runtime change or new container class justified in this pass. Tests for Ray
OOM splitting, Cosmos/Anima CLI adaptation and MAGI integration: 67 passed.
The remaining AST candidates still require source/caller review; this subsection
is not a repository-wide completion claim or a mandate to inline short functions.

## Trainer metric assembly and continuous-owner boundary review

- Inlined the single-use `_mean_reward_components` into OnlineTrainer's metric
  assembly. The comprehension retains averaging and empty-component omission;
  a standalone function added a navigation step without owning a boundary.
- Kept `_rollout_reward_components`: it validates sample alignment and converts
  rollout component payloads before filtering, rather than merely calculating
  a metric. This is a distinct representation boundary.
- Inspected continuous owner submission/shutdown and capacity accounting. Kept
  `_await_owner_future`, shared by three production operations, because it
  centralizes cancellation shielding for owner transitions and cleanup. Kept
  GeneratedRolloutCapacity's reservation/scoring/release accounting together;
  it retains invariants across those transitions, not generated payloads.
- Non-goal: merging reward-service and continuous-owner cancellation semantics
  into a generic owner solely for fewer lines. Their cancellation contracts
  differ. No scheduling behavior changed in this slice.
- Validation: all 142 online trainer tests passed, including the existing
  component mean assertion in test_advantage_and_metrics. Touched-file Ruff
  lint/format checks passed. Continuous-owner findings are source review, not
  a new runtime validation claim. Repository-wide review remains incomplete.

## Continuous prompt group-size ownership and current-state comments

- Traced group_size from schedule/owner commands into producer batch installation.
  Removed repeated int coercion in owner comparisons, identity snapshots and
  producer construction. The owner now rejects invalid input before initial
  weight publication; _ActivePromptBatch validates direct producer inputs with
  the same shared exact-integer validator. Positive integer inputs are unchanged.
- Kept the batch state objects, command failure/cleanup policy and prefetch
  matching method. They own real lifecycle/identity constraints. No new wrapper,
  configuration knob or helper was introduced. Unit conversion and owner timeout
  constants were not changed by this input-validation slice.
- Replaced outdated Sprint 1/2 commentary on receipt attempts and batch IDs with
  current behavior: collection failure counts produce the attempt gauge, and
  batch identity already selects the demanded iteration. Removed historical
  sprint references from per-item timing comments while retaining their rationale.
- Validation: 190 continuous orchestration tests passed. Ten new cases reject
  bool, fractional/string, zero and negative group sizes through owner and
  producer boundaries, with no weight publication or collection. Touched-file
  Ruff lint/format and git diff --check passed. These tests do not prove the
  remaining repository-wide architecture review complete.

## Transactional prompt-batch replacement

- Found set_prompt_batch clearing installed batches before constructing its
  replacement. An empty prompt list or invalid group size therefore erased a
  completed batch even though installation failed.
- Renamed _install_prompt_batch to _new_prompt_batch and made it construct only.
  Replacement and append now explicitly publish the validated object and advance
  the ID in their own methods. Failed construction preserves state and the ID.
- Keep the shared constructor: both paths must capture the same policy version,
  slot identities and admission timestamps. Do not introduce a replacement flag,
  new factory class, or duplicate the batch construction. Existing incomplete-work
  and ready-item guards remain unchanged; no constants were added or moved.
- Validation: 192 continuous tests passed. Two new regressions drain the existing
  batch, reject empty prompts/invalid group size, verify its identity survives,
  then successfully generate the next batch with the next contiguous ID. Initial
  tests used a nonexistent queue method; corrected to the actual snapshot/remove
  API before the successful full run. Touched-file Ruff and diff checks passed.

## TeaCache shared metric honors its documented precision

- Reviewed rel_l1 with its runtime and teacache_drift_probe callers. Keep it as
  the shared mathematical definition: inlining would let analysis disagree
  with runtime skip decisions. Keep the asynchronous D2H helper in pipeline.py
  as well; pinned copies and source record_stream manage a real stream boundary.
- Found rel_l1 documenting FP32 reduction while subtracting and summing in the
  input dtype. Convert both signals to FP32 before subtraction/reduction. No
  new abstraction, constants, threshold changes or cache policy changes.
- Reproduced with 1024 FP16 values: 100 -> 101 previously yielded zero rather
  than 0.01; -40000 -> 40000 yielded NaN rather than 2.0. Both now match those
  expected ratios. A decision-level regression ensures 1% change exceeds a
  0.5% threshold instead of incorrectly skipping the forward.
- Validation: 141 denoise-step/full-sequence binding tests passed; three new
  numerical/decision regressions. Touched-file Ruff and diff checks passed.
  CPU numerical checks do not establish a GPU speedup or model-quality result.
- Follow-up candidate discovered, not changed here: TeaCacheConfig.from_sampling
  coerces enabled/warmup values and threshold validation does not reject NaN.
  Review configuration consumers before consolidating that validation.

## TeaCache runtime configuration validation

- Inspected YAML TeaCacheSection, DenoiseRequestOptions projection, direct probe
  construction and denoise consumption. YAML already uses StrictBool/StrictInt;
  runtime parsing must not reintroduce truthiness or truncation for direct callers.
- TeaCacheConfig now owns finite positive numeric threshold and exact nonnegative
  warmup validation. from_sampling validates its enable switch as boolean and
  forwards parameter values unchanged to the dataclass. Malformed strings/bools,
  fractional warmup and NaN/Inf threshold no longer silently alter skip behavior.
- Keep from_sampling as the optional bool/mapping construction boundary, the
  runtime dataclass as parameter owner, and the lightweight YAML schema without
  importing torch. Defaults and explicit disabled handling remain unchanged.
  No new wrapper class, standalone helper or constant was introduced.
- Validation: 185 step/binding/drift-guard tests passed; 23 new malformed-input
  cases exercise both direct construction and mapping parsing. After tightening
  test regex escaping for Ruff, all 34 TeaCache tests passed again. Touched-file
  Ruff lint/format and diff checks passed. No performance/quality claim inferred.

## Worker executor construction owns its protocol check

- Inlined the sole-use _require_chunked_executor into _build_executor immediately
  after dynamic construction. Retained callable checks and their error text for
  forward_batch/gather_batches. This is still a necessary protocol check; its
  separate private function had no caller outside the construction owner.
- Kept recursive _debug_metric_value as the diagnostic representation boundary.
  Kept sample_batches helpers: planner and family gatherers share row alignment,
  ordering and replay merging without importing one another. No generic helper
  container class or protocol/taxonomy constants changed.
- Validation: 142 generation execution tests passed, touched-file Ruff lint/format
  and diff checks passed. Many worker lifecycle tests inject the executor rather
  than build a real model; this suite is compatibility evidence, not a claim of
  full dynamic-family construction coverage. The moved check is otherwise
  behavior-preserving and does not justify a new model-loading integration test.

## Replay gathering count contract

- Reviewed full-sequence and chunk-autoregressive gather callers. Their shared
  gather_replay_tensors function is a legitimate cross-family boundary; keep its
  explicit sample-aligned wrapper and static context comparison rather than
  inferring sample axes from sequence lengths.
- Replaced positive-only count checking with the existing exact-integer guard,
  before payload traversal. Bool/floating/string counts cannot masquerade as
  row cardinalities, including for empty/static payloads where tensor row checks
  would never run. No extra helper or class introduced; production callers
  continue passing validated GenerationSampleBatch.sample_count values.
- Validation: 285 execution/binding tests passed, two skipped. Eighteen new
  malformed-count cases cover empty, static and tensor replay payloads. Touched
  Ruff lint/format and diff checks passed. This strengthens the direct shared
  API contract; it does not imply validated production batches were malformed.

## Chunk gatherer helper ownership

- Moved its three module-private helpers into ChunkAutoregressiveDenoiseGatherer
  as static methods. Repository search found only this class calling them. The
  two field helpers now accept ChunkAutoregressiveDenoiseResult sequences rather
  than untyped batches, making their actual ownership explicit.
- Keep separate methods for batch homogeneity, required fields and optional
  fields: the required-field operation is reused six times, while optional KL
  must distinguish all-absent from partially absent values. Keep the shared
  ordering/coverage/concatenation functions in execution.sample_batches for
  cross-family consistency. No new class, constant or runtime state introduced.
- Non-goal: reducing line count or removing meaningful checks. This is an
  ownership change preserving validation order and output/error behavior.
- Validation: 42 shared batch-gatherer/chunk-binding tests passed. Touched-file
  Ruff and diff checks passed. Full repository review remains incomplete.

## Chunk result axis cardinalities

- Inspected the executor result boundary, MAGI generation-only construction and
  CausVid runner trajectory mapping. Retain the common result class and executor
  adapter: they connect two families to one typed gather protocol.
- The result dataclass now requires exact integer temporal/transition counts
  using the existing shared validator. Range-only comparisons accepted bools
  and floats, even though these fields describe discrete tensor axes. Validation
  stays in the existing owner; no new helper, class or constants were added.
- Keep generation-only transition count None/zero/positive semantics and the
  stronger positive requirement when trainable tensors are present. Do not
  merge family-owned temporal scheduling into the transport executor.
- Validation: 96 chunk-binding, shared gather, CausVid and MAGI tests passed.
  Ten malformed-count and three valid optional-count regressions added.
  Touched-file Ruff and diff checks passed. No real model generation claimed.

## CausVid artifact construction ownership

- Moved the single-purpose _resolve_artifacts factory into the existing
  CausVidResolvedArtifacts.from_build classmethod, updating backend construction
  and its existing source-import-gate test. No additional object or forwarding
  compatibility function remains.
- Kept individual source/checkpoint/base-model resolvers as external-resource
  boundaries. Kept pinned repository/revision/file constants for reproducibility
  and protocol paths. Preserve the license/source/import/attention checks before
  weight resolution; this refactor does not change download or loading behavior.
- Validation: all 20 CausVid tests passed. Existing artifact-resolution test
  exercises the classmethod and retained source gate. Touched-file Ruff and
  diff checks passed; search confirms the removed factory has no references.
  Real checkpoint/model loading was not run. Whole-repository review continues.

## MAGI subprocess command ownership

- Moved build_magi_command onto the existing Magi1SubprocessConfig as build_command.
  Interpreter and entry path now come from self; model execution and its CLI
  contract test call that owner directly. Removed the old export/function.
- Keep prepare_magi_runtime_config as the adapter between base JSON, process
  config and per-sample inputs. Keep magi_subprocess_environment shared by
  preflight probing and generation. Keep pinned hashes, source paths and the
  sampling-to-runtime key map as explicit upstream protocol boundaries.
- No new launcher abstraction, CLI flag changes, environment changes or process
  execution changes. Validation: all 21 MAGI tests passed; touched-file Ruff and
  diff checks passed. No Python references to the removed function remain.
  Real upstream subprocess inference was not run. Repository review continues.

## MAGI process config reuse and finite timeout

- Reused the shared require_timeout guard in Magi1SubprocessConfig. The previous
  <= 0 comparison admitted NaN and infinity into subprocess waits. Normalized
  finite positive timeouts retain existing defaults and execution limits.
- from_build now uses dataclasses.replace on its preflight config to attach
  resolved checkpoint/T5/VAE paths instead of repeating interpreter/source/
  revision/config/timeout construction. Validation still runs on replacement;
  preflight remains before weight resolution.
- Keep subprocess execution, environment adapter and distinct preflight/generation
  deadlines unchanged. No new timeout helper, configuration field or class.
- Validation: 26 MAGI tests passed, including five invalid-timeout regressions.
  Touched-file Ruff lint/format and diff checks passed. This validates adapter
  behavior without claiming an upstream model or subprocess benchmark run.

## MAGI sampling conversion does not hide invalid values

- Removed int coercion and duplicate positivity checks from request preparation.
  Raw sampling overrides now reach the existing shared sampling-contract validator,
  which requires exact integers before geometry/divisibility checks. The same
  validator serves build preflight and prepared requests. Runtime JSON geometry
  and schedule fields can no longer be silently truncated either.
- Sample index is an exact nonnegative integer; seed is an exact signed integer
  before adding the sample offset. Keep existing omitted/None seed precedence,
  official runtime field mapping, and alignment/range errors. No new helpers,
  schema classes or constants introduced; the base config remains deep-copied.
- Validation: 37 MAGI tests passed. Eleven new cases cover fractional/string/bool
  sampling values and invalid sample indices, alongside existing seed/path and
  geometry tests. Touched-file Ruff and diff checks passed. No upstream inference
  or repository-wide completion claim follows from these tests.

## Shared artifact path boundary after symlink resolution

- Reviewed utils JSON, lifecycle, memory and artifact helpers against consumers.
  Keep atomic JSON replacement separate from trainer diagnostic append semantics;
  keep recursive tensor summaries distinct from generation scalar debug conversion.
  Keep artifact resolution shared by manifest validation and reward references.
- Found relative artifact paths rejecting lexical '..' but allowing symlinks to
  resolve outside data_root despite the stated containment contract. Verify the
  resolved path remains relative to the canonical root. Explicitly allowed
  absolute paths and internal symlinks retain their behavior.
- No new path-policy object or constant. DATA_ROOT_ENV and IMAGE_SUFFIXES remain
  environment/schema taxonomy boundaries; no business vocabulary moved into flow.
- Validation: 89 data tests and 27 Codex image-QA tests passed. New tests exercise
  external symlink rejection, internal symlink acceptance and explicit absolute
  paths. Touched-file Ruff and diff checks passed. This is path resolution, not
  protection against filesystem mutation after resolution. Existing manifests
  relying on relative symlinks outside data_root now fail the declared policy.

## Host-memory snapshot parser ownership

- Moved _read_proc_field_mb into HostMemorySnapshot as a private static method;
  only capture() calls it. Keep the parser shared for RSS/available/total units
  and missing-field handling instead of inlining three copies.
- Keep log_host_memory as the shared trainer/worker logging facade. No new
  memory-monitor object, polling state or constants. Missing metrics remain None.
- Validation: 44 memory-guard, online reward-flow and lifecycle tests passed.
  A direct capture against this host's /proc also returned positive RSS/total
  values. Touched-file Ruff and diff checks passed. Repository review continues.

## Config conversion exposes required-dependency failures

- Confirmed pyproject declares OmegaConf as required. Removed broad import-error
  fallback from plain_mapping and to_builtin_deep: broken dependency initialization
  must propagate rather than silently return incompletely converted data.
- Keep lazy imports, shared conversion functions and their existing representation
  rules. No ConfigConverter wrapper or new dependency added. Their users include
  trajectory storage policy, Ray resource parsing and generation runtime config.
- Validation: 368 utility/config/trajectory tests and 56 Ray resource tests passed.
  New tests preserve the original injected import failure through both APIs and
  check nested interpolation/tuple conversion. Touched-file Ruff and diff checks
  passed. Environments without the declared required dependency now fail clearly.

## Dynamic imports can address class-owned factories

- Found import_from_path documenting dotted attribute chains while performing
  only one getattr on the module. Resolve each explicit attribute component so
  module:Class.from_build can address a constructor without an external wrapper.
- Keep the shared import function and mandatory colon grammar; direct module
  attributes remain supported. No alternate module guessing, exception fallback,
  registration table or compatibility forwarding function added. Existing registry
  paths remain unchanged; no production recipe was migrated in this slice.
- Validation: 72 utility/family-registry/checkpoint-identity tests passed. New
  tests call a class-owned factory, preserve missing-attribute errors, retain
  direct class lookup and reject missing module/attribute separators. Touched-file
  Ruff and diff checks passed. Repository-wide review remains incomplete.

## Token configuration projections remain a uniform adapter boundary

- Reviewed registry TokenFamilyBuild.config_builder declarations, the shared
  build_token_family_bundle consumer, and Janus/NextStep/Emu3/GLM/LlamaGen
  projections. These produce dictionaries; the shared builder subsequently
  constructs config_cls and model_cls. They are not simple object factories.
- Keep these named family projection functions and token_model_config_base.
  They preserve family-specific fields while leaving absent LoRA defaults with
  config dataclasses. Keep _validate_token_lora_path shared by projection and
  assembly so direct assembly fails before model construction.
- Non-goal: migrate every registry entry to classmethods merely because dynamic
  imports now support them. Preserve the common cross-family adapter shape;
  no runtime code or registry constants changed in this review.
- Validation: token LoRA/default and training-capability tests were run together
  (30 passed). Their assertions exercise both
  projection and configuration construction; no actual pretrained model loaded.
- Remaining candidate: LlamaGen projection compares fixed geometry after int()
  conversion, potentially hiding fractional overrides. Requires focused caller/
  schema review before changing it; this keep decision is about function shape,
  not proof that every value conversion is correct.

## LlamaGen geometry projection preserves exact dimensions

- Followed the prior projection-review candidate into the shared square-grid and
  decoded-size helpers. They now require integer token counts/strides rather
  than truncating. Projection passes model token count unchanged and checks
  sampling dimension types before comparing them with checkpoint geometry.
- Keep the shared geometry helpers, architecture constants and dictionary
  projection interface. Positive/square-grid and mismatch rules remain; no new
  helper/class or alternate geometry inference was introduced.
- Validation: 64 LlamaGen and shared token-LoRA tests passed. Nine new cases
  reject fractional/string/bool geometry through projection and shared helpers.
  Touched-file Ruff and diff checks passed. Model/config fields elsewhere that
  are already validated were not mechanically stripped of all conversions.

## Combined regression after configuration and ownership cleanup

At 69db06136, ran together:

`tests/rollouts tests/trainers tests/generation/execution tests/generation/bindings
 tests/generation/steps tests/config tests/utils tests/trajectory`

Result: 1703 passed, nine skipped, 16 warnings in 63.96 seconds. Skips include
unavailable bitsandbytes, opt-in distributed FSDP tests, and two vLLM internal
paged-attention imports unavailable in the installed environment. Do not count
those paths as verified by this run. Earlier isolated paged-attention results do
not override these current combined-run limitations.

The run checks compatibility across recently changed config conversion, generation
geometry/metrics, continuous scheduling, and common utility boundaries. It is not
proof that all repository helpers or architecture ownership have been reviewed.
Revisited scripts/common/factory.py alongside AlgorithmConfigContract: the existing
factory structural review above remains relevant. Do not turn its dispatch branches
into config facts merely to remove a local set; changing dispatch ownership requires
tracing algorithm construction and evaluator selection together.

## Categorical log-prob token identity boundary

- Inspected shared math callers in replay and GLM/LlamaGen/paged token sampling.
  Keep gather_categorical_log_probs as their common temperature/normalization
  implementation; moving it onto one model would duplicate policy math.
- Reject floating, complex and boolean token IDs before dtype/device conversion.
  Previously an ID such as 1.9 silently selected token 1. Integer tensor widths
  still convert to long for gather; no formula, chunking or temperature change.
- No new validator function/class/constant introduced. Validation: 50 replay,
  fused-linear log-prob and LlamaGen runner tests passed, including five invalid
  dtype cases and an int32 numeric comparison. Touched-file Ruff and diff checks
  passed. This protects the shared function's input boundary; it is not evidence
  that current production samplers were emitting fractional token IDs.

## Fused categorical path retains the same token identity contract

- Followed the eager identity check into fused_linear_logprob, which independently
  converted IDs to long. Reject float/complex/bool IDs before projection there
  too, preserving integer-width conversion and forward/backward implementations.
- Keep this small entry-point condition local rather than adding a separate
  validator class/function just to hide one predicate. Shared normalization math
  and the custom autograd/Triton boundary remain unchanged.
- Validation: 28 fused/eager log-prob tests passed. Five invalid-dtype tests assert
  failure before F.linear, plus int32/eager parity. Touched-file Ruff and diff
  checks passed. No speed or end-to-end training-quality claim was made.

## Log-prob chunk cardinality and default ownership

- Fused chunk_rows lacked validation: a negative range step with nonempty rows
  skipped every projection and returned an uninitialized output. Require positive
  exact integer overrides before custom autograd execution. Eager chunk_size uses
  the same existing integer validator instead of accepting bools/range errors.
- Inlined sole-use _chunk_rows_for into the default selection site. Keep the
  isolated kernel buffer-budget and tile-size constants and paired Torch/Triton
  forward/backward adapters; those are real implementation boundaries. Default
  chunk selection and positive-integer execution behavior remain unchanged.
- Validation: 38 fused/eager log-prob tests passed, including ten invalid chunk
  arguments. Touched-file Ruff/diff checks passed and removed helper has no
  remaining references. No throughput claim follows from this correctness fix.

## Empty token batches preserve eager/fused parity

- Eager categorical gathering previously reached torch.cat([]) for zero token
  positions, whereas the fused implementation returned an empty result. Return
  the empty fp32 reduction shape while retaining the logits' autograd connection.
- Keep input validation before the empty branch and leave nonempty chunking,
  normalization and fused backward unchanged. No empty-batch helper/class added.
- Validation: 40 fused/eager log-prob tests passed. New cases cover zero batch
  rows and zero sequence length; outputs match shape/dtype and hidden/weight/bias
  gradients are present, equal and zero. Touched-file Ruff and diff checks pass.
  This proves boundary consistency, not occurrence in a production training run.

## Replay request segment-name boundary

- ReplayRequest now rejects bare str/bytes segment_names instead of iterating
  characters as independent names. Validation stays in the existing request
  dataclass; None and sequence handling remain unchanged.
- Corrected ReplaySegmentResult.logprobs documentation: directly supplied log-probs
  come from the family's current replay computation, not reused rollout scores.
  Keep payload selection/access methods as the shared family/evaluator boundary;
  no new helper, wrapper or constants.
- Validation: 142 interface/replay tests passed, including four bare-name cases.
  Touched-file Ruff and diff checks passed. Whole-repository review continues.

## Recorded replay dimensions retain family units and exact values

- Traced Emu3/GLM replay grid reconstruction through replay_context_image_size.
  Require positive integer dimensions instead of int truncation. Preserve the
  recorded context source order and family-specific expected-token calculation.
- Corrected the shared docstring: Emu3 records latent grid dimensions, GLM records
  pixels. Do not infer image aspect ratios from token count or apply a common
  pixel conversion. Keep the shared cross-family helper and model grid adapters.
- Validation: 54 Emu3/GLM replay and interface-contract tests passed. Eight new
  cases enter the actual tiny Emu3 replay method with invalid recorded dimensions.
  Touched-file Ruff/diff checks passed; no pretrained model run was performed.

## Runtime protocol acceptance checks callable members

- _require_protocol previously returned on runtime isinstance before checking
  callable members, leaving the callable check only in its diagnostic path.
  Require both checks for acceptance; a same-named noncallable attribute no
  longer passes the runtime/replay model boundary.
- Inlined the sole-use _missing_callables comprehension into the guard. Keep
  typed require_replay_model/require_runtime_model facades and derive required
  names from the actual Protocol rather than a parallel hardcoded method list.
- Validation: 308 interface/replay/generation-execution tests passed. New tests
  replace every protocol member with None/42 and check rejection, while complete
  callable implementations are returned unchanged. Touched-file Ruff/diff pass.

## Reference token scoring consistently excludes autograd

- ref_forward now applies no_grad to both explicit-reference and adapter-disabled
  paths. Keep reference selection/adapter lifecycle and the shared three-evaluator
  helper. A reference model with trainable parameters must not create a loss graph.
- Traced multi-segment evaluation beyond forward: its later log-prob conversion
  also runs under no_grad, since payloads can retain trainable tensors/weights.
  Current policy normalization remains differentiable. No generic context wrapper
  or constants introduced; denoise reference conventions are not merged here.
- Validation: 43 replay tests passed. New tests check both reference-selection
  paths, exception/gradient/adapter restoration, and actual multi-segment scoring
  with trainable payloads (current gradients retained, reference gradients absent).
  Touched-file Ruff/diff checks passed. No training-speed claim is inferred.

## Multi-segment selection is stable across evaluations

- Materialize enabled_segments once as a tuple in the existing evaluator. A
  generator previously worked for one evaluation, then was exhausted on the
  next. Validate individual names and duplicates without a new config object.
- Keep empty selection construction supported: interface tests use it to check
  evaluator capabilities, and evaluation retains its existing no-enabled-segment
  error. An initial stricter empty-selection check was corrected after that
  existing contract test exposed the compatibility requirement.
- Validation: 74 replay/common-factory tests passed. New tests reuse a generator-
  configured evaluator for two actual iterations and reject bare/invalid/duplicate
  names. Touched-file Ruff and diff checks passed. Selection order is preserved.

## Multi-segment evaluator consumes typed trajectory segments directly

- Removed the typed-segment -> temporary payload dict -> tensor extraction round
  trip. The evaluator retains TrajectorySegment objects and reads action,
  old_log_prob and mask by their declared roles. Removed the payload factory and
  the single-use log-prob forwarding method (28 net source lines removed).
- Keep the small shared role-to-Torch guard and categorical selection method;
  they own actual checks. Keep enabled order, primary selection, model result
  lookup, reference no_grad and signal construction. Unselected segments are no
  longer unpacked just to discard their temporary dictionaries; consumed roles
  still require unique declared tensors and Torch values.
- Validation: 74 replay/factory tests plus eight Janus-R1 wiring/model tests passed.
  An initial command used a nonexistent janus_pro_r1 test directory and ran no
  tests; corrected to the actual registered Janus test locations. Touched-file
  Ruff/diff checks passed. No extra container class or schema constants added.

## Signal builder removes a single-use role forwarding method

- Inlined _old_log_prob_from_trajectory into segment_signal: read the declared
  old_log_prob role and call the existing loss-value selection method directly.
  Typed the mask helper's segment argument as TrajectorySegment and removed an
  unnecessary return temporary.
- Keep the shared builder, mask selection and timestep/shape logic: these serve
  denoise and token evaluators. No representation conversion or new abstraction
  introduced. Device movement, override precedence and final signal validation
  remain unchanged; this slice does not redesign timestep-axis selection.
- Validation: 78 replay/trajectory tests passed. Touched-file Ruff/diff checks
  passed; the removed method has no remaining references. Full review continues.

## Signal timestep selection follows declared trajectory axes

- Replaced the signal builder's rank/shape heuristic with selection of the
  tensor axis declared as denoise_step. Previously an extra token dimension
  could be mistaken for a timestep; selection also assumed dimension one.
  The new method reads TrajectoryTensor axes and the trajectory axis kinds,
  independently of the replay output shape. Multiple denoise-step axes fail
  explicitly. Removed _same_shape and the old conditional selection method.
- Keep the shared builder, role lookup, mask-name preference, explicit value
  overrides and final signal shape validation. The remaining selection method
  serves both recorded old log probabilities and masks; no new helper module,
  class or ALL_CAPS table is needed. Cross-family signal construction remains
  shared rather than duplicated for line-count reduction.
- Denoise callers supply timestep_idx; token and chunk-autoregressive callers
  retain their complete loss axes. Malformed token signals now reach the shape
  error instead of being silently sliced to fit. Selection still precedes
  device movement.
- Validation: 81 replay/trajectory tests passed, including new renamed/reordered
  denoise-axis cases and rejection of accidental token-axis slicing, plus the
  existing diffusion deferred-device-move regression. Touched-file Ruff and
  diff checks passed. The full repository review remains ongoing.

## Denoise evaluator reuses its replay payload for optional caches

- Removed _cached_ref_noise_pred and _old_prev_sample_mean. Both independently
  rebuilt the same replay dictionary already available in evaluate, then
  guessed that a rank-one value was already step-selected. The consuming
  branches now read that dictionary and explicitly select [:, timestep_idx]
  before device movement, matching DenoiseTrajectoryBuffers allocation and
  FullSequenceDenoiseExecutor export contracts.
- Keep absent-cache behavior, reference-demand gating, device movement, and
  cached/fresh reference math. No new class, helper module or constants added.
  Shared trajectory resolution remains the cross-family boundary; changing
  its representation is not necessary to remove these duplicate consumers.
- Remaining schema work: optional replay tensors currently declare only sample
  alignment even when their payload includes steps. This slice preserves that
  stored representation. A future axis-schema change must cover persisted
  trajectory reads as well as producers; it must not cause these caches to be
  sliced twice or old payloads to be silently interpreted as step-selected.
- Validation: 180 replay, full-sequence-denoise binding and trust-region tests
  passed, including cached/fresh reference parity, requested proposal-mean
  selection and rank-one cache rejection. An initial test-helper edit also
  inserted setup into an unrelated test; corrected that insertion before the
  successful run. Touched-file Ruff and diff checks passed.

## Replay resolver rejects incomplete axis requests

- The replay_tensor_dict boundary previously ignored axis-only/index-only
  requests and unknown axis names, returning the entire payload. Negative
  indices also selected from the end under tensor indexing semantics. Require
  paired axis/index arguments, a declared axis, and an exact nonnegative integer
  index before resolving payloads. Reuse require_exact_int instead of adding
  another independent numeric validator.
- Keep static tensors without the selected axis unchanged and retain the
  existing per-tensor upper-bound check, slicing-before-device-movement and
  shared resolver API. The helper is a cross-family resolution boundary, not
  removable forwarding boilerplate. No ALL_CAPS data or runtime class added.
- Inspected denoise model-base, SDE evaluator and CausVid call sites: production
  step selections already supply both arguments; full replay uses neither.
  storage.py controls runtime placement/dtype rather than persisted schema
  migration. Optional-cache axis declarations remain a separate open item;
  this change does not claim to migrate them or complete the repository audit.
- Validation: 198 trajectory, replay and full-sequence-denoise binding tests
  passed. Nine new resolver cases cover invalid/partial requests, upper bounds,
  correct step selection and unchanged static payload identity. Touched-file
  Ruff and diff checks passed.
