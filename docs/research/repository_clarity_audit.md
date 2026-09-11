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
