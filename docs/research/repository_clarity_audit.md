# Repository clarity audit

Status: in progress. This audit covers the whole repository, with priority on
core generation, scheduling, and external helpers. Function counts identify
inspection candidates; they do not establish that a function needs relocation.
No repository-wide completion claim is supported yet.

## Current scope inventory (source snapshot a64a33ac1)

This AST inventory counts direct module-level definitions in `vrl/**/*.py`.
It excludes methods, nested functions, tests and files outside the package.
Counts locate review work; they do not measure defects or prove coverage.
Historical sections below record individual slices, not directory completion.

| Area | Python files | Module functions | Module classes |
| --- | ---: | ---: | ---: |
| (root) | 4 | 8 | 7 |
| algorithms | 14 | 15 | 30 |
| config | 16 | 38 | 56 |
| generation | 49 | 19 | 85 |
| math | 8 | 13 | 2 |
| models | 128 | 223 | 214 |
| nn | 22 | 35 | 25 |
| ray | 7 | 31 | 22 |
| rewards | 70 | 115 | 109 |
| rollouts | 36 | 11 | 48 |
| scripts | 98 | 542 | 56 |
| trainers | 27 | 114 | 59 |
| trajectory | 9 | 25 | 11 |
| utils | 11 | 49 | 11 |

The next inspection focus moves beyond recently rechecked generation and rollout
paths to scripts and their model/reward consumers. Candidate files include
`vrl/scripts/perf/nsys_report.py` (interval math versus report ownership),
`vrl/scripts/rewards/install_countgd.py` (installation state and subprocess
boundaries), and `vrl/scripts/eval/sana_aesthetic_report.py` (report construction
versus protocol validation). These are inspection candidates, not approved
relocations: read callers and tests before choosing any change.

Retain CLI adapters, numerical helpers, protocol/schema constants and genuinely
shared utilities when they explain a boundary. Do not introduce classes merely
to reduce this table's function counts. Whole-repository completion still needs
current source review beyond these local regressions and beyond the `vrl` package.

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

## Resolver owns explicit tensor and sequence slicing

- Moved axis slicing into TrajectoryResolver and removed the standalone
  _slice_sequence_axis and _shape helpers. The old len(value) shape fallback
  classified nested lists as rank one, rejecting legal second-axis selection.
  It also retried arbitrary failed tensor indexing as sequence iteration.
- Dispatch lists/tuples explicitly and recurse over their declared axis;
  tensor-like values retain shape bounds, select or tuple indexing. Tensor
  operation failures are wrapped with the tensor reference and original cause,
  never retried as an unrelated operation. No eager Torch import introduced.
- Keep _split_ref as the small reference-format parser and retain the shared
  cross-family resolver API. No new class or ALL_CAPS table; consistency of
  tensor/sequence resolution matters more than flattening every helper.
- Validation: 201 trajectory, replay and full-sequence-denoise binding tests
  passed. New tests cover accepted nested lists/tuples and a backend indexing
  failure whose identity must survive without iteration. Touched-file Ruff and
  diff checks passed. Optional-cache schema and full-repository review remain
  open; this slice does not claim either is complete.

## Axis records validate their declared semantics

- TrajectoryAxis now requires a nonempty string name, a kind from the existing
  AxisKind Literal, and an exact nonnegative integer length when specified.
  Previously booleans and fractional lengths passed construction, and a typo
  in denoise_step could cause semantic axis selection to skip that dimension.
- Reuse require_exact_int and derive kind membership directly from AxisKind;
  no parallel ALL_CAPS vocabulary or standalone validation helper added.
  Keep the axis schema, arbitrary valid axis names, None/zero lengths and the
  batch validator's cross-record checks. No changes to cached replay payload
  layout or stored-record migration are claimed.
- Validation: 250 trajectory, replay and generation-binding tests passed;
  two vLLM paged-attention tests skipped when its internal dependencies were
  unavailable. Corrected literal-dot regexes flagged by Ruff and reran the
  16 new axis tests successfully. Touched-file Ruff and diff checks passed.
  Full repository review remains ongoing.

## Denoise probe step limit is explicit

- DenoiseLoopConfig now validates optional execute_steps as an exact positive
  integer. The public forward_probe_batch entry uses the same existing integer
  validator before encoding or model preparation. Previously floats/bools could
  reach int() in the loop, and nonpositive direct config values were repaired
  silently by max(1, ...).
- Removed the loop's cast and lower-bound repair. Keep the upper bound at the
  available schedule length, full buffer allocation for memory sizing, and the
  probe's reuse of the canonical generation flow. run_denoise_loop remains a
  cross-family execution kernel; no wrapper class or new validation helper is
  warranted. No ALL_CAPS data added.
- Validation: 184 denoise-step, full-sequence binding and batch-memory-shadow
  tests passed. Ten new cases verify early probe rejection without encoding
  and rejection of invalid directly constructed loop configs. Touched-file
  Ruff and diff checks passed. Repository review remains ongoing.

## Token output lane names are checked before updates

- TokenAutoregressiveEnvelope.apply_step_output previously mutated known lanes
  before discovering an unknown name later in the output dictionary. Validate
  all names first, then scatter. Removed the single-use _require_row_lane lookup
  method; the error still names the unknown lane.
- Keep ARCacheRows as the shared batching/cache boundary and preserve scheduling
  order, subset updates and scatter semantics. This is not a transaction or a
  rollback guarantee for backend scatter failures; no transaction wrapper,
  schema table or additional runtime class is introduced.
- Validation: 30 token composition/binding tests passed, with two vLLM internal
  dependency tests skipped. New cases place the unknown lane before and after
  a valid update and verify that original cache row identity and value survive.
  Touched-file Ruff and diff checks passed. Full repository review continues.

## AR row merging rejects structure loss

- Plain cache concatenation previously traversed only the first row's mapping
  keys or sequence length. Extra fields/elements in later rows were silently
  discarded. Require equal mapping key sets and matching list/tuple structure
  and lengths before recursively merging that level. Dictionary insertion
  order may differ; output retains the first row's order with key alignment.
- Keep ar_split_rows/ar_concat_rows and their plain/HF adapters: they serve
  ARCacheRows, torch attention and GLM generation, not a single incidental
  caller. No new owning class, helper function or ALL_CAPS taxonomy is needed.
  Existing tensor concatenation and DynamicCache handling remain unchanged.
- Validation: 64 cache-row, token composition and GLM-family tests passed.
  New cases reject missing/extra fields, unequal sequence lengths and mixed
  containers; reversed mapping insertion order preserves tensor alignment.
  Touched-file Ruff and diff checks passed. Full repository review continues.

## DynamicCache splitting shares exact row-count semantics

- Replaced direct per-row K/V slices in _split_hf_cache_rows with existing
  ar_split_rows for every layer's key and value before constructing output
  caches. Previously requesting too many rows produced empty cache rows and
  requesting too few silently discarded rows, unlike the plain tensor path.
- Keep the HF read/reconstruction adapters as external API boundaries and the
  recursive public split/concat API shared by attention and generation. No new
  helper, class or constant introduced. Valid row ordering and tensor views
  remain unchanged; empty DynamicCache objects still have no tensor batch size
  to validate.
- Validation: 68 cache-row, token composition and GLM-family tests passed.
  New cases cover too few/many requested rows and mismatched K or V in a later
  layer. Touched-file Ruff and diff checks passed. Full review remains active.

## Combined regression after replay and cache-boundary cleanup

At commit 4ea7c543d, with a clean worktree, ran:

```text
.venv/bin/pytest -q tests/rollouts tests/trainers tests/generation/execution tests/generation/bindings tests/generation/steps tests/generation/composition tests/config tests/utils tests/trajectory tests/nn/layers/test_attention_cache_rows.py
```

Result: 1,815 passed, nine skipped, 16 warnings in 66.33 seconds. This combines
recent axis selection, resolver dispatch, denoise probe limits, token output
validation and plain/DynamicCache row handling with trainer, scheduling and
configuration consumers. No additional runtime edits were required by this run.
Skipped tests are not counted as verified optional-backend/distributed behavior.

The original repository-wide scope remains open. In particular, optional denoise
cache axis declarations and their stored representation need a coherent review,
and model-family/script ownership coverage is not established by this test run.
Historical entries above describe their respective slices and may name helpers
removed by later entries; they are not an inventory of current symbols. Passing
this combined regression is compatibility evidence, not architectural completion.

## Denoise replay does not conceal device lookup failures

- Removed the broad exception fallback around self.device in the shared denoise
  replay input method. A failed device property previously became device=None,
  silently disabling movement and deferring the error to later computation.
  Read the declared model device directly before resolving any replay payload.
- Keep the shared replay method, lazy trajectory import and step-before-device
  selection. No new class, helper or constant; this is an existing model-owned
  responsibility. An explicitly returned None retains its existing semantics.
- Validation: 272 replay, model-interface and full-sequence-binding tests passed.
  New cases verify no slicing/movement after RuntimeError or AttributeError in
  device lookup. The initial test overclaimed AttributeError identity: nn.Module
  replaces property AttributeError during attribute lookup. Corrected that
  assertion; RuntimeError identity is retained, both error types propagate.
  Touched-file Ruff and diff checks passed. Full review remains open.

## Shared weight preflight establishes actual tensor identity

- require_weights_for now requires Torch Tensor payload values instead of only
  checking shape/dtype attributes. A non-tensor impostor could pass preflight,
  causing load_state_dict to fail after earlier parameters had been copied.
  Removed the redundant tensor check in verify_weights_in; it already invokes
  the shared preflight. Torch remains lazily imported at the validation call.
- Keep shared load/require/readback functions: single-transformer and multi-root
  families use this boundary, and readback is explicitly acceptance-only.
  No helper class or ALL_CAPS constants added. This does not promise rollback
  for all loader/backend errors, only rejection of non-tensor values before
  live copying starts.
- Validation: 134 model-weight, trainer-weight-sync, online-state and model
  interface tests passed. A valid first parameter followed by a shape/dtype
  impostor is rejected without changing either live parameter. Touched-file
  Ruff and diff checks passed. Repository review remains ongoing.

## Policy slots preserve exact version identity

- TrainableStateSlots now requires an exact positive retention count and exact
  nonnegative versions for install/has/get, using require_exact_int. Removed
  int conversions that allowed booleans, strings and fractional versions to
  alias an existing policy. Denoise activation and acceptance readback validate
  before active-version comparisons, including the idempotent fast path.
- Keep the existing version container, numeric-version eviction, None-payload
  aliasing and shared integer validator. No new version wrapper/helper/table;
  version identity is an existing scheduling boundary, not a parsing heuristic.
- Validation: 205 weight-container, replay, versioned-worker and model-interface
  tests passed. New cases verify invalid install/read operations preserve stored
  state and invalid activation cannot bypass validation through the current-slot
  fast path. Touched-file Ruff and diff checks passed. Full review remains open.

## Versioned transfer boundary retained after caller review

- Followed exact policy versions through GenerationWorkerCore.update_weights,
  verify_active_weights, RayWeightSync and staged-transfer construction. These
  boundaries already validate nonnegative exact integers before their state
  changes; no upstream conversion bypass of the slot fix was found here.
- Keep weight_manifest and iter_weight_chunks/iter_weight_buckets as stateless
  sender protocol operations. Cloned chunks bound serialized backing storage;
  bucket packing and manifest validation are distinct operations. Keep
  StagedWeightTransfer as the receiver buffer/order/completion owner. Combining
  these into a namespace class would not remove actual shared complexity.
- Keep worker begin/receive/commit/abort methods as the transport-facing facade;
  they enforce transfer lifecycle and coordinate installation. No runtime edit,
  new class, helper or constant is justified by this review. The wire-byte
  ceiling does not bound full sender/receiver state RAM.
- Validation: 58 transfer, Ray weight-sync and versioned-worker tests passed in
  17.16 seconds, including local real-Ray ACK/deadline/partial-transfer tests.
  This does not establish multi-node or full-model training performance. Full
  repository review remains open.

## Weight namespace unwrapping recognizes framework types

- unwrap_compile_and_ddp now peels actual OptimizedModule, DDP and FSDP1
  instances. Attribute-name guessing previously peeled any ordinary child named
  module or _orig_mod, potentially excluding the parent's other parameters from
  export and changing the accepted weight namespace.
- Keep this shared framework adapter and its lazy imports, nested wrapping,
  and PEFT preservation. No wrapper registry, extra class or ALL_CAPS table.
  Updated the DDP test double to identify as DDP and replaced a compile-shaped
  fake with torch.compile; nominal wrapper types are now part of the contract.
- Validation: 197 weight, checkpoint, DDP and denoise-base tests; 61 FSDP/strategy
  tests (two optional tests skipped); and 11 PEFT adapter tests passed, totaling
  269 passes. New export/load regressions retain ordinary module/_orig_mod
  children and a sibling trainable parameter. Touched-file Ruff and diff checks
  passed. No full multi-node training claim; repository review remains active.

## FSDP block discovery unwraps declared framework types

- FSDP unwrap_module now recognizes OptimizedModule and PeftModel instead of
  guessing from _orig_mod/get_base_model/base_model.model attributes. Removed
  the base_model.model fallback; ordinary library-model children and methods
  no longer redirect block discovery to an unrelated object.
- Keep this FSDP-specific helper: its purpose is underlying block discovery,
  unlike weight namespace unwrapping, which must retain PEFT. No new wrapper
  class, dispatch table or standalone helper. Tests now use real PEFT/compile
  wrappers rather than SimpleNamespace objects with matching field names.
- Validation: 70 FSDP, strategy and FSDP-gather tests passed, two optional tests
  skipped. New tests preserve ordinary named children and never call an
  unrelated get_base_model method. Touched-file Ruff and diff checks passed.
  Repository-wide review is still incomplete.

## FSDP rejects ineffective block declarations

- iter_blocks rejects malformed class-name declarations and raises when no
  module matches. Previously a string became a set of characters and a typo
  silently yielded nothing, letting apply_fsdp shard only the root despite its
  per-block contract. The unmatched case now fails before fully_shard is called.
- Keep the model-owned _no_split_modules declaration and shared discovery
  function. No per-family table or wrapper class added. A declaration may name
  optional classes absent from one model variant as long as some blocks match;
  requiring every declared class would unnecessarily narrow that contract.
- Validation: 69 FSDP/strategy tests passed, two optional tests skipped. New
  malformed-declaration and unmatched-block tests verify no sharding begins on
  these invalid inputs. Touched-file Ruff and diff checks passed. Full repository
  review remains active.

## FSDP module ownership description matches implemented operations

- Replaced stale module documentation claiming all operations were collective
  and optimizer state sharding belonged to later work. The current module
  implements optimizer gather/restore as well as local wrapper/block/policy
  inspection. Documented those responsibilities and primary/non-primary export
  behavior directly, without historical roadmap claims.
- Keep normalize_fsdp_parameter_dtype as a framework preparation operation.
  Keep _full_cpu_tensor, _gather_named_full_cpu and _materialize_full_cpu: these
  share collective ordering and explicitly avoid retaining full tensors on
  non-primary ranks. Generic tensor-tree mapping would lose that retention
  behavior. No helper-container class or ALL_CAPS data warranted.
- Follow-up identified in FSDPStrategy.prepare_model: absent handle.dtype falls
  back to the first parameter dtype. The relationship to resolved model
  precision requires caller/family review before changing mixed-dtype behavior.
- Documentation-only edit; no runtime or numerical changes. Touched-file Ruff
  and diff checks passed. Existing test results above are not presented as a
  new run. Full repository review remains open.

## FSDP strategy no longer chooses mixed precision by first parameter

- Removed the strategy fallback to next(handle.parameters()).dtype. Without a
  handle dtype, require a single common parameter dtype; mixed values require
  an explicit target. Invalid non-None dtype declarations fail instead of being
  ignored. Parameter registration order can no longer select the target through
  this fallback and silently cast all other parameters.
- Keep the existing handle dtype contract, normalization function and strategy
  owner. Model-specific dtype properties still own their implementation; this
  change does not redefine those properties as config provenance. No additional
  precision class or standalone helper introduced.
- Validation: 78 FSDP/strategy/gather tests passed, two optional tests skipped.
  Mixed-source actor fixture now declares its target and still forwards after
  normalization; new opposite-first-dtype cases reject ambiguity without casts.
  Invalid string dtype is rejected. Touched-file Ruff and diff checks passed.
  Full repository review remains active.

## Strategy removes two single-use forwarding helpers

- SingleProcessStrategy.__init__ constructs its default DistributedTrainingContext
  directly; removed _single_process_context, which had no other caller.
  Training-memory parking updates its existing identity set from _module_tensors
  directly; removed _module_tensor_ids and its unnecessary intermediate set.
- Keep the shared module/tensor movement and CPU coordination helpers, and the
  FSDP compile guard. These serve cross-strategy execution or a distinct config
  boundary. No namespace class or constant table added. Default rank/device,
  tensor deduplication and parking/restore order remain unchanged.
- Validation: 84 strategy/FSDP/DDP tests passed, two optional tests skipped.
  Removed symbols have no remaining source/test references. No new tests for
  this behavior-preserving relocation. Touched-file Ruff and diff checks passed.
  Full repository review remains active.

## Training parking device recording has a narrower contract than placement snapshots

- Inspected _module_device, module/tensor restore records, rollback and the
  diffusion move_frozen_components hook. Added a docstring identifying the
  actual module-level contract: one restore destination, not arbitrary mixed
  placement preservation. The current first registered tensor chooses that
  destination; tensor-free modules use the supplied training device.
- Keep _move_module and separate optimizer/EMA tensor restore handling. Model
  movement includes unregistered frozen pipeline components, so replacing it
  with a parameter-only tensor walker would omit live accelerator allocations.
  No new snapshot class or inference helper introduced.
- Open limitation: mixed per-submodule placement is not recorded/restored by
  this path. A proper extension needs model-owned placement capture including
  unregistered components, plus rollback and accelerator tests. This review
  does not label that limitation fixed or claim generic mixed-device support.
- Documentation-only change; touched-file Ruff and diff checks passed. No new
  runtime test run claimed. Full repository review remains active.

## Replay declares absence of generation components

- Removed RuntimeError-as-absence handling from generation_memory_targets and
  move_frozen_components. Pipeline failures now propagate. ReplayRolloutStubs
  explicitly returns no generation targets and performs no frozen-component
  moves; this common owner covers Diffusers replay plus Wan and Anima's custom
  replay inheritance. Pipeline-free generation still supports its own VAE.
- Keep the model memory interfaces and component traversal. The two small replay
  implementations are intentional protocol behavior, not forwarding wrappers;
  no new class or frozen-component name list added.
- Validation: final denoise/interface/Wan/Anima run passed 346 tests; an earlier
  denoise/interface/Echo run passed 307 before moving the no-op methods to the
  shared replay mixin. New tests exercise replay absence explicitly and preserve
  pipeline RuntimeError identity. Touched-file Ruff/diff checks passed. Full
  repository review remains active.

## Frozen offload rejects malformed component inventories

- move_frozen_components distinguishes absent pipeline from an existing pipeline
  without a valid components mapping. The latter now raises instead of silently
  skipping frozen offload. Pipeline-free models and replay no-op interfaces
  remain valid; module deduplication and transformer/non-module exclusions stay.
- The component inventory remains pipeline-owned, with no frozen-name table,
  extra helper or runtime class. This validates the existing interface rather
  than guessing another way to discover components.
- Validation: 443 denoise-model, rollout-orchestration and trainer-strategy tests
  passed, two optional tests skipped. New malformed-inventory cases fail before
  movement; absent pipeline remains a no-op. Touched-file Ruff and diff checks
  passed. Full repository review remains active.

## Frozen-component movement follows module registration

- move_frozen_components excludes every pipeline component registered within
  the model, using nn.Module.modules identities. Previously only self.transformer
  was excluded, so a second registered transformer could be moved again by the
  frozen-component hook. nn.Module.to owns registered modules; the hook owns
  the unregistered pipeline remainder.
- Keep pipeline component inventory, frozen-component identity deduplication
  and the shared movement interface. Registration expresses ownership directly;
  no per-family transformer name table or extra helper/class is introduced.
- Validation: 304 frozen-offload, Wan, orchestration and strategy tests passed,
  two optional tests skipped. A two-expert regression leaves both registered
  modules untouched by this hook while moving the unregistered VAE. Touched-file
  Ruff/diff checks passed. Full repository review remains active.

## Pipeline load dtype audit identifies premature precision reduction

- Inspected diffusers_pipeline_dtypes and both production consumers: shared
  DiffusersPipelineModelBase.from_build and MiniMaxH3Model.from_build. Keep the
  helper as a cross-family load projection, not a single-owner constructor.
- Confirmed an unresolved precision issue: model_dtype != float32 emits one
  scalar torch_dtype for the whole pipeline. Subsequent VAE.to(float32) cannot
  recover precision discarded while loading FP32 source weights into BF16/FP16.
  A prompt encoder override differing from model_dtype is also applied only
  after that initial scalar load on this branch.
- A coherent fix must project component-owned load dtypes, covering dual
  transformers and MiniMax audio_vae as well as the ordinary VAE. Adding only
  a hardcoded vae exception would leave the other component owners incomplete.
  Existing model dtype/provenance propagation and Hub revision kwargs must stay.
- No runtime change or new test run in this inspection. This is a concrete open
  issue, not a claim of preserved FP32 source precision. Full review is active.

## Component load dtypes prevent avoidable VAE precision loss

- Replaced scalar whole-pipeline dtype loading with a component mapping:
  default=model dtype, vae=FP32, and declared encoder names=prompt encoder dtype.
  Both callers pass their existing _frozen_encoder_names declaration. Additional
  transformers retain the model default instead of inheriting encoder precision.
  MiniMax's own builder adds audio_vae=FP32 alongside its existing post-load VAE
  handling; the shared function does not acquire a MiniMax component taxonomy.
- Keep the cross-family projection helper, model/encoder dtype owners and
  pretrained revision/offline kwargs. No new policy class or duplicate encoder
  list. This prevents loss when source weights contain FP32 detail; it cannot
  recreate precision absent from the checkpoint itself.
- Validation: 255 loader, denoise, MiniMax and Wan tests passed. A real local
  DiffusionPipeline save/load preserves VAE value 1.0001 exactly while loading
  its transformer in BF16; the old whole-pipeline BF16 mapping loses that value.
  Encoder override projection and MiniMax audio/image VAE mappings are covered.
  Initial new fixture omitted required revision; corrected before the final
  passing run. Touched-file Ruff and diff checks passed. Full review is active.

## Wan custom constructors use component precision at initial load

- Follow-up caller inspection found that Wan T2V/I2V override from_build and
  bypass diffusers_pipeline_dtypes. The preceding shared-helper change therefore
  did not fix these entry points; running Wan tests alone had not proven that.
- Both constructors now reuse their existing eager_module_dtypes mapping for
  from_pretrained and subsequent placement. The load default remains model dtype
  for both transformers, while VAE loads directly in FP32. Existing Wan encoder
  dtype behavior and CPU-offload staging are preserved.
- Keep family-owned construction/offload and component schema keys; no global
  component taxonomy or new helper/class. This corrects the earlier coverage
  implication rather than treating it as already complete.
- Validation: 53 Wan, loader and tiny-pipeline wiring tests passed. I2V offload
  assertions and a T2V constructor regression inspect the actual load mapping;
  the shared local FP32 preservation regression also passes. Touched-file Ruff
  and diff checks passed. Full repository review remains active.

## Cosmos custom loaders share component precision projection

- Predict2, Predict2.5 and Cosmos3 custom from_build methods now use the shared
  component dtype projection instead of scalar whole-pipeline loading. VAE
  loads in FP32; Predict2/2.5 encoder movement uses the same resolved encoder
  dtype as loading. Cosmos3 passes an empty encoder declaration because its
  joint transformer consumes token IDs directly.
- Cosmos3 also inherits pretrained_kwargs (including local_files_only) rather
  than separately projecting revision only. Keep family pipeline construction,
  safety-checker handling and Predict2.5's component-only skip-encoder branch;
  its VAE loader already explicitly uses FP32. No new helper or component table.
- Validation: 39 Cosmos/loader tests passed, including direct custom-constructor
  load-argument checks for FP32 VAE, encoder override and offline/revision kwargs.
  Shared real local pipeline precision preservation still passes. Touched-file
  Ruff/diff checks passed; full repository review remains active.

## Cosmos loaders preserve caller autograd mode

- Predict2, Predict2.5 and Cosmos3 from_build previously unconditionally enabled
  gradients after loading, overriding callers running under no_grad and failing
  to restore loader changes when construction raised. Scope the load operation
  with Torch's existing set_grad_enabled context using the entry mode. This
  restores thread-local state on both success and failure without a new helper.
- Keep family constructors, safety-checker scopes, component precision and the
  Predict2.5 skip-text-encoder branch. These express actual backend construction
  boundaries; reducing their number is not a goal. No new constant or taxonomy.
- Replace the old test's unconditional gradient reset with a scoped context and
  correct its description. Sixteen regression cases cover enabled/disabled entry
  modes, successful/failing loaders and all four construction paths. They use
  fake loaders that deliberately change grad mode; no model download is needed.
- Validation: 83 Cosmos and shared denoise model-base tests passed. Touched-file
  Ruff and diff checks passed. Full repository clarity review remains active.

## Chunk replay axes are declared by their producer

- Found another shape-based semantic guess in _chunk_replay_axes: any replay
  tensor beginning [B, C, S] was labelled sample/chunk/transition, even when C
  and S were coincidentally the token count and embedding width. The resolver
  then legitimately sliced those incorrectly declared axes.
- Remove that helper. The existing chunk result now carries replay_tensor_axes;
  the trajectory builder requires exactly one declaration for each extra tensor,
  requires sample alignment, and delegates axis/shape validation to the existing
  trajectory validator. Missing declarations and built-in name collisions fail
  instead of silently dropping or inferring data. Empty extras need no mapping.
- CausVid's existing trajectory_mapping owns prompt_embeds=(sample,) and
  next_sigmas=(sample, temporal_chunk, denoise_transition). The gatherer requires
  identical declarations across sample batches before using the common schema.
- Keep the result class, gatherer, family projection and public trajectory
  builders: they separate producer facts, transport and shared validation. No
  new wrapper class, independent helper, or ALL_CAPS vocabulary was introduced.
  Axis strings are existing schema keys. This changes newly constructed chunk
  trajectories; it does not reinterpret persisted trajectories or claim to fix
  full-sequence optional-cache axis metadata.
- Validation: 145 trajectory, chunk binding, CausVid, chunk replay, collector and
  trainer granularity tests passed. A collision regression selects a chunk while
  preserving the full prompt embedding; missing and inconsistent declarations
  fail explicitly. Touched-file Ruff and diff checks passed. Review remains active.

## Batch gathering preserves producer tensor dtypes

- Reproduced implicit dtype promotion in gather_replay_tensors: merging an
  int64 ID of 16777217 with a float32 batch returned float32 value 16777216.
  Transport reassembly was silently changing producer data via torch.cat.
- The existing concatenate_sample_values now rejects differing tensor dtypes
  with the field name and ordered batch index before concatenation. Route the
  full-sequence gatherer's six direct cat calls through this same function;
  chunk fields and both replay gatherers already use it.
- Keep the shared helper as a real cross-family boundary, its lazy Torch import,
  and separate gatherer classes as driver-side protocol implementations. Keep
  list/tuple concatenation and Torch's shape/device checks. No new helper,
  class, constant, or implicit cast policy; producers own intended conversion.
- Validation: 391 generation execution/binding, CausVid and trajectory tests
  passed, two optional backend tests skipped. Regressions cover unchanged values
  and dtype for homogeneous inputs, the integer-loss example, and each of the
  six full-sequence fields. Touched-file Ruff and diff checks passed after
  removing an extra import-block blank line. Full review remains active.

## Schedule construction consumes its declared config directly

- OnlineTrainer supplies RolloutOrchestrationConfig to build_rollout_schedule.
  Annotate that actual boundary and read reward_collection_mode directly rather
  than treating a missing field as an implicit default. The strict schedule test
  fixture now uses the real config and its default instead of a partial namespace.
- The factory passes config.continuous to ContinuousRolloutSchedule.from_config,
  whose parameter is now ContinuousRolloutConfig. Remove parent-object probing,
  the synthetic missing-block fallback and unnecessary bool conversion while
  projecting settings. TYPE_CHECKING imports preserve the runtime import boundary.
- Keep the factory (mode selection), class constructor (runtime projection and
  algorithm/isolation gates), and shared settings carrier (owner-loop handoff).
  Their distinct responsibilities justify these interfaces; no extra wrapper or
  config-default table is added. Existing config defaults and scheduling behavior
  for valid inputs remain unchanged. Partial duck-typed configs are not the API.
- Validation: 579 orchestration and config tests passed; touched-file Ruff and
  diff checks passed. This is an ownership/interface clarification, not a claim
  of throughput improvement or completed repository review.

## Continuous split-reward flag is validated at its config owner

- Follow-up inspection of the direct config constructor found that the bool
  annotation alone allowed split_generation_reward="false". Producer and owner
  control flow then treated the non-empty string as enabled. Removing a bool()
  projection does not by itself establish a valid boolean boundary.
- ContinuousRolloutConfig.__post_init__ now requires an actual bool. Keep direct
  field projection and downstream reads; do not spread conversions or duplicate
  guards through the scheduler/producer. No helper, wrapper, or constant added.
- Scope: direct dataclass construction; YAML/Pydantic parsing may normalize input
  before this constructor, and this change does not claim strict raw-YAML bool
  parsing. Existing True/False values and defaults remain unchanged.
- Validation: 536 continuous orchestration and config tests passed, including
  rejection of string/integer/None flags and preservation of both boolean values.
  Touched-file Ruff and diff checks passed. Repository-wide review remains active.

## Sample selection preserves batch-shared context

- Found a concrete shape collision beyond chunk axes: select_trajectory_batch
  applied _select_value to context, treating every list/tuple whose length equalled
  the sample count as sample-aligned. MiniMax exports vae_geometry as three fixed
  values; selecting two rows from a three-sample batch reduced that geometry to
  two values, incompatible with replay's three-value geometry unpacking.
- Keep context as a shallow dictionary copy during sample selection. Document
  its shared-metadata meaning on TrajectoryBatch; sample-aligned replay values
  belong in segment tensors with explicit axes. This agrees with generation's
  require_matching_batch_context contract and family context exporters.
- Correct the earlier synthetic selection test that assumed context captions/IDs
  were row metadata. It now verifies static geometry/schedules remain whole while
  sample identities and tensors are selected. A regression calls the actual
  MiniMax context exporter and verifies geometry survives the colliding count.
- Keep selection/rebuild/device helpers: they provide shared trajectory structure
  and validation across training consumers. No new context wrapper, field-name
  taxonomy or family-specific exception. This supersedes the earlier audit's
  endorsement of implicit list/tuple context slicing.
- Validation: 663 trajectory, rollout, MiniMax and online trainer tests passed;
  touched-file Ruff and diff checks passed. Full repository review remains active.

## Sample selection follows the declared axis position

- TrajectoryValidator accepts tensor axes (token, sample), but selection only
  visited tensors whose first axis was sample. A same-count row permutation
  could therefore reorder sample identities while silently retaining the old
  tensor order; a shorter selection failed only in subsequent shape validation.
- select_trajectory_batch now locates sample in each tensor's declared axes.
  The existing _select_value indexes that dimension directly instead of testing
  whether a leading length happens to equal the batch size. Nested list/tuple
  payloads preserve their container type; dictionaries recurse at the same axis.
- Keep the shared selection API, recursive helper and structural rebuild/validator
  boundary. No new wrapper or axis taxonomy; tensors without a sample axis and
  shared context remain unchanged. Other consumers' axis support is not implied.
- Validation: 651 trajectory, rollout and online trainer tests passed. New cases
  cover same-count permutations and subsets with a nonleading sample axis for
  tensors, lists and tuples. Touched-file Ruff and diff checks passed. Full
  repository review remains active.

## Sequence replay payloads validate their declared dimensions

- TrajectoryValidator previously checked dimensions only through .shape, leaving
  list/tuple replay values unchecked although the resolver supports them. Wrong
  sample counts, ragged declared step dimensions and scalar rows could therefore
  pass validation and fail later while selecting replay steps.
- Extend the existing axis-validation method to walk sequence containers only
  as far as the declared axes. Compare each declared length against that level;
  scalar values before a remaining axis fail. Undeclared trailing dimensions
  remain free to be ragged, such as sample-aligned prompt token lists.
- Keep the existing validator owner, schema-axis lookups and runtime-state
  rejection. No new helper/class/taxonomy. Nested tensors inside sequences still
  follow the prior serialization rejection; an initial new test expected a later
  shape error and was corrected to assert that existing earlier boundary.
- Validation: 804 trajectory, generation binding, rollout and online trainer
  tests passed, two optional backend tests skipped. Touched-file Ruff and diff
  checks passed. Full repository review remains active.

## Trajectory roles and segment kinds use their existing schema definitions

- TensorRole, SegmentModality and DistributionKind already define closed Literal
  vocabularies, but their dataclasses accepted arbitrary strings. Tensor role
  typos could surface as missing-role failures later, while distribution values
  route collector batch construction and evaluator selection.
- Add constructor checks in TrajectoryTensor and TrajectorySegment using get_args
  of those existing definitions, matching TrajectoryAxis.kind. No duplicated
  ALL_CAPS allowlist, external validation helper or new class. Keep the explicit
  custom distribution extension and downstream family dispatch interfaces.
- This validates newly constructed records, including dataclass reconstruction;
  it does not claim arbitrary post-construction mutation is prevented.
- Validation: 813 trajectory, generation binding, rollout and online trainer
  tests passed, two optional backend tests skipped. Regressions cover invalid
  roles/modality/distribution and preservation of custom. Touched-file Ruff and
  diff checks passed. Full repository review remains active.

## CUDA occupancy sampling distinguishes absence from failure

- cuda_occupancy_snapshot documented None for non-CUDA execution but caught all
  exceptions from CUDA queries and returned the same sentinel. A broken CUDA
  query was therefore indistinguishable from an ordinary missing reading in the
  denoise loop's memory record path.
- Remove the blanket catch. Keep the explicit CUDA-unavailable branch; query
  failures now propagate with their original exception and traceback.
- Keep this free function as a pre-generation measurement boundary. It records
  only the starting occupancy; a full BatchMemoryReading also needs peaks that
  do not exist yet. Keep the field-name check against the existing wire record,
  lazy Torch import and affine-fit owner. No new wrapper or duplicate taxonomy.
- Validation: 387 generation execution, step and binding tests passed, two
  optional backend tests skipped. CPU absence and original exceptions from all
  three CUDA queries are covered with mocked CUDA APIs; this is not a real-GPU
  fault-injection run. Touched-file Ruff and diff checks passed. Review is active.

## Combined regression and remaining Ray device-discovery ambiguity

- At e87f9fea8, ran tests/ray, tests/trainers, tests/generation,
  tests/trajectory and tests/config together: 1739 passed, 9 skipped, 16 warnings
  in 99.28 seconds. This covers the recent cleanup across those suites, including
  Ray placement tests, but does not prove full repository review or real multi-GPU
  training completion. No runtime code changed during the run.
- Independent source inspection found remaining ambiguity in generation/ray/config:
  _get_device catches every exception, while _cuda_device_index substitutes zero
  for an invalid textual ordinal. Direct reproduction confirmed cuda:broken -> 0
  and a policy.device RuntimeError -> an empty driver device set. The latter
  reaches _validate_driver_cuda_ownership's early return without checking overlap.
- Next change must distinguish a genuinely absent optional device from a broken
  property, and reject malformed CUDA ordinals instead of selecting GPU zero.
  Keep torch-free import behavior, explicit CPU handling, trainable-module fallback
  for policies without a device, and the cross-node ordinal-space exception.
  Review the device helpers together because they implement one discovery boundary;
  do not introduce a second device parser or flatten the cycle-safe module walk
  merely to reduce function count. This finding is reproduced but not fixed here.

## Ray driver device discovery reports failures instead of guessing GPU zero

- Fix the reproduced RayGenerationConfig discovery gap: _get_device now returns
  the optional device directly, preserving RuntimeError and other failures.
  For AttributeError, inspect the static declaration to distinguish a failing
  property from a genuinely absent attribute before falling back to modules.
  Remove the redundant (has_device, device) return pair from both consumers.
- The existing CUDA parser now requires cuda or cuda:<nonnegative decimal index>
  for CUDA strings. Typed CUDA indices use require_exact_int instead of int
  coercion. Invalid ordinals no longer silently become zero.
- Keep non-CUDA handling, the existing bare-cuda default, no-device trainable
  module fallback, cycle-safe module traversal and cross-node ordinal isolation.
  The helpers remain one torch-free discovery boundary; no new parser module,
  wrapper class or ALL_CAPS vocabulary. Bare cuda still means zero here; this
  slice does not claim to resolve an implicit current-device ordinal.
- Validation: 382 Ray runtime-config and config tests passed. Regressions check
  original RuntimeError/AttributeError identity, malformed CUDA text and module
  fallback without requiring GPU allocation. Touched-file Ruff/diff checks pass.
  This addresses the preceding reproduced finding; repository review stays active.

## Ray placement queries preserve their actual failure

- RayGenerationWorker.worker_metadata swallowed node/GPU query errors and
  fabricated node_ip=unknown plus gpu_ids=[]; launcher rank validation likewise
  swallowed a driver-node query error and supplied None. These values feed
  placement checks, not optional presentation metadata.
- Remove both catches so startup fails at the actual query. Keep the worker
  metadata method as the actor RPC boundary and rank validation as the shared
  startup guard; actor-group/launcher cleanup already owns startup exceptions.
  No new adapter or constant. HEALTH_CONCURRENCY_GROUP remains a real Ray
  protocol name and is unrelated to these fallbacks.
- Validation: 264 generation-Ray, RPC deadline and cross-node preflight tests
  passed, including original-error identity for both worker queries and the
  driver query. Existing metadata-timeout coverage still confirms candidate
  actors are killed. New query tests mock API failures rather than claiming
  multi-node fault injection. Touched-file Ruff and diff checks passed.
- Remaining separate concern: current_gpu_ids in dependencies.py still skips
  malformed ID values during normalization. This slice only removes exception
  swallowing at the worker/launcher call sites; full review remains active.

## Ray GPU ID normalization does not drop or truncate assignments

- Installed Ray get_gpu_ids declares List[int] or List[str], using IDs from
  CUDA_VISIBLE_DEVICES when set. The repository placement protocol stores integer
  ordinals. current_gpu_ids previously skipped failed int conversions and accepted
  lossy conversions such as 1.5 -> 1, yielding an incomplete or altered assignment.
- Retain integer/decimal-string normalization in the existing dependency adapter;
  reject malformed strings and non-integer values with their list position.
  Reuse require_exact_int after string parsing, with no new helper or vocabulary.
  Empty assignments remain valid for actors that do not request GPUs.
- GPU UUID strings remain outside this repository's integer placement protocol;
  they now produce an explicit error instead of silently disappearing. This does
  not add UUID/MIG topology support. Keep the lazy Ray import and metadata APIs.
- Validation: 52 dependency, cross-node preflight, global placement and rollout
  launcher tests passed. Cases cover integer/string/mixed valid IDs and rejection
  of UUID, malformed, fractional, boolean, None and negative IDs. Existing real
  Ray placement integration passes. Touched-file Ruff/diff checks passed; full
  repository review remains active.

## Cluster topology construction belongs to its existing record

- Move inspect_cluster into ClusterTopology.from_ray and update the preflight
  consumer, tests and fixture documentation. Remove the free constructor export;
  no compatibility forwarding function remains because no repository caller needs
  it. The existing class now owns construction of its two aggregate values.
- Remove the driver-IP query catch: previously a failed lookup classified every
  live node's GPUs as non-driver resources. Discovery now propagates the original
  failure rather than constructing a misleading topology.
- Keep cross_node_preflight independent: it compares live topology with requested
  rollout resources and owns the driver-GPU isolation check. Keep lazy dependency
  adapters and shared actor cleanup functions; grouping those by name alone would
  not improve ownership. No new class or ALL_CAPS vocabulary was introduced.
- Validation: 53 dependency, cross-node preflight, placement and rollout launcher
  tests passed, including original driver lookup error identity. Touched-file Ruff
  and diff checks passed. Full repository review remains active.

## Local and distributed OOM retry use one classifier

- Reproduced divergent retry decisions: the driver's text matcher accepted CPU,
  CUDA and HIP 'out of memory' messages; the worker's shared classifier accepted
  only CUDA text (plus Torch's typed OOM). Transport changed the retry decision.
- Extend existing is_cuda_out_of_memory to accept remote text and recognize HIP
  alongside CUDA. Remove the driver-only _is_oom_error and route batch splitting
  through the shared classifier. Plain CPU allocation errors no longer trigger
  GPU batch splitting; typed Torch OOM behavior is retained.
- Keep the shared classifier as a local/remote error boundary, and preserve the
  separate result-correlation guard because it compares returned results with
  submitted envelopes. No new error class, tag taxonomy or forwarding function.
  This unifies current fallback recognition; it does not claim all remote errors
  retain their original exception type or traceback.
- Validation: 753 generation, utility and Ray batch-dispatch tests passed, two
  optional backend tests skipped; process exited successfully after teardown.
  Added local-exception/remote-text parity cases for CUDA, HIP, CPU and non-memory
  errors. Existing allocator-message and split-retry tests pass. Touched-file
  Ruff/diff checks passed; repository-wide review remains active.

## Local batch OOM retry releases failed frame locals before cache cleanup

- run_sample_batches_with_oom_retry called empty_cuda_cache while the caught
  exception still retained the failed forward frame and its temporary tensors.
  The allocator cannot release storage whose tensor remains referenced there.
- Clear completed traceback frames only after recognizing a splittable OOM and
  before cache cleanup. Keep terminal/non-OOM exceptions untouched for debugging.
  The existing retry function still owns split ordering; use the standard-library
  operation directly rather than adding a separate cleanup helper or owner class.
- Validation: 343 generation execution/binding and Ray OOM tests passed, two
  optional backend tests skipped. A weak-reference regression confirms failed
  forward tensors are gone at the cache-cleanup callback; terminal failures retain
  diagnostic frame locals. This verifies lifetime/order on CPU, not GPU capacity
  improvement measurements. Touched-file Ruff and diff checks passed; a deliberate
  traceback-only test local is annotated for Ruff. Full review remains active.

## Startup probe clears failed trial ownership before allocator cleanup

- Probe run_trial kept failed forward frame locals while emptying the CUDA cache.
  An OOM raised during the post-forward synchronize could additionally leave its
  completed batch_result referenced by the still-running trial frame.
- Initialize the result explicitly and, after recognized OOM recovery/sync,
  discard it and clear completed traceback frames before empty_cache. Keep probe
  fitting, confirmation/bisection, pipeline-hook recovery and non-OOM propagation
  unchanged. No new helper/class or repeated policy table.
- Validation: 201 generation execution and Ray OOM tests passed. Two weak-reference
  cases verify cleanup ordering for forward and synchronize failures while the
  probe still converges to the correct simulated capacity. CUDA operations are
  mocked; these tests prove object lifetime, not physical-GPU throughput. Touched
  Ruff and diff checks passed. Full repository review remains active.

## Unindexed CUDA resolves the current device at driver validation

- Follow up the explicit limitation from the earlier device-discovery slice:
  bare cuda and torch.device('cuda') were assigned ordinal zero, which could
  miss a conflict when the driver selected a different current device.
- The existing parser now returns explicit indices directly and queries
  torch.cuda.current_device only for unindexed CUDA. Local Torch source confirms
  that API reports the selected device. Its import remains inside the runtime
  query branch; config parsing and explicit/non-CUDA handling need no CUDA query.
- Keep one parser and the existing topology guard, including cross-node rules.
  No new device wrapper, mapping table or fallback. This supersedes the earlier
  documented bare-cuda default rather than claiming it was already resolved.
- Validation: 385 Ray runtime-config and config tests passed. String and typed
  unindexed CUDA detect a mocked GPU-1 conflict; explicit GPU-0 validation proves
  no current-device query is made. Touched-file Ruff/diff checks passed. Full
  repository review remains active.

## Metric row construction validates component names before normalization

- from_step_metrics converted component_names to tuple before its existing
  validator, allowing a raw string such as ocr to become three components o/c/r.
  Validate the original Sequence first, then retain its immutable tuple form.
- Read TrainStepMetrics.phase_times and reward_components directly: both are
  declared dictionaries with defaults in the owning metric type. Remove the
  missing/None attribute fallback from this typed projection boundary.
- Keep _csv_field: its metadata binds serialization format and phase lookup to
  one column declaration, removing a real duplicate mapping. Keep the metric
  record, CSV owner and shared header/row checks; no new wrapper or name table.
- Validation: 163 metrics-IO and online trainer tests passed. Raw strings/bytes
  fail, while list/tuple containing ocr preserve one component and its value.
  Touched-file Ruff/diff checks passed. Full repository review remains active.

## Metrics resume validates the checkpoint position before touching CSV contents

- MetricsCSV only tested resume_position < 0. NaN passed and made every row's
  position < resume_position false, so alignment could discard all metrics.
  Fractional and boolean positions likewise had no valid checkpoint meaning.
- Reuse require_exact_int with a nonnegative bound before resume file creation
  or alignment. Check the declared position column before the missing-file branch
  too; previously an invalid column was accepted only when creating a new file.
- Keep initialization/resume inside MetricsCSV and retain atomic replacement,
  complete-line recovery and checkpoint-position semantics. No new free helper,
  wrapper, config vocabulary or parallel validation table.
- Validation: 176 metrics-IO and online trainer tests passed. Invalid positions
  preserve existing contents or leave absent files absent; an unknown resume
  column is rejected for a new file. Touched-file Ruff/diff checks passed. Full
  repository review remains active.

## Metrics header uses the same CSV grammar as readers

- MetricsCSV accepted quote characters in column names but joined names with
  commas directly. A literal name such as '"loss"' was read as loss by a CSV
  parser, so the initialized schema did not round-trip through the file format.
- Use csv.writer with an explicit newline to produce the header inside the
  existing initialization owner. Ordinary headers remain byte-identical. Keep
  current column validation and resume schema comparison; no wrapper function
  or expanded vocabulary. Existing incorrectly escaped quoted-name files fail
  schema matching rather than being silently interpreted as the new schema.
- Validation: 178 metrics-IO and online trainer tests passed. Standard CSV reads
  preserve both leading and embedded quotes after initialization, append and
  resume alignment. Touched-file Ruff/diff checks passed. Full review is active.

## Supervised launch construction belongs to TrainLaunch

- Move build_train_launch into the existing TrainLaunch.from_root classmethod.
  It builds exactly this record's command and expected_world_size, so its owner
  is unambiguous. Update CLI and tests; remove the old free-function entry point
  rather than adding a forwarding compatibility layer with no repository caller.
- Keep command construction, strategy routing, one-host torchrun arguments and
  multi-node rejection unchanged. Keep build_parser/_bounded_number as argparse
  adapters and the separate supervisor-owner guard as launch-environment policy.
  No new class, protocol vocabulary or configuration table.
- Validation: all 59 supervisor tests passed, covering direct launch, DDP/FSDP
  torchrun construction and multi-node rejection through the new entry point.
  Touched-file Ruff/diff checks passed. Full repository review remains active.

## Supervisor retains child ownership through health-check failures

- _run_attempt cleared its child reference in finally even when health polling
  raised while training was still running. The supervisor could therefore exit
  without stopping the process group it owned.
- On attempt-monitoring exceptions, invoke existing request_stop and reap the
  child before re-raising the original error. Annotate cleanup failures and retain
  a still-running child handle instead of unconditionally dropping ownership.
  Normal attempt completion continues to clear the finished handle.
- With that cleanup boundary in place, metrics reads now suppress only absent
  files. Other OSError failures propagate instead of pretending no metrics exist.
- Keep process-group signaling/grace escalation in RunSupervisor and numerical
  judgment in MetricsHealthGate. No new cleanup class/helper or policy vocabulary.
- Validation: 61 supervisor tests passed. A real sleeping subprocess is stopped
  and reaped when the health check raises PermissionError, preserving the same
  exception; a missing metrics file remains normal while a directory at that path
  raises. Touched-file Ruff/diff checks passed. Full review remains active.

## Health CSV reader rejects ambiguous column identity

- csv.DictReader silently overwrites earlier values under duplicate header names.
  A later loss column could therefore mask NaN in the first loss column before
  MetricsHealthGate evaluates the row.
- Validate non-empty, unique parsed column names in the existing complete-row
  reader before constructing row dictionaries. Keep dynamic reward columns and
  numerical health checks; no hardcoded schema copy or new reader wrapper.
  Reader failures use the preceding supervisor-owned child cleanup path.
- Validation: 99 supervisor and metrics-IO tests passed. Duplicate loss and an
  empty column are rejected before health evaluation, including a NaN followed
  by a healthy-looking duplicate value. Touched-file Ruff/diff checks passed.
  Full repository review remains active.

## Resume checkpoint loading belongs to TrainingCheckpoint

- Move load_training_checkpoint_for_resume into TrainingCheckpoint.load_for_resume,
  reusing cls.load when the resolved policy selects a checkpoint and preserving
  None for a fresh run. Remove the old free-function export and update both online
  and Wan DPO recipe entry points, their test seams and the run-module ownership note.
- Keep checkpoint I/O in the checkpoint module rather than config resolution.
  Keep cross-type resume/model guards and shared checkpoint discovery separate;
  they do more than construct a single record. No new class or forwarding layer.
- Validation: 147 checkpointing, online lifecycle, Wan DPO identity and DPO config
  tests passed. Repository search finds no old callable references. Touched-file
  Ruff/diff checks passed. Full repository review remains active.

## Precision metadata owns its model dtype lookup

- _model_transformer_dtype and _dtype_label served only OnlineTrainer's precision
  metadata, adding two jumps around a short projection. Move the lookup and
  formatting into _precision_metadata and remove both module-level helpers.
- Preserve the family getter / transformer dtype / parameter-source order, but
  propagate failures from an explicitly supplied getter or parameter iterator.
  Previously these errors were swallowed and replaced by another source or None.
  A parameterless model still reports unknown; first-parameter fallback remains
  the existing convention and is not claimed to describe heterogeneous models.
- Keep _precision_label for configured precision vocabulary and the existing
  metadata method as the trace owner. No new class, alias table or adapter layer.
- Validation: 145 online trainer tests passed. New cases verify query exception
  identity and parameterless unknown; existing first-step diagnostics still report
  float32. Touched-file Ruff/diff checks passed. Full review remains active.

## Share role labels and reuse the existing dtype conversion boundary

- Remove trainer._precision_label and its duplicate dtype alias table. Trainer
  metadata now uses models.dtypes.dtype_to_precision_token for evaluator math
  dtype and the guard's named normalize_role_precision_label for role labels.
- Keep the short shared role-label function: traces and guard decisions must
  agree on legacy empty/no normalization and retain quantization/autocast
  suffixes. Config normalize_precision cannot parse those composite labels.
  No new class, vocabulary table, or forwarding module is introduced.
- Configured role labels come from PrecisionRolePolicy.label through
  TrainerConfig.from_root; arbitrary torch dtype aliases are not role policy
  labels. Math dtype conversion now rejects unsupported non-plain dtypes
  instead of emitting an unchecked diagnostic token.
- Validation: 148 online trainer tests passed, including three math dtypes and
  composite rollout policy preservation. Touched-file Ruff and diff checks
  passed. This supersedes the previous decision to retain _precision_label;
  the repository-wide review remains incomplete.

## Keep prepared-weight checks at their only consuming boundary

- Move _validate_prepared_weight_snapshot into
  RolloutRuntimeCoordinator.prepare_weight_sync_state, using an explicit stack
  in the same traversal order. Remove the single-caller recursive helper.
  Detached CPU checks and accepted mapping/list/tuple nesting remain unchanged.
- Clarify ownership: the strategy getter copies live state (both strategy
  exporters call to_cpu_snapshot); this check cannot establish storage
  independence. Preparing stays on the trainer thread for DDP/FSDP collectives;
  pushing still does not call the live getter.
- Keep pipeline's stream-scoped tensor copier and continuous owner's future
  adapter: they centralize CUDA lifetime and cross-loop cancellation semantics.
  This is not a move of all module functions into classes, and no vocabulary
  constants or runtime interfaces change.
- Validation: 262 orchestration and weight-sync tests passed; touched-file Ruff
  and diff checks passed. The broader generation/scheduling audit remains open.

## Scope generation counter formatting to its metric owner

- Move the single-caller _debug_metric_value into GenerationWorkerCore's
  _batch_metrics as a local recursive counter_value converter; remove the
  redundant outer dict copy. Debug-disabled calls still return before reading
  output properties. Counter values, nesting, scalar extraction and repr
  fallback remain unchanged.
- Do not combine this with trainers.diagnostics._json_safe: trainer tensor
  diagnostics produce statistics, while generation counters retain scalar
  values. Nor does ordinary JSON file IO own tensor interpretation. Keeping
  these distinct avoids changing diagnostic semantics for cosmetic reuse.
- The recursive local function remains useful for nested mappings/sequences;
  no class, shared serializer, or constants are added. The existing item-error
  fallback is preserved, not claimed to be a strict serialization contract.
- Validation: 179 generation execution tests passed, including nested scalar
  counter conversion and disabled-debug property isolation; touched-file Ruff
  and diff checks passed. Full repository review remains incomplete.

## Make rank rendezvous integer requirements explicit

- RankGroupSpec previously accepted fractional port/rank/world-size values
  and bool ranks through numeric range comparisons. Reuse require_exact_int
  for all three fields before preserving the existing range checks. Invalid
  launch contracts now fail at construction, before distributed initialization.
- Replace the stale module claim that multi-rank backends have not landed:
  GenerationWorkerCore already initializes before model build and destroys
  the group during policy release.
- Keep init/destroy as the torch.distributed lifecycle adapter and keep
  build_rollout_schedule as the factory selecting two protocol implementations;
  neither belongs on one concrete schedule or on serialized rendezvous data.
  The topology guard remains cross-type. No new wrapper class or constants.
- Validation: 30 rank-group, rollout-launcher and sequence-parallel tests passed,
  including real two-process gloo communication and 18 invalid-field cases.
  Touched-file Ruff and diff checks passed. No real NCCL validation is claimed;
  repository-wide clarity review remains incomplete.

## Release only the rank process group acquired by this worker

- GenerationWorkerCore previously used presence of RankGroupSpec as proof of
  process-group ownership. If init rejected an already initialized group,
  load_policy's cleanup destroyed that pre-existing group anyway.
- Track successful init explicitly in the worker and clear ownership only
  after successful destroy. A model-build failure still destroys the acquired
  group; rejected init and repeated release leave another owner's group intact.
- Keep init/destroy as the distributed framework adapter. Ownership belongs
  to the existing worker, not to the immutable rendezvous spec; no wrapper
  class or additional helper is introduced. Preserve memory-release ordering
  and its quarantine behavior. Partial distributed-init failures and external
  replacement of an owned default group are not addressed by this change.
- Validation: 207 generation execution/launcher tests passed, including the
  rejected-init and post-init model-failure cases, each followed by repeated
  release with an externally initialized group. Touched-file Ruff/diff checks
  passed. The full repository review remains open.

## Put diffusion result row declarations in the gatherer

- Remove DiffusionRequestLayout.ordered_batches, a single-caller static
  forwarding method on the request parser. DiffusionBatchGatherer now calls
  ordered_covering_batches directly with its six row-bearing result fields.
  Drop the gatherer's runtime import of the request parser and remove the
  parser's unused result typing/imports. Ordering and validation are unchanged.
- Keep ordered_covering_batches, replay gathering, row checks and static-value
  comparison shared across the three generation bindings: their consistency
  prevents different coverage or replay semantics between families. The chunk
  gatherer's ordering method adds actual family checks and is not equivalent
  to the removed forwarding method. No new class or constant table is needed.
- Validation: 188 binding and batch-gatherer tests passed, two skipped; touched
  Ruff/diff checks passed and the removed method has no remaining code/test
  references. The full repository review remains incomplete.

## Own encoded tensor expansion in the executor

- Remove DiffusionRequestLayout.repeat_batch and its torch import. The existing
  DiffusionBatchExecutorBase.build_batch_encoded now expands singleton tensor
  rows directly, preserves already-sized tensors, scalar/non-tensor values and
  declared batch_passthrough_keys, and names a mismatched encoded field in errors.
- Cosmos Predict2 also called the old method. It now delegates text embedding
  preparation to the base executor and retains its reference-image selection.
  Keep family passthrough declarations: FLUX text IDs do not have a sample axis.
- Keep SDE window selection in request parsing, shared by every sample batch.
  No replacement repeat helper/class or vocabulary table is introduced.
- Validation: 366 binding/execution/Cosmos Predict2 tests passed, two skipped;
  after adding explicit Cosmos positive/negative embedding checks, all 93 layout
  tests passed. Touched-file Ruff/diff checks passed; no repeat_batch references
  remain in code/tests. Full repository review remains open.

## Inherit single-sample encoded preparation instead of overriding it

- Remove Cosmos3 and MiniMax-H3 build_batch_encoded overrides that only copied
  the encoded mapping. The base executor already preserves their supported
  inputs: token-ID lists and scalar guidance, or singleton prompt embeddings
  plus max_text_tokens. Preserve their family-specific prompt encoders.
- Keep ReferenceConditionedBatches: it shares real image loading and argument
  threading for Wan/Cosmos, not an empty override. Update its stale docstring
  and the generic executor description to reflect current ownership. No
  capacity changes, new configuration table, or per-family wrapper is added.
- Validation: 171 generation binding and MiniMax-H3/Cosmos3 model tests passed,
  two skipped. Explicit single-sample cases verify mapping copies retain value
  identity through both concrete family executors. Touched-file Ruff/diff
  checks passed. Full repository review remains incomplete.

## Construct complete diffusion sampling parameters once

- Fold the sole production use of select_sde_window into parse_sampling_params.
  Validate max_sequence_length before consuming RNG, resolve the stochastic
  window, then construct DiffusionSamplingParams once. Remove the provisional
  params object, dataclasses.replace and the separate selection method.
- Preserve the exact seed XOR stream, inclusive randint bounds and unseeded
  module RNG. The request parser remains the owner; window selection stays
  outside per-batch execution so all samples share the request window. No new
  helper, class or configuration constant is introduced.
- Validation: 347 binding/execution tests passed, two skipped. The updated
  parser-level regression checks one unseeded draw and no draw for disabled
  windows; existing seeded reparse and request-window tests also pass. Removed
  method has no remaining code/test references; touched-file Ruff/diff checks
  passed. Full repository review remains incomplete.

## Open defect: unseeded SDE windows are not request-owned

- Follow-up call-chain review contradicts earlier "once per request" claims:
  DiffusionBatchExecutorBase._forward_chunk calls parse_sampling_params for
  every batch, including retries. There is no request-level params cache.
  Seeded parsing reproduces a window; unseeded parsing consumes fresh RNG.
- Reproduced with the same GenerationRequest, window size 2/range (0, 10),
  and controlled module draws 0 then 8: successive parses yield (0, 2) and
  (8, 10). Adding seed 1234 yields (8, 10) on both parses.
- Correct misleading layout/executor comments now. The prior construction
  simplification preserved this pre-existing behavior; its tests proved one
  draw per parse, not one draw per request across batches.
- Required follow-up: establish request-owned stochastic-window state at a
  boundary serialized to workers, then verify splits, retries and multi-rank
  execution agree. A local executor cache alone cannot cover remote dispatch;
  deriving randomness from a correlation ID would introduce new semantics.
  Preserve the seeded sequence and avoid changing latent-noise seeding as an
  accidental side effect. No runtime change or full completion is claimed here.
- Validation: executed the controlled reproduction above and checked touched
  comments with Ruff/diff. This finding supersedes earlier audit wording that
  all sample batches already share the same unseeded request window.

## Resolve unseeded window ownership on GenerationRequest

- Close the unseeded reparse defect above: GenerationRequest now creates a
  sde_window_seed once when a window is enabled and sampling.seed is absent.
  This field crosses the existing request/envelope boundary and survives
  dataclasses.replace, pickle and cloudpickle. The parser derives its window
  from this value instead of worker-local global RNG.
- Preserve explicit sampling.seed and its existing XOR/random stream exactly.
  The fallback seed is not inserted into sampling, so latent-noise seeding
  remains unchanged. No new YAML knob, cache owner, helper or request-ID-based
  random convention is introduced. Disabled windows create no random seed.
- Keep window bounds resolution in the parser, where num_steps is validated.
  Missing fallback state after mutating an already-built request fails explicitly
  instead of reverting to per-worker random draws. Requests must establish
  their denoise options at construction or via dataclasses.replace.
- Validation: 1152 generation/rollout tests passed, two skipped. After adding
  the serialized-envelope OOM split regression, all 96 layout tests passed.
  Tests exercise independent serialized request copies, preserved seeds across
  width replacement, three identical windows across failed batch and split
  retries, and unchanged absent sampling.seed. These are CPU serialization/
  orchestration checks, not a multi-GPU training run. Touched-file Ruff/diff
  checks passed; full repository review remains incomplete.

## Align request version identity with weight installation

- GenerationRequest.policy_version previously rejected only negative values;
  bools, fractional values and NaN passed that boundary. True and 1.0 can
  compare equal to installed integer version 1 in worker version checks.
- Reuse require_exact_int with minimum zero in the request, matching launch
  contracts, weight installation and staleness validation. None remains the
  explicit unspecified-version case. No conversion, fallback, helper class
  or new vocabulary constant is introduced.
- Keep the validation in the request data boundary and the shared primitive
  in utils.config: both have existing consumers and clear ownership.
- Validation: 462 execution/orchestration tests passed, including invalid
  request versions through dataclasses.replace and preserved None/zero/positive
  versions. Touched-file Ruff/diff checks passed. Full review remains open.

## Initialize the rank group before RNG collectives

- execute_batch called _synchronize_rank_rng before load_policy, although
  load_policy owns rank-group initialization. A cold multi-rank worker could
  therefore query/broadcast through an uninitialized default group. Move RNG
  synchronization after successful loading, also keeping failed model builds
  out of this collective. No new helper or state is introduced.
- Keep _synchronize_rank_rng: it owns the shared multi-rank RNG operation.
  Keep the two execution entrypoints' version-result adaptations distinct:
  batch execution returns wire errors while pipelined execution raises typed
  errors. Their similar text alone does not justify a new adapter class.
- Validation: 220 execution/launcher tests passed. A cold-load failure regression
  verifies init/build/cleanup occur without entering the RNG collective.
  Touched-file Ruff/diff checks passed; no multi-GPU training claim is made.
  Repository-wide review remains incomplete.

## Synchronize RNG for multi-rank pipelined execution too

- Pipelining is restricted to one engine, not one rank. RayGenerationExecutor
  dispatches the whole request through engine.remote to its ranks, but the
  worker's pipelined entry omitted the shared RNG synchronization used by
  execute_batch. Add that same operation after load_policy.
- Keep the existing RNG method as a common multi-rank boundary; single-rank
  requests remain no-ops there. No new helper or pipeline-specific seed state.
- Extend the real two-process gloo test with the worker pipelined entry and a
  tiny CPU producer: deliberately different rank-local Python/Torch seeds
  yield identical gathered output on two successive calls. Existing isolated
  pipeline fakes now explicitly declare rank_group=None, matching constructor
  state rather than requiring a production fallback for incomplete fixtures.
- Validation: 220 execution/launcher tests passed; touched-file Ruff/diff
  checks passed. This verifies gloo RNG coherence, not NCCL GPU pipelining
  performance or a full model run. Repository review remains incomplete.

## Keep batch-probe search state inside the probe call

- Replace GenerationWorkerCore._bisect_batch_probe with a local bisect_capacity
  inside probe_batch_size. Its trial function and accumulated trial list belong
  to that invocation; the two search sites now pass only their good/bad bounds.
  Remove the unused Callable import. Search order and returned capacity remain
  unchanged, with no new abstraction or copied implementation.
- Keep probe tuning constants: they are a deliberately fixed, isolated policy
  used by both probing branches and test fixtures, not an algorithm vocabulary.
- Validation: 212 execution tests passed, including probe OOM/bisection paths;
  touched-file Ruff/diff checks passed and no old helper references remain.
- Open multi-rank issue from source review: RayGenerationExecutor fans a whole
  probe call to all engine ranks. Each worker independently chooses fit, confirm
  and bisection trials from local memory/timing/OOM results. No coordinated
  trial decision is present, so differing measurements can produce different
  collective shapes/control flow. This is source evidence, not a reproduced
  GPU deadlock. Adding RNG sync alone cannot establish collective safety; the
  probe needs coordinated decisions or an explicit supported-topology boundary.
  Full repository review remains incomplete.

## Declare the supported automatic-probe topology before dispatch

- Automatic probing now rejects multi-rank engines instead of running
  independently chosen collective trial shapes. RayGenerationExecutor checks
  the complete fleet before dispatching anything; GenerationWorkerCore rejects
  direct multi-rank probe calls before CUDA/model work. These are the driver
  dispatch and direct worker API boundaries, not duplicate policy classes.
- Preserve multi-engine probing when each engine has one rank. Multi-rank
  generation with an explicit samples_per_generation_batch is unchanged.
  No fallback batch size is guessed and no new config switch is introduced.
- This contains the unsafe unsupported path; it does not implement coordinated
  multi-rank probing. That feature needs shared trial decisions plus rank-failure
  handling and remains a distinct unimplemented capability.
- Validation: 476 execution/Ray tests passed. New cases verify zero dispatch
  even when a single-rank engine precedes an unsupported engine, and direct
  rejection before CUDA queries. Existing real fleet auto-probe tests pass.
  Touched-file Ruff/diff checks passed. Full clarity review remains incomplete.

## Open defect: generation error payloads disappear in rank aggregation

- EngineCallRef awaits every rank but, without combine, returns results[0].
  Generation execution uses that default. Worker execute_batch returns some
  failures as GenerationBatchResult.error/stale_slot; the pipelined entry
  returns CUDA OOM as PipelinedRequestOutOfMemory. Neither is a raised exception.
- Reproduced with completed awaitables: rank0 batch success plus rank1
  error='decode failed' yields rank0 with error=None. Rank0 successful value
  plus rank1 PipelinedRequestOutOfMemory yields rank0 success.
- Correct the engine module's stale single-rank-only description and clarify
  that generic aggregation only checks raised exceptions. Runtime behavior is
  unchanged in this evidence commit.
- Follow-up must define generation-specific aggregation: preserve non-primary
  error identity, prioritize terminal errors over retryable OOM, preserve stale
  discard semantics, and retain primary output only after all ranks succeed.
  Cover normal batch dispatch, flexible scheduling, OOM child retries and
  pipelined requests; weight-version echo already uses an explicit uniform
  combiner and should retain that separate protocol. Do not solve this with a
  global method-name vocabulary table.
- Validation: executed both controlled reproductions; touched-file Ruff/diff
  checks passed. Full repository review remains incomplete.

## Preserve non-primary generation failures during engine aggregation

- Add generation-specific combiners on RayGenerationExecutor and attach them
  to every remote batch path (fixed/flexible dispatch and OOM child retries)
  and pipelined request dispatch. The generic EngineCallRef remains a neutral
  awaitable aggregator; weight updates keep their uniform echo combiner.
- Batch aggregation preserves terminal error before stale discard before OOM,
  then primary success. Validate returned types/request/batch identity and
  agreement of successful policy versions. Pipeline aggregation preserves any
  rank's typed OOM and checks request identity. Raised remote exceptions retain
  existing cancellation behavior. The pipeline's rank correlation gate now
  accepts any actual member rank, not only rank 0, without rewriting identity.
- These methods own genuinely shared result-protocol rules; no free helper,
  method-name dispatch table or new result wrapper is added. Single-rank
  EngineCallRef bypass remains unchanged.
- Validation: 481 execution/Ray tests passed. Additional identity-disagreement
  regressions brought the focused engine suite to 19 passing tests. Controlled
  multi-rank awaitables verify non-primary batch failures/stale/OOM, pipeline
  OOM, error precedence and primary selection after success. All remote call
  sites were inspected for combiner wiring. Touched-file Ruff/diff checks
  passed. No multi-GPU model run is claimed; full review remains incomplete.

## Verify non-primary errors through real Ray dispatch and retry

- Extend the existing capacity-worker fixture with an actor-readable dispatch
  history and exercise two real Ray actors through RayGenerationExecutor,
  RayGenerationEngine and the production actor dispatcher. No production
  code or protocol changes in this validation slice.
- Rank0 accepts eight samples while rank1 accepts two. The driver's request
  degrades to four two-sample outputs; all split diagnostics retain rank1, and
  both actor histories contain the same seven original/child dispatches.
  A separate rank1 decode failure reaches the caller with only the initial
  dispatch, proving terminal errors are not converted to OOM retries.
- The small fixture method is necessary to inspect actor-owned state across
  the actual process boundary. Existing OOM message constants remain fixture
  vocabulary, not production policy tables.
- Validation: 24 OOM-split tests passed, including both real Ray cases;
  touched-file Ruff/diff checks passed. Actors use CPU scripted capacity, so
  this verifies transport/admission/aggregation/retry, not CUDA collective
  recovery after a real rank OOM. Full repository review remains incomplete.

## Validate published policy versions without truncation

- RolloutRuntimeCoordinator.current_policy_version used int(value), turning
  fractional/bool/string provider values into apparently valid request versions
  before request and staleness checks could reject them. Reuse require_exact_int
  with minimum zero and preserve the provider's integer value.
- Keep runtime-then-syncer resolution: the syncer can publish before the
  collector attaches its runtime. None means no version published; an invalid
  version raises instead of falling through to another provider. This existing
  lifecycle method is the shared owner, requiring no extra validator class.
- Validation: 411 orchestration/online trainer tests passed. New tests exercise
  both provider paths and reject malformed versions even when a valid fallback
  exists; existing attach/versionless cases preserve their semantics.
  Touched-file Ruff/diff checks passed. Full review remains incomplete.

## Own collector section projection in its constructor

- Fold _merge_flat_section_values into RolloutCollectorConfig.from_root as one
  loop over rollout and sampling sections. Remove the single-owner external
  helper and unused ConfigBase typing import. Preserve rollout-before-sampling
  order, explicit-field filtering, nested-block exclusion and duplicate-owner
  errors. No new schema vocabulary or wrapper class.
- Keep _DENOISE_OPTION_FIELDS: it is derived from the denoise dataclass schema,
  not a hand-maintained algorithm list. Keep the optional non-draining runtime
  capability lookup: GenerationRuntime currently does not require that member,
  and absent support must retain the draining barrier.
- Validation: 793 rollout/config tests passed; touched-file Ruff/diff checks
  passed and no removed helper references remain in code/tests. Full repository
  review remains incomplete.

## Give collection timing reduction to RolloutStats

- Move generation/reward duration and overlap reduction into existing
  RolloutStats.add_collection_timing. The collector supplies measured intervals
  and total wall time; the stats owner records the same four collect.* phases.
  Remove the single-caller external _interval_overlap_seconds function.
- Keep per-call interval measurement and scheduling in the collector, and keep
  the ordered non-overlapping timeline assumption (one generation/one scoring
  task). Metric names and additive aggregation remain unchanged. No new stats
  wrapper class or metric-key vocabulary table is introduced.
- Validation: 457 rollout tests passed, including serial-zero-overlap, streaming
  overlap and stats accumulation cases. Touched-file Ruff/diff checks passed;
  no removed helper references remain. Full review remains incomplete.

## Scope reward timing validation to its accumulation operation

- Move the single-owner _timing_value function into fold_reward_timing as
  a local validator. Preserve normalization and complete validation before
  mutating counters, latency samples or accumulated timing values. No public
  interface change or new class.
- Keep _sum_optional shared between fold_reward_timing and merge: both need
  identical None-aware accumulation. The primitive keyword interface avoids
  coupling stats to reward runtime types; do not replace it with a permissive
  untyped mapping adapter solely to shorten the collector call.
- Validation: 36 stats/prompt-collection tests passed; touched-file Ruff/diff
  checks passed and no old helper references remain. Full review remains open.

## Protect standard reward timing columns from extension overwrite

- extra_ms names previously needed only an _ms suffix. Values named latency_ms,
  queue_wait_ms, inference_ms or latency_p50_ms/latency_p95_ms overwrote the
  standard flattened reward timings, including computed percentiles. Reject
  those collisions during fold_reward_timing validation before any mutation.
- The five names are the actual fixed timing output schema, not a business
  vocabulary; keep this explicit boundary local to the existing stats owner.
  Custom phase names and output names remain unchanged. No new helper/class.
- Validation: 37 stats/prompt-collection tests passed. The new regression
  checks all five collisions leave the full accumulator unchanged and retain
  its computed P95. Touched-file Ruff/diff checks passed. Full review stays open.

## Name ready-queue clearing according to its actual lifecycle

- Rename ContinuousRolloutQueue.close to clear and update its sole owner call.
  The method empties receipts and byte accounting; it never closes admission
  or makes the container unusable. This distinguishes it from generated
  capacity.close, which does enforce terminal state. No compatibility alias.
- Correct the queue put docstring to cover the installed current/prefetched
  batch window. Preserve FIFO/capacity rules, identity removal and consumer-owned
  staleness decisions. The queue remains a meaningful storage owner; no new
  class or constant is introduced.
- Validation: 200 continuous orchestration tests passed; touched-file Ruff/diff
  checks passed and no rollout queue.close callers remain. Full review remains
  incomplete.

## Retain pipeline owners until stop succeeds

- _ContinuousOwnerRuntime._stop_pipeline cleared producer/queue/consumer
  references before awaiting producer.stop. A stop exception caused subsequent
  shutdown attempts to skip the producer and close the collector prematurely.
- Clear those references and installed-batch markers only after producer stop
  and queue clear succeed. Keep the existing shutdown retry owner and bounded
  producer-stop semantics; no new state flag, wrapper or helper.
- Validation: 201 continuous orchestration tests passed. A real owner-thread
  regression injects one stop failure, verifies collector shutdown has not run
  and producer state remains observable, then confirms the next shutdown stops
  the same producer before closing the collector exactly once. Touched-file
  Ruff/diff checks passed. Full repository review remains incomplete.

## Consistent finite producer wait budgets

- Producer stop silently clamped negative budgets to zero and accepted infinity;
  prompt-batch drain also accepted non-finite budgets. Both undermined their
  bounded-wait contract, whereas the adjacent consumer already uses
  require_timeout. Reuse that shared validator before touching producer state
  or admitting work, and remove the redundant conversion/clamp at the wait.
- Zero is now rejected alongside negative and non-finite values. Inspected
  production callers and existing tests use positive budgets. Keep cooperative
  cancellation, timed abandonment, and weight-sync drain behavior unchanged for
  valid budgets. Keep require_timeout as a shared consistency boundary; no new
  helper, class, or constant is needed. Broader timeout API redesign is outside
  this change.
- Validation: 209 continuous orchestration tests passed, including eight cases
  proving invalid stop/drain budgets leave active generation running and able
  to finish scoring. Touched-file Ruff and diff checks passed. The full repository
  clarity review remains incomplete.

## Executor-owned result identity validation

- Move the module-level _require_correlated_result into RayGenerationExecutor
  as _validate_result_identity. Only its initial dispatch and OOM retry paths
  use this check. Remove the unused envelope return and type the retry envelope
  map as GenerationBatchEnvelope instead of Any. Keep one implementation for
  both paths, with the same validation order and error messages.
- Preserve rel_l1 as a shared numerical API used by TeaCacheState and the offline
  drift probe. Preserve the pipelined D2H helper, whose recursive tensor mapping
  and stream lifetime handling remove real complexity. Do not change retry,
  dispatch, copy behavior, or move shared helpers merely to reduce line count.
- Validation: all 272 generation/Ray tests passed, including result request-id
  rejection and OOM retry coverage; touched-file Ruff and diff checks passed.
  No old helper references remain in source or tests. Full review is incomplete.

## Driver ownership includes training roots even with a declared model device

- Reproduced two missed overlaps: a model reporting CPU or cuda:1 suppressed
  discovery of a training root on rollout GPU cuda:0. The old early return used
  trainable modules only as a fallback. Earlier device-discovery review entries
  did not establish that the primary device covers all training roots.
- RayGenerationConfig.validate_driver_state now collects both model and training
  root devices. Remove its private single-caller _driver_cuda_devices helper and
  annotate the boundary with RuntimeBundle, whose model/trainable_modules fields
  are required. Update two memory-guard fakes to include those fields.
- Keep _get_device for declared-property error handling, _iter_parameter_devices
  for cycle-safe traversal, and _cuda_device_index for shared address parsing.
  Keep cross-node ordinal isolation and approved colocation behavior. No new
  wrapper class or constants; exhaustive model buffer/storage discovery is not
  claimed by this change.
- Validation: the new cases failed before the fix (two missed exceptions), then
  55 runtime-config/memory-guard tests passed. After preserving the original
  absent-device case in the parameterization, all three focused cases passed
  again. Touched-file Ruff/diff checks passed. Full repository review remains
  incomplete.

## Launcher-owned startup helpers and explicit partial ownership

- Move rank GPU validation, fleet slot-capability probing, and rendezvous-port
  selection into RayGenerationLauncher. Each serves only its launch path; tests
  now address the owning class. Preserve shared require_actor_gpu_ids and
  get_ray_refs boundaries rather than duplicating placement/deadline logic.
- Replace the locals()-membership cleanup check with an explicit optional
  actor_group initialized before launch. Cleanup still runs only after launch
  returns an owned group; failures inside RayActorGroup.launch remain its own
  responsibility. No additional owner class or capability state is introduced.
- Rename _free_port to _find_rendezvous_port and correct its claim: it samples
  a driver-local ephemeral port, then closes the socket. It does not reserve
  that port or prove availability on a remote engine node. Rendezvous allocation
  behavior is unchanged; remote allocation and reservation are not solved here.
- Keep worker health protocol constants and generic placement/timeout helpers.
  Preserve capability reduction, RPC failure propagation, actor cleanup, and
  placement-group ownership instead of shortening away those boundaries.
- Validation: 66 runtime-config, rollout-launcher, and operation-deadline tests
  passed, including capability failures and cleanup. Touched-file Ruff/diff
  checks passed. Full repository review remains incomplete.

## Weight-sync owner contains its ACK validation

- Move _require_installed_policy_version into RayGenerationWeightSync. Its three
  consumers are the owner's local, remote whole-state, and bucket-broadcast
  paths. Preserve one validation implementation, strict ACK rejection, and
  engine-specific diagnostics; no compatibility wrapper or new class.
- Correct the module's synchronous description and the ACK ordering test's
  outdated gather explanation. RayActorDispatcher sorts result pairs by
  job_index; that contract supports pairing ACKs with engines even when
  completion order differs.
- Keep weight_manifest and chunk/bucket iterators as sender wire-format APIs,
  and StagedWeightTransfer as the receiver's buffer/integrity owner. They do
  not require a shared wrapper class. No wire-format, bucket-size, retry,
  admission, or cancellation behavior changes.
- Validation: 47 weight-sync and weight-transfer tests passed, including real
  Ray wrong-ACK attribution, bucket transfer, and abort coverage. Touched-file
  Ruff/diff checks passed. Full repository review remains incomplete.

## Name weight payloads separately from transport references

- The generation protocol, runtime pending install, session, syncer and worker
  called their payload state_ref even though serialization occurs only in the
  syncer and Ray dereferences the argument before the worker receives it. Rename
  the production chain to trainable_state and the actual ray.put result to
  shared_state_ref. Update the pending-install assertion accordingly.
- Clarify the protocol and pending-install documentation. Preserve Any because
  this change does not redefine supported payloads; preserve None semantics,
  bucket transport, version publication, and deferred-install lifetimes.
- No compatibility alias, new class, helper, or constants. This is a parameter
  naming change: external keyword callers must use trainable_state; repository
  searches found no state_ref keyword callers remaining. Test-double positional
  parameter names need not be rewritten to establish this production boundary.
- Validation: 944 generation and continuous-orchestration tests passed, two
  skipped; touched-file Ruff/diff checks passed. Coverage includes real Ray
  dereferencing and deferred-install lifecycle behavior. Full review remains
  incomplete.

## Continuous rejection reports capability instead of inventing algorithm facts

- ContinuousRolloutSchedule.from_config interpreted every false staleness
  capability as a likelihood-free objective without importance correction, and
  recommended a GRPO-family algorithm. VGRPO explicitly rejects stale samples
  because its refreshed behavior adapter does not identify the generating
  policy; its ratio is not sufficient to establish stale-sample support.
- Report tolerates_off_policy_staleness=False and the actionable strict_on_policy
  setting. Remove family-wide claims from the factory documentation and the
  algorithm-specific reason from the generic schedule. Admission behavior stays
  unchanged; the algorithm continues to own its capability and explanation.
- Keep build_rollout_schedule as the protocol implementation factory and
  validate_rollout_schedule_topology as a cross-type guard. Their independent
  function form serves real boundaries; do not introduce wrapper classes or
  per-algorithm tables into orchestration.
- Validation: all 272 orchestration tests passed, including the false-capability
  rejection with its updated diagnostic. Touched-file Ruff/diff checks passed.
  Full repository review remains incomplete.

## AR layout owns shared token padding

- Move right_pad into the existing ARRequestLayout as a static method. Its
  consumers are layout.align_pair and the shared executor tokenizer adapter;
  remove the module-level export and update both callers without an alias.
- Clarify that target_length is a minimum width, not a truncation request, and
  that added attention-mask entries are zero. Tensor contents, dtype/device,
  and the no-op behavior for sufficiently wide inputs stay unchanged.
- Keep the tokenizer adapter: Janus, NextStep and LlamaGen share its translation
  from max_text_length to the padding operation. Emu3 also consumes pair alignment.
  Preserve this cross-family interface rather than duplicating padding code.
  No new class, schema table, or truncation policy.
- Validation: 12 token-binding tests passed with two skips; 145 Janus, NextStep,
  LlamaGen, Emu3 and Janus-R1 wiring tests passed. Touched-file Ruff/diff checks
  passed. Full repository review remains incomplete.

## AR gatherers call the shared coverage boundary directly

- Remove ARRequestLayout.ordered_batches, which forwarded all arguments without
  adding behavior. The discrete AR, Janus-R1, and NextStep gatherers now call
  ordered_covering_batches directly, matching the diffusion gatherer.
- Remove ARBatchPayload and its TypeVar/export: their only consumer was that
  forwarding method. The shared BatchResultWithIdentity protocol still defines
  the actual coverage boundary. Update the OOM test's stale helper reference.
- Preserve layout.cat_batch_fields, shared by three gatherers, and all family
  trajectory assembly. Preserve row-field schemas, sorting, range checks and
  exact sample coverage; no new class, field vocabulary, or compatibility shim.
- Validation: 319 binding, sample-batch, AR-family and Janus-R1 wiring tests
  passed with two skips. Touched-file Ruff/diff checks passed; no removed
  protocol or layout forwarding callers remain. Full review is incomplete.

## Janus-R1 owns refinement selection and segment assembly

- Move _resolve_refine_mode into JanusProR1BatchExecutor and the segment
  concatenation helper into JanusProR1GenerationBatchGatherer. Each has one
  production consumer in its owner. Rename _cat_segment_extra to
  _concatenate_segments and the corresponding local to segments: these are the
  full trajectory segments, not incidental extras.
- Preserve refinement-policy mapping, concatenation order, optional log-prob
  handling, metadata behavior, and unknown-segment warning. This relocation
  does not establish stronger cross-batch metadata validation.
- Keep JANUS_R1_SEGMENTS as the model/runtime shared trajectory schema. Keep
  janus_config_from_build as the existing family build adapter; preserve the
  cross-family shape rather than adding a configuration wrapper class. No
  new constants, owner objects, or compatibility forwarding helpers.
- Validation: 26 Janus family and R1 rollout wiring tests passed; touched-file
  Ruff/diff checks passed and old free-helper calls are gone. Full review
  remains incomplete.

## Janus-R1 rejects incompatible segment metadata across batches

- Public gatherer regressions reproduced three silent losses: later batches'
  visual/cfg flags were replaced by the first batch's values, and a first-batch
  None log-prob field discarded later batches' populated log-probs. All three
  cases previously returned a trajectory without rejecting the disagreement.
- Before concatenation, reuse require_matching_batch_context to compare visual,
  cfg and log-prob presence for every segment. Add the segment name to mismatch
  diagnostics while preserving the original cause. No new helper or validator
  class; these local keys describe the existing segment schema/presence contract.
- Preserve valid tensor concatenation and all-absent log-prob handling. This
  strengthens the previous ownership-only change; it does not claim exhaustive
  validation of segment names, tensor row counts or dtypes. JANUS_R1_SEGMENTS
  remains the shared model/trajectory schema.
- Validation: 29 Janus family and R1 wiring tests passed, including the three
  regressions that failed before the change. Touched-file Ruff/diff checks
  passed. Full repository review remains incomplete.

## Byte estimation owns its traversal state locally

- Keep trajectory_tensor_bytes as the shared generation telemetry and continuous
  capacity API. Move its sole private recursive helper into that function, with
  one captured seen set and one lazy Tensor import per call. Remove the repeated
  seen argument without changing traversal order or estimated-byte arithmetic.
- Clarify object-identity deduplication: distinct tensor views can share backing
  storage, so this estimate is not unique allocated storage or peak GPU usage.
- Keep TrajectoryStoragePolicy separate from estimation and preserve its
  map_tensor_tree conversion boundary. Keep _VALID_DEVICES/_VALID_DTYPES because
  they derive legal schema values from Literal declarations, not algorithm data.
  No new wrapper class or changes to capacity/admission policy.
- Validation: 125 trajectory, generated-capacity, and ready-queue tests passed;
  touched-file Ruff/diff checks passed. Full repository review remains incomplete.

## AR field concatenation uses the shared dtype boundary

- ARRequestLayout.cat_batch_fields used torch.cat directly, unlike diffusion's
  concatenate_sample_values. A regression with int64 and float64 token fields
  reproduced silent dtype promotion; the integer 2**53+1 cannot survive that
  conversion exactly. Route AR fields through the existing shared concatenation
  function, retaining field-specific error names and valid tensor behavior.
- Keep the AR field-list adapter because multiple family gatherers consume it.
  Keep ChunkAutoregressiveDenoiseGatherer._ordered_batches: beyond shared sample
  coverage it validates temporal chunk count, trainability, replay axes and
  transition count. It is not a redundant forwarding method. No new helper,
  class or dtype table; cross-family consistency is the purpose of this change.
- Validation: the new dtype-mismatch case failed before the fix; 342 gatherer,
  binding, AR-family and R1 wiring tests passed afterwards with two skips.
  Touched-file Ruff/diff checks passed. Full repository review remains incomplete.

## Janus-R1 segment tensors share dtype-safe concatenation

- Six public gatherer regression cases reproduced implicit dtype promotion in
  segment token IDs/log-probs/masks, prompt embeddings and attention masks. The
  preceding AR field fix covered top-level payload fields, not these nested
  Janus-R1 segment tensors.
- Route nested fields through concatenate_sample_values, reporting segment and
  field names. Replace repeated required-field cat blocks with one local schema
  tuple; keep optional log-prob handling explicit. No new helper or owner class.
- Preserve matching metadata checks, optional absence, tensor values for valid
  homogeneous batches, and JANUS_R1_SEGMENTS. The local tuple describes concrete
  trajectory fields rather than a workflow-owned algorithm taxonomy.
- Validation: six added dtype cases failed before the fix; all 80 Janus family,
  R1 wiring and batch-gatherer tests passed afterwards. Touched-file Ruff/diff
  checks passed. Full repository review remains incomplete.

## Janus-R1 validates the segment set of every batch

- First-batch names previously selected which segments survived gathering; only
  that batch's schema generated a warning. Missing first-batch or extra later
  segments could be silently omitted, while other mismatches surfaced as an
  incidental KeyError. Four public gatherer cases reproduced these gaps.
- Check each batch against the existing JANUS_R1_SEGMENTS schema before any
  concatenation and report its ordered index and actual/expected names. Remove
  the warning-and-continue path and now-unused module logger. This supersedes
  earlier ownership-only entries that deliberately retained the warning.
- Preserve valid first-batch ordering, tensor concatenation and metadata checks.
  Keep the shared schema constant; no duplicate field taxonomy, new helper, or
  wrapper class. Tensor row-count validation remains a separate concern.
- Validation: all 39 Janus family and R1 wiring tests passed, including four
  schema regressions that failed before the fix. Touched-file Ruff/diff checks
  passed. Full repository review remains incomplete.

## Janus-R1 validates per-batch segment row ownership

- Reproduced a public gatherer accepting two one-sample batches whose final-image
  segments held zero and two rows respectively. The combined row count passed,
  but sample provenance was wrong. Validate each segment tensor against its own
  batch.sample_count before concatenation, allowing only optional log-prob None.
- Promote the existing _require_rows implementation to require_sample_rows and
  reuse it from coverage, replay and Janus-R1 gathering. This is one shared
  validation boundary, not a new helper implementation. Reuse the local tensor
  schema tuple for validation and concatenation.
- Preserve valid segment ordering, dtype checks and optional-field consistency.
  No model-family taxonomy, extra owner class or compatibility alias.
- Validation: the shifted-row regression failed before the fix; all 110 batch
  gatherer, sample-batch, Janus and R1 wiring tests passed afterwards. No old
  helper references remain; touched-file Ruff/diff checks passed. Full review
  remains incomplete.

## Configuration helper ownership review: retained shared boundaries

- Inspected config/builders.py, data.py, algorithm.py, precision.py and the
  package facade against their production consumers. No code change is justified
  solely by these functions being module-level. This records reviewed boundaries
  rather than adding wrapper classes to produce a smaller function inventory.
- Keep build_precision_split_safety_configs: online TrainerConfig and the
  quantized rollout drift probe consume the same correction/guard pair; neither
  individual dataclass owns that cross-type policy. Keep build_configs as the
  validated training assembly entrypoint and its package-level lazy facade,
  which avoids loading trainer dependencies during config discovery.
- Keep manifest_sources: DataConfig validation, trainer prompt loading and data
  bootstrap consume the same accepted manifest spellings. Keep resolve_data_loader
  as the shared loader/format conflict boundary. No ConfigLoader owner is needed.
- Keep precision normalization helpers: schema field validators and runtime
  RolePrecision/PrecisionPolicy consume them. The quantization format table is
  isolated in the precision module; allowed device/dtype values derive from
  type/schema declarations. Do not distribute duplicate normalization among
  classes merely to remove free functions.
- Keep algorithm_config_class as lazy typed-config dispatch. This review does
  not prove exhaustive config validation or whole-repository completion. It
  confirms these ownership boundaries through source/call-site inspection; no
  executable behavior changed and no tests were rerun for this documentation.

## Metrics and training trace ownership review

- Confirmed MetricsCSV owns initialization, resume alignment and append IO;
  OnlineMetricsCSV owns the online schema and its two precision outputs. Keep
  _csv_field as a dataclass field-declaration adapter so formatting and phase
  mappings have one source. No additional wrapper class is justified.
- Confirmed TrainingRunTrace owns capture, loading, artifact sealing and
  verification. Its constants are persisted schema identities or explicit
  environment keys, and its local git function scopes subprocess mechanics to
  one snapshot. Retain those boundaries.
- The remaining continuous_lookahead_requested spelling is a deliberate
  persisted metrics contract, not the scheduling API. Owner code already uses
  prefetch_next_batch names and explicitly preserves the metric key. Add the
  same explanation beside the CSV declaration: renaming it without a migration
  would break MetricsCSV's resume header check. Preserve schema compatibility
  instead of performing an isolated cosmetic rename.
- Evidence: inspected the owner emission, CSV initialization/header comparison,
  continuous-field mapping test and schema-change resume rejection test. This
  comment/documentation change passed touched-file Ruff/diff checks; execution
  tests were not rerun. Full repository review remains incomplete.

## Reward service capacity fields are integer protocol values

- Reviewed client/wire/protocol ownership. Keep encode/decode functions shared
  by both HTTP endpoints, _parse_status as the client's protocol-error adapter,
  and WIRE_VERSION/_ARTIFACT_WIRE_FIELDS as real wire/schema boundaries.
- RewardServiceInfo only compared capacities numerically, accepting booleans,
  fractional counts and NaN. Six info_from_wire regression cases reproduced
  these invalid payloads passing validation. Reuse require_exact_int for both
  capacity fields before checking pending capacity against concurrency.
- Preserve JSON shape, error adaptation, client identity checks and valid service
  values. No new wrapper or validation function, and no request scheduling or
  cancellation change.
- Validation: all 61 service tests passed after the fix; six new wire cases
  failed before it. Touched-file Ruff/diff checks passed. Full review remains
  incomplete.

## Reward CLI scopes signal registrations to its lifetime

- _run_cli removed loop signal handlers only after shutdown_async returned; a
  shutdown exception skipped removal. Its signal.signal fallback never restored
  the previous process handler. Replace the installed list with ExitStack cleanup
  callbacks, registered after each successful installation. Loop removal precedes
  process-handler restoration; partial registration also unwinds earlier handlers.
- Keep _run_cli as the CLI/framework adapter rather than making RewardService own
  process signals. SIGINT/SIGTERM are actual process boundaries. Preserve service
  start/wait/shutdown order and propagated shutdown errors; no new owner class or
  cleanup helper. This does not claim preservation of preexisting asyncio loop
  callback registrations for arbitrary embedded use.
- Validation: 63 service tests passed. Two added cases simulate startup/shutdown
  failure and confirm original process handlers are restored for loop and fallback
  registration paths. Touched-file Ruff/diff checks passed. Full review remains
  incomplete.

## Artifact report construction owns its private helpers

- Move _artifact_values and _assert_readable into ArtifactManifestReport; only
  its from_examples constructor consumes them. Inline the single-use episode
  accumulation there and read PromptExample.metadata directly, whose declared
  field has a dict default. Remove the defensive copy/fallback for absent metadata.
- Preserve artifact-field conversion, file/Pillow checks, error messages, sorted
  unique source episodes and train/eval overlap reporting. No new wrapper class
  or compatibility helper.
- Keep shared prompt artifact/reference resolution APIs and SFT shard load/save
  functions as path and persistence boundaries. Keep SFT_LATENTS_SCHEMA_VERSION
  as the file-format version. Those functions need no synthetic loader object.
- Validation: all 89 data tests passed, covering manifest validation, provenance
  and dataset behavior. Touched-file Ruff/diff checks passed; removed free-helper
  callers are gone. Full repository review remains incomplete.

## Artifact path fields reject coercion from unrelated values

- _artifact_values treated any Iterable as paths, including mapping keys and
  byte integers, and converted numeric/None list items to file names. Four
  regressions reproduced misleading missing-file errors instead of type errors.
- Accept strings and list/tuple containers of strings explicitly, preserving
  empty-entry filtering. Keep the implementation inside ArtifactManifestReport;
  no new parser/helper or schema constants. Generator/dict/bytes path containers
  are no longer accepted implicitly. Shared path resolution remains unchanged.
- Validation: 93 data tests passed; after adding positive list/tuple coverage,
  all 15 focused manifest tests passed. Touched-file Ruff/diff checks passed.
  Full repository review remains incomplete.

## Image-caption manifest errors identify their source

- ImageCaptionPromptDataset decoded each line without adding manifest context.
  JSONDecodeError therefore reported a location inside the isolated line rather
  than the actual file/row. Add the manifest path and existing zero-based physical
  row index while retaining the decoder exception as cause.
- Keep the short Dataset __len__/__getitem__ methods as framework adapters;
  creating another base class merely to share those lines adds no useful owner.
  Keep load_prompt_image_manifest as the shared list-returning entry for config
  loading and provenance. PreferenceBatch.collate already owns construction.
- No accepted-row, blank-line, metadata or task-type behavior changes. No new
  helper/class/schema constants.
- Validation: 132 prompt/data tests passed, including a malformed third physical
  line after a blank line and preserved JSONDecodeError cause. Touched-file
  Ruff/diff checks passed. Full repository review remains incomplete.

## Prompt sampler validates dimensions at construction

- PromptBatchSampler only range-checked num_examples, prompts_per_rank,
  num_replicas and rank. Fractional/bool dimensions could survive construction
  and fail later in randperm/range/slicing, obscuring the bad field. Reuse
  require_exact_int before existing range checks, keeping their diagnostics.
- Keep _sample_with as the shared sample/preview implementation: cloning the
  generator for preview must use the same draw and rank slicing as sampling.
  Preserve RNG consumption, strategy semantics and torch-free config discovery.
  The local field tuple is the sampler schema, not a new workflow taxonomy;
  no extra validation function or class.
- Validation: 151 sampler and config-schema tests passed, including eight
  constructor rejection cases for bool/fractional dimensions. Touched-file
  Ruff/diff checks passed. Full repository review remains incomplete.

## Reference selection stays beside trainer construction

- Inline the single-use _default_reference_model into run_online_recipe's
  reference-selection block. The denoise/evaluator gate, optional kl_coef lookup,
  and LoRA condition are now visible together before trainer construction.
  Remove the outdated algorithm-inheritance explanation; capability/field
  presence determines this behavior, not a manually enumerated family list.
- Preserve evaluation order and the same selected bundle.model/None values.
  Keep SFT shard loading as the provenance-checked persistence boundary and
  host-memory checking as the streaming update diagnostic. No new owner class,
  helper, or constants; keep other CLI/framework adapters unchanged.
- Validation: 103 reward-update-flow, online precision-bridge and SFT-loading
  tests passed; touched-file Ruff/diff checks passed and no removed helper
  callers remain. This refactor does not claim real-GPU KL validation. Full
  repository review remains incomplete.

## Joint regression after generation, data and service cleanup

- Reviewed online topology preflight helpers against their callers. Retain
  _require_supported_distributed_rollout_topology (training context plus resource
  ownership) and _validate_reward_placement (reservation plus execution ordinal).
  Neither has one natural object owner. Retain the production Kling preflight
  as a conditional import/error-context boundary. No wrapper classes or new
  configuration vocabulary are warranted by this inspection.
- Ran the combined generation, rollouts, trainers/data, config and rewards/service
  suites on the current implementation after the preceding cross-layer changes:
  1677 passed, two skipped, four dependency warnings in 57.30 seconds. The process
  exited successfully. This checks interactions beyond the individual change
  slices; it does not certify a real-GPU training run or review uninspected files.
- No implementation changed in this slice. The audit remains in progress; joint
  green tests are not proof that whole-repository clarity work is complete.

## Reward owner repeated cancellation retains execution ownership

- Reviewed `RewardScorerOwner.score_batch` alongside the service cancellation,
  admission and shutdown callers. Keep the dedicated loop, pre-start cancellation
  flag and execution-completion acknowledgement: these are thread/lifecycle
  boundaries, not redundant free helpers. No additional wrapper class is needed.
- Found that shielding the completion waiter once does not prevent a second
  caller cancellation from escaping the wait. Repeated DELETE requests or a
  concurrent shutdown can consequently release service admission while the
  synchronous runtime call is still executing.
- Retain one completion task and shield it through repeated cancellations before
  propagating cancellation. Runtime cancellation is still requested once; this
  does not preempt synchronous GPU work or change the shutdown timeout policy.
- Added a regression that blocks a fake synchronous scorer, cancels its caller
  three times and requires the caller to remain pending until execution exits.
  It failed on the previous implementation. All 64 reward service tests now pass;
  touched-file Ruff lint and formatting checks pass. No real GPU execution is
  claimed. The repository-wide clarity audit remains in progress.

## Shared denoise metric naming and pipeline helper review

- Rename `rel_l1` to `relative_l1_change` in the TeaCache implementation, its
  export, drift probe and tests. The name now describes the measured quantity.
  Keep it as a shared free function: runtime skip decisions and offline drift
  analysis must use the same fp32 reduction. No compatibility alias or new class
  is introduced; external imports of the old name must update. Historical sprint
  notes remain historical rather than being rewritten as current implementation.
- Retain `_move_tree_to_cpu_async` in generation execution: it owns pinned-host
  allocation, stream-scoped copies and source `record_stream` lifetime tracking.
  The pipeline calls it for both intermediate and final batches. Folding that
  logic into both branches would duplicate a CUDA synchronization boundary.
- No cache thresholds, skip decisions or pipeline synchronization changed.
  All 57 denoise tests passed; touched-file Ruff checks passed after formatting
  the updated probe import. No real-GPU benchmark was run. This slice does not
  complete the repository-wide audit.

## Continuous owner command wait belongs to the thread facade

- Move the private free `_await_owner_future` into `ContinuousRolloutOwner` as
  `_await_command`. All three production callers belong to this facade. Retain
  a named method because it expresses the cross-thread cancellation boundary:
  trainer waiter cancellation must not cancel an already submitted owner command.
- Update the test-only owner snapshot adapter to use the same owner method.
  The first regression run exposed its old private import; after migrating the
  adapter, all 209 continuous orchestration tests passed. Touched-file Ruff lint
  and formatting checks pass. No compatibility forwarding helper remains.
- Preserve shutdown, command ordering and cancellation behavior. Do not unify
  this with reward-request cancellation: the two owners deliberately differ in
  whether waiter cancellation should stop the submitted work. Retain the named
  owner lifecycle timeout values and byte conversion constant; these describe
  waiting limits and units, not an algorithm vocabulary table.
- This ownership cleanup does not complete the repository-wide audit.

## Local OOM retry uses an explicit deque

- Replace list-front removal/insertion with deque popleft/appendleft in
  `run_sample_batches_with_oom_retry`. Pending batches form a queue; recursive
  splits are prepended right then left to preserve left-first sample order.
  This avoids shifting every remaining batch on each operation.
- Keep the shared execution function and replay/coverage helpers: they enforce
  cross-family consistency at the batch boundary. No wrapper class, retry policy,
  exception handling or allocator cleanup changes are introduced. This is not
  a claim of measurable end-to-end training acceleration.
- Existing tests cover recursive split order, failed-forward reference release
  before cache clearing, and preservation of terminal failure tracebacks. All
  121 sample-batch and full-sequence layout tests passed, as did touched-file
  Ruff lint and formatting checks. The broader audit remains incomplete.

## Weight sync carries the public buffer option name through assembly

- Rename the `RayGenerationWeightSync` constructor parameter and stored member
  from `bucket_bytes` to `update_weight_buffer_size`, matching the public worker
  config. Update launcher, delivery probe and direct test construction. External
  Python callers using the old keyword must update; no alias is retained.
- Keep the low-level chunk/bucket iterator parameter `bucket_bytes`: it describes
  the actual transport byte ceiling. Keep the probe's persisted `bucket_bytes`
  report key to preserve its existing schema. Sender manifest/chunk construction
  and receiver staging remain distinct protocol responsibilities; no wrapper
  class or transfer behavior changes are introduced.
- Weight-sync, transfer and delivery-probe tests: 57 passed with one Ray future
  warning. Launcher tests: eight passed with the same dependency warning. The
  first attempted test command referenced a nonexistent launcher test filename;
  the corrected file was run successfully. Touched-file Ruff checks pass.
  These tests do not establish real-GPU throughput. The overall audit continues.

## MiniMax-H3 owns replay batch expansion

- Move the module-level `_expand_batch` into `MiniMaxH3Model` as
  `_expand_replay_batch`, next to replay export. Both production calls belong to
  that exporter (prompt embeddings and per-step audio rows). The method name
  now states its scope; no new owner class or forwarding alias is introduced.
- Keep one method for the common rule: preserve an already matching batch,
  broadcast a singleton, reject every other batch-size mismatch. Replay tensor
  values, contiguous layout and error behavior are unchanged.
- Keep family loader adapters and model architecture constants outside this
  change. Cross-family shape consistency and framework boundaries matter more
  than minimizing function counts.
- All 18 MiniMax-H3 tests passed, including replay parity and model loading;
  dependency/test warnings were emitted. Touched-file Ruff checks pass after
  formatting. This does not claim a full pretrained GPU training validation or
  completion of the repository-wide audit.

## Echo owns timestep-to-sigma conversion

- Move `_sigma_from_timestep` onto `EchoModel`, adjacent to its sole production
  caller `forward_step`. The replay subclass inherits that forward path, so both
  rollout and replay retain the same normalization and positive sigma floor.
  Update the existing focused tests to address the owning class.
- Correct the docstring: the implementation normalizes in fp32 and clamps the
  minimum; it does not enforce an upper bound of one. No numerical change or
  additional validation is introduced. Keep the method as the named conversion
  boundary and retain unrelated family loading adapters and architecture values.
- All 21 Echo tests passed, including velocity/replay checks. Touched-file Ruff
  checks pass. This is a local ownership/documentation cleanup, not a full model
  training validation or completion of the repository-wide audit.

## GLM-Image owns its decode token upsampling

- Move `_upsample_token_ids` onto `GlmImageModel`, immediately before
  `decode_image_tokens`, its sole production caller. Preserve the reference
  nearest-neighbor conversion and dependency boundary; no new utility class or
  forwarding alias is introduced.
- Retain grid/token/position functions as the explicit shared layout rules and
  retain pretrained/decode loading functions as lazy framework boundaries.
  This change does not generalize model-specific raster rules across families.
- All 22 GLM-Image tests passed, with two dependency warnings. Touched-file Ruff
  checks pass. No full pretrained image-generation benchmark is claimed, and
  the repository-wide clarity audit remains incomplete.

## Cosmos3 owns packed input assembly and replay row extraction

- Move `_single_sample_input_ids` and `_assemble_packed_static` onto
  `Cosmos3Model`. Their six production calls all belong to sampling preparation
  or replay restoration on that model; the replay subclass inherits the same
  behavior. Remove the module-level definitions without compatibility wrappers.
- Retain separate named methods for the batch-one replay boundary and static
  text/vision assembly. No generic utility class, layout schema change, token
  coercion change or default-value change is included in this ownership slice.
- Five Cosmos3 family tests and 87 shared gatherer/family-MRO tests passed, with
  dependency warnings. Touched-file Ruff checks pass. These checks do not prove
  full pretrained GPU training parity. The repository-wide audit continues.

## Cosmos3 replay token IDs are validated rather than coerced

- Replace `int(token_id)` in the model-owned replay row reader with the existing
  `require_exact_int` boundary, minimum zero and a field/index error path.
  Fractional values, numeric strings, booleans and negative IDs no longer pass
  through as silently converted model inputs. Valid Python integer rows and
  packed input layout are unchanged; no new validator helper is introduced.
- Add eight public restore-entry regressions covering both conditional and
  unconditional rows. Previously they proceeded beyond the ID boundary and
  failed later on deliberately minimal latent geometry rather than identifying
  the invalid token. They now report the offending field/index before segment
  building. The tests do not claim an upper vocabulary-bound check.
- All 58 Cosmos3 and shared gatherer tests passed, with dependency warnings;
  touched-file Ruff checks pass. The repository-wide clarity audit continues.

## MAGI subprocess model owns mode selection and unsupported-training diagnostics

- Move `_magi_mode` onto `Magi1SubprocessModel` as `_generation_mode`, beside
  its sole generation caller. Keep the task-to-CLI mapping as a model-specific
  adapter; accepted task types and conditioning validation are unchanged.
- Move `_generation_only_error` onto the same model beside replay entry points.
  Keep a single diagnostic shared by replay and nonempty trainable-state loading,
  without introducing a module-level business-message constant or wrapper class.
- Preserve source/path resolution, environment construction and installation
  checks in this slice. They belong to the loading workflow and require their
  own caller review, not mechanical relocation based on function counts.
- All 37 MAGI subprocess runtime tests passed; touched-file Ruff checks pass.
  No official checkpoint inference was run. The full repository audit continues.

## MAGI optional path initialization stays in its config

- Replace three repeated frozen-field assignments in `Magi1SubprocessConfig`
  with one loop over its checkpoint, T5 and VAE path schema fields. Handle None
  directly there and delete the one-line `_optional_source_path` forwarding
  function. Normalization order and relative-path semantics remain unchanged.
- Retain `_resolve_from_source`: configuration initialization and runtime path
  validation share the same source-root resolution rule. Retain `_optional_path`
  in weight resolution for now; it normalizes absent/string source values at a
  different input boundary. No path policy or constructor behavior change.
- All 37 MAGI tests passed and touched-file Ruff checks pass. The repository-wide
  audit remains in progress.

## MAGI config normalization and preflight boundary review

- Remove the redundant `float()` conversion from `Magi1SubprocessConfig.from_build`.
  Its constructor already delegates conversion and finite/positive validation to
  `require_timeout`. Both direct and build-based construction now pass through
  that one normalization boundary; accepted values and timeout policy are unchanged.
- Retain `_validate_single_process_config`: both original JSON preflight and
  sample-specific runtime configuration call it. Retain environment probing as
  the dedicated-interpreter import boundary before weight downloads. Keep runtime
  path validation distinct from source resolution, since existence checks and
  relative-path interpretation serve different stages.
- All 37 MAGI tests passed; touched-file Ruff lint/format checks pass. No new tests
  were added for this redundant conversion removal. Remaining loading helpers
  still require individual ownership review; the repository audit is incomplete.

## Joint model regression and semantic dtype assertions

- Ran all model tests after the preceding family ownership changes. The first
  run reported 868 passed, one skipped and one SD3 loading assertion failure;
  isolated execution reproduced it. The test expected an older spelling of the
  component dtype mapping, while the loader explicitly names all three frozen
  encoders and uses the model dtype as its default.
- Update that test to assert the effective dtype for transformer, VAE and each
  encoder, retaining freeze/device checks and the single-load/source assertion.
  Do not change production loading to match an obsolete dictionary shape. This
  fake-loader test verifies requested policy, not an actual GPU memory peak.
- The full model rerun passed: 869 passed, one skipped, 61 warnings in 10.01s,
  process exit zero. Touched-test Ruff checks passed.
- Retain shared diffusers loading functions as lazy cross-family framework
  adapters. Retain Anima rotary helpers as recognizable model math beside the
  adapter architecture; no new wrapper is justified solely by function counts.
  Full-repository clarity completion remains unproven.

## Distributed context owns launcher environment parsing

- Move `_require_env_int` onto `DistributedTrainingContext`, its sole production
  owner. Keep this method as the environment string-to-integer boundary used for
  rank, local rank and world size. No new context or forwarding alias is added.
- Replace the hardcoded FSDP-only missing-environment diagnostic with a distributed
  training message: both FSDP and DDP use this path. Extend the existing missing-env
  test across both strategies. Parsing and device-selection behavior are unchanged.
- Retain `_TORCHRUN_ENV_KEYS` as launcher protocol keys and retain shared process
  group lifecycle functions outside the identity class. Group creation is not
  part of constructing a process identity.
- All eight distributed-context tests passed; touched-file Ruff checks pass.
  This does not exercise a multi-GPU process group. The repository audit continues.

## Distributed identity validates the global rank range

- `DistributedTrainingContext.from_root` already checked world size against
  topology and local rank against GPUs per node, but accepted global ranks
  outside `[0, WORLD_SIZE)`. Those identities reached process-group initialization.
- Add the missing global-rank check beside the existing topology checks, before
  device discovery. No new helper, class or environment vocabulary is introduced;
  valid device mapping and process-group lifecycle are unchanged.
- Four regressions (negative and upper-bound rank for DDP/FSDP) failed on the old
  implementation because it did not raise. All 12 distributed-context tests now
  pass; touched-file Ruff checks pass. No multi-GPU execution is claimed. The
  full repository clarity audit remains in progress.

## Trainable state selection contains its parameter-name check

- Inline the sole-use `_trainable_parameter_names` into `select_trainable_state`.
  The shared operation now reads in one place: identify trainable names, reject
  empty/missing entries, and return the prefixed policy payload. Name the local
  comprehension variable `parameter_name` to distinguish it from the module name.
- Retain `select_trainable_state` as the shared single-process/FSDP policy-facing
  selection boundary, and retain `to_cpu_snapshot` for independent storage during
  asynchronous synchronization. No new utility class or filtering changes.
- Weight sync, strategy and FSDP tests: 83 passed, two skipped, dependency/profiler
  warnings. Touched-file Ruff checks pass. No multi-GPU training claim is made;
  the repository-wide clarity audit remains in progress.

## Precision guard validates replay indices instead of coercing them

- Replace `int(t)` in timestep selection with the existing exact-integer check,
  minimum zero and an indexed error path. Fractional/string/bool indices no longer
  become a different step, and negative indices cannot select from the end of
  a trajectory. Preserve deduplication order and evenly spaced selection.
- Retain the shared drift-record key: local timestep selection and distributed
  rank selection both compare whole records, preserving provenance. No guard
  class, threshold policy or mode-resolution changes are introduced.
- Four invalid-index regressions failed before the fix. All 22 precision guard
  tests now pass; touched-file Ruff checks pass. This does not establish real-GPU
  drift behavior or complete the repository-wide audit.

## Strategy assembly states its FSDP compile restriction directly

- Inline `_assert_fsdp_config_supported` into the FSDP branch of `build_strategy`.
  Its sole check concerns replay compilation; the generic helper name and old
  sprint-gate history obscured that narrow responsibility. Preserve the lazy
  role-compile import, check order, exception type and diagnostic text.
- Retain CPU coordination barrier and all-rank success helpers as shared
  communication boundaries. Do not introduce a strategy validator class or move
  process-group behavior into configuration objects.
- Strategy and FSDP suites passed: 72 passed, two skipped, dependency warnings.
  Touched-file Ruff checks passed. No new compile support or multi-GPU validation
  is claimed; the overall repository audit remains incomplete.

## Parking helper review and capability-based wrapping diagnostic

- Retain the strategy tensor/module migration helpers after reviewing callers:
  module enumeration includes gradients; in-place tree moves deduplicate tensor
  identity and record original devices; DTensor moves preserve the wrapper while
  relocating only local shards. These functions remove real shared complexity.
- Remove family names and the old sprint reference from the missing-trainable-root
  diagnostic. State the actual required interfaces (`trainable_modules` and
  `set_module_root`) instead. Validation remains capability-based and its exception
  type is unchanged; no family support or device movement behavior is modified.
- Strategy/FSDP tests: 72 passed, two skipped, dependency warnings. Touched-file
  Ruff checks pass. The repository-wide audit remains incomplete.

## Adapter export overlap condition stays with its error

- Inline the sole-use `_output_paths_overlap` into adapter export resolution.
  The same-path and ancestor/descendant checks now sit directly beside the
  conflicting-export diagnostic; no algorithm or publication-order change.
- Retain `_safe_relative_output_path` as the shared boundary for adapter names
  and export directories. Its absolute/drive/traversal/separator checks protect
  two distinct entry points. Checkpoint/file-name constants remain real schema
  boundaries, and no path-policy class is introduced for this cleanup.
- All 110 checkpointing tests passed, including overlapping output paths failing
  before IO and identical effective PEFT output paths. Touched-file Ruff checks
  pass. This does not complete the repository-wide audit.

## Resolved adapter export owns namespace projection

- Move `_adapter_relative_state` onto the existing `_ResolvedAdapterExport` as
  `select_state`. The method reads its own state prefix, so publication no longer
  extracts that field merely to pass it into a separate free function. Keep the
  artifact name argument for contextual errors; no new carrier is introduced.
- Preserve empty-state rejection, prefix removal and checkpoint publication
  ordering. Retain rank agreement and EMA parameter-order helpers as distinct
  cross-rank/model-boundary checks, outside any individual adapter export.
- All 110 checkpointing tests passed with dependency warnings; touched-file Ruff
  checks passed. This slice does not complete the repository-wide audit.

## Checkpoint family validation states its actual contract

- Rename `_require_checkpoint_family` to `_validate_checkpoint_family` and remove
  its unused return value. All six callers use it only as a guard. Correct the
  docstring: it validates a non-empty trimmed identifier, not canonical alias
  resolution. No family vocabulary or compatibility rule changes are introduced.
- Retain the shared identity/schema validators. Sidecar preflight, authoritative
  payload validation and sidecar/payload consistency checks serve distinct stages;
  merging them into a single unchecked loading path would lose those boundaries.
- All 110 checkpointing tests passed with dependency warnings; touched-file Ruff
  checks pass. The repository-wide clarity audit remains incomplete.

## Checkpoint completeness validates recorded byte counts

- Remove `int()` coercion from the optional checkpoint byte-count check. A
  recorded size must be a non-negative integer equal to the actual file size;
  malformed types return false instead of silently matching or raising conversion
  exceptions during discovery. Missing/None size retains existing compatibility.
- Keep explicit metadata global_step ordering, with no directory-suffix progress
  inference. Keep discovery/completeness functions and file-name constants as
  shared persistence boundaries; no new wrapper or metadata vocabulary is added.
- Six size cases cover fractional, boolean, string, malformed string/list and
  negative values. Five failed before the fix (the negative case already passed).
  All 116 checkpointing tests now pass; touched-file Ruff checks pass. The overall
  repository clarity audit remains incomplete.

## Checkpoint consumer regression after completeness tightening

- Verified supervisor recovery delegates to `find_latest_complete_checkpoint`,
  while evaluation uses the shared completeness predicate and `CheckpointTarget`.
  Retain these public persistence boundaries and checkpoint/file-name constants;
  no duplicate progress inference or new wrapper is needed at these call sites.
- Retain supervisor `_bounded_number` as an argparse type adapter and its
  single-owner environment guard as a process-launch boundary. Their small size
  alone does not justify relocation into unrelated model/checkpoint classes.
- Ran supervisor, image checkpoint evaluation and SANA checkpoint evaluation
  suites on the current implementation: 116 passed in 7.30s, exit zero. No
  implementation changed in this slice. This validates consumer integration
  beyond the checkpoint unit suite; full repository clarity remains unproven.

## Diagnostic record summarization is local to append

- Move the sole-use recursive `_json_safe` into `append_jsonl_record` as
  `summarize`. Its responsibility is nested tensor summarization and fallback
  string formatting, not generic JSON validation. The lazy torch import now
  belongs to one append invocation rather than every recursive visit.
- Retain the public append API and shared `tensor_stats` function. No writer
  class, record schema, nonfinite-value policy or append semantics change is
  introduced; the docstring now describes the actual operation directly.
- All 20 diagnostic utility/online tests passed; touched-file Ruff checks pass.
  The repository-wide audit remains incomplete.

## Joint trainer regression after ownership and validation changes

- Ran the entire trainer test tree on the current implementation: 580 passed,
  seven skipped, 15 warnings in 42.53s, process exit zero. This extends evidence
  beyond individual checkpoint, context, weight-sync and diagnostic suites.
- Retain scalar distributed wrappers: each fixes a meaningful dtype/reduction
  combination (integer max, float max, boolean all-true), while their shared
  scalar collective handles backend/device placement. Uniform names make the
  calling control flow clearer than repeated op/dtype arguments.
- Reviewed online config construction and identified two sole-use helpers,
  `_null_key` and `_required_field_paths`, as the next ownership candidates.
  No production edits were made while this regression was running. Green trainer
  tests do not establish full-repository clarity or real multi-GPU training parity.

## Online config owns null and required-field handling

- Inline `_null_key` and `_required_field_paths` into their sole caller,
  `TrainerConfig.from_root`. Explicit-null rejection and nested required-field
  enumeration now sit inside the missing-value branch; remove both module-level
  helpers without adding methods or a wrapper class.
- Preserve dataclass-derived required paths, public-section ownership checks,
  defaults and error text. Keep `OnlineBatchPlan` and precision construction as
  separate meaningful owners; no duplicate field vocabulary is introduced.
- Online config, online run config and config suites passed: 361 tests in 11.59s.
  Touched-file Ruff checks pass. The repository-wide audit is still incomplete.

## Online trainer owns EMA restore shape validation

- Move `_validate_ema_state_shapes` onto `OnlineTrainer`, its sole production
  owner. Read `self.model` directly instead of passing the same owner's model
  through a free function. No new state object or compatibility wrapper.
- Retain the named validation step for EMA parameter count, tensor type and
  shape checks. Preserve strict/non-strict behavior and the existing EMA loader;
  this ownership change does not alter checkpoint compatibility policy.
- State-restore and checkpointing suites: 131 passed, dependency warnings.
  Touched-file Ruff checks pass. The repository-wide audit continues.

## Replay sample batch construction belongs to its carrier

- Move `_training_sample_batches` onto `_TrainingGenerationSampleBatch` as
  `from_prompt_group`, constructing through cls. Update balanced planning, replay
  debug paths and the direct loss-weight test. No new class or alias is added.
- Preserve batch slicing, advantage alignment, full-group loss weights and empty
  handling. Keep cross-rank slot balancing outside the carrier because it owns
  collective coordination and dummy-slot planning across multiple prompt groups.
- Advantage/metrics, distributed skip-backward agreement and diagnostics tests:
  30 passed. Touched-file Ruff checks pass. The repository-wide audit continues.

## Replay slice naming distinguishes it from generation work

- Rename the private `_TrainingGenerationSampleBatch` to `_ReplaySampleBatch`
  throughout trainer construction, balanced planning, debug paths and tests.
  Document its actual payload: replay slice, advantages and full-group loss
  contribution. It does not describe a generation dispatch task.
- Preserve the real `GenerationSampleBatch` identity and all slicing, loss and
  dummy-slot behavior. No alias, new type hierarchy or scheduling change.
- The 30 advantage/metrics, distributed skip-backward and diagnostic tests passed;
  touched-file Ruff checks pass. Repository-wide completion remains unproven.

## Replay slicing consumes the validated batch-plan value

- Remove the redundant int conversion of samples_per_replay_batch in the private
  replay-slice constructor. Every production caller passes the OnlineBatchPlan
  field, whose constructor already enforces a non-negative integer. Document
  that zero preserves the whole group; no duplicate validator is introduced.
- Preserve selection and loss-weight calculations for valid plans. Direct private
  callers must honor the annotated integer contract rather than rely on coercion.
  Keep OnlineBatchPlan as the validation owner and distributed balancing separate.
- Advantage/metrics, reward update flow and trajectory granularity tests passed:
  27 tests, including existing zero-width whole-group paths. Touched-file Ruff
  checks pass. The repository-wide audit remains incomplete.

## Online collection owns reward component aggregation

- Move `_rollout_reward_components` onto `OnlineTrainer` as
  `_collect_reward_components`, adjacent to collection. Its only caller needs
  these values for advantage calculation before constructing TrainingBatch, so
  the trainer collection flow is the owner rather than the passive batch carrier.
- Preserve per-batch score-count validation, tensor conversion and aggregation.
  No utility class, reward schema or advantage behavior changes are introduced.
  Update the adjacent next_prompts docstring to use next-batch prefetch wording.
- Advantage/metrics and reward update flow tests passed: 18 tests. Touched-file
  Ruff checks pass. The repository-wide clarity audit remains incomplete.

## Online trainer owns GradScaler construction

- Move `_create_grad_scaler` onto `OnlineTrainer` as a static factory, next to
  initialization, and update the precision-matrix test to call it through the
  existing public trainer import. Keep explicit device/model inputs for testing.
- Preserve CUDA-only creation and the FP16 autocast/native-gradient criteria.
  Keep master-weight detection shared between optimizer creation and restore;
  no precision policy or new wrapper class is introduced.
- GradScaler and state-restore suites passed: 32 tests. Touched-file Ruff checks
  pass after removing a duplicate trainer import introduced during the move.
  The repository-wide clarity audit remains incomplete.

## Online trainer owns master-weight detection

- Move `_requires_fp32_master_weights` onto OnlineTrainer as an instance method
  beside optimizer initialization. Both callers belong to the trainer and always
  inspect self.model, so remove the redundant model argument and module helper.
- Keep one dynamic query shared by optimizer creation and state restoration.
  Do not cache the answer or change FP16/BF16 criteria, optimizer construction,
  checkpoint requirements or the master-weight implementation.
- GradScaler and restore suites: 32 passed; touched-file Ruff checks pass.
  The overall repository clarity audit remains in progress.

## Trainer restore validates both progress fields before assigning either

- Replace coercive int conversion of step/global_step with the existing exact
  non-negative integer validator. Validate both locals before assigning trainer
  progress, so a malformed second field cannot partially overwrite the first.
- Retain missing-field zero defaults and strict/non-strict optimizer/EMA policy.
  Invalid progress is rejected in either mode; this does not make the entire
  optimizer/EMA restoration transaction atomic.
- Sixteen malformed-progress cases failed before the fix. State restore plus
  checkpoint suites now pass: 147 tests, dependency warnings. Touched-file Ruff
  checks pass. The full repository clarity audit remains incomplete.

## Offline DPO restore shares the exact progress boundary

- Replace global_step int coercion with require_exact_int, minimum zero, before
  assigning progress or resetting the accumulation window. This matches the
  online restore boundary without introducing a common restore base class.
- Preserve missing-field zero defaults, optimizer restore policy and valid
  checkpoint accumulation reset behavior. Invalid values fail in either strict
  mode without changing progress or the micro-step counter.
- Eight regressions failed before the fix. Offline timestep/restore and builder
  suites passed: 33 tests. Touched-file Ruff checks pass. Repository-wide clarity
  completion remains unproven.

## Storage policy checks its no-op fields without constructing another policy

- Replace comparisons against a fresh TrajectoryStoragePolicy() with direct
  device/dtype preserve checks in both application methods. The common no-op
  path no longer creates and validates a temporary default policy per call.
- Keep the two explicit conditions instead of adding a thin property solely for
  this predicate. Keep trajectory_tensor_bytes as a shared payload estimate and
  derived allowed-value constants as configuration schema boundaries.
- Storage policy suite: 4 passed, including object/tensor identity and placement
  and dtype behavior. Touched-file Ruff checks pass. No storage conversion rule
  or byte accounting is changed; repository-wide completion remains unproven.

## Joint math and NN regression after input-boundary cleanup

- Ran tests/math, tests/nn, NextStep family and CausVid family suites together
  after the recent scale, step-count, shape and ownership changes: 300 passed,
  16 dependency warnings in 6.81 seconds. Log:
  /tmp/vrl-math-nn-clarity-regression.log.
- Inspected coverage for unchanged-policy sample/replay parity, fp64 autograd
  gradcheck and empty-batch gradients; the combined run also includes the real
  CUDA paged KV-cache write test now enabled by the restored uvloop dependency.
- Keep diffusion_pretraining_pair as a scheduler-contract adapter shared with
  training, and keep categorical-temperature validation as a common policy
  boundary. No additional classes or function moves are justified in this slice.
- This verifies these combined tested paths, not all training workloads or
  repository-wide clarity completion. The full audit remains in progress.

## Token Gaussian noise scale rejects invalid densities explicitly

- Require finite positive noise_level in the existing shared _flow_noise_std.
  Sampling and replay previously accepted zero, negative, NaN and infinite scales
  despite requiring a nondegenerate Gaussian density. Eight regressions reproduced
  the missing field error before the fix.
- Preserve the scale formula and existing small-log-scale floor for valid inputs.
  No new helper, policy class or deterministic scoring convention is introduced.
  Validation happens at scale construction, after the terminal mean calculation.
- Token math and NextStep suites: 70 passed, two dependency warnings. Touched-file
  Ruff checks pass. Full repository clarity completion remains unproven.

## Token-flow Euler step count has one exact integer boundary

- Validate num_steps with require_exact_int, minimum one, in the shared terminal
  mean calculation before creating the time grid or calling the velocity head.
  Both direct sampling and replay now reject malformed step counts consistently.
- Keep shared mathematical helpers and valid Euler/noise calculations unchanged.
  No new wrapper is added. This check precedes velocity execution but does not
  promise to precede sampling's initial-prior allocation when none was supplied.
- Ten diagnostic regressions failed before the fix and also assert no velocity
  call on rejection. Token math and NextStep family suites: 62 passed, two
  dependency warnings. Touched-file Ruff checks pass. The full audit continues.

## Token-flow documentation states the actual conditional density

- Correct the sampling formula to use delta=token-mean when dividing by std²,
  rather than dividing standard-normal eps by std² again. State that noise_level=1
  gives variance 1/num_steps and remove the false zero-log-prob-mass claim for a
  deterministic action. The Gaussian scorer requires positive scale.
- Clarify that replay preserves the initial prior, not the trajectory after a
  policy update. The current velocity field recomputes the trajectory and mean.
- Keep the shared math helpers, APIs and computation unchanged. This change is
  docstrings only; no tests were added or rerun. Touched-file Ruff and diff checks
  pass. Runtime scale validation is not changed here; the full audit continues.

## Token-flow CFG calls the velocity network without a forwarding wrapper

- Remove the local _velocity function that only forwarded to image_head.net.
  The guided velocity calculation now calls the declared velocity-network
  interface directly for conditional and unconditional inputs.
- Keep shared terminal-mean, noise-scale and Gaussian-density functions: sampling
  and replay both consume them and must use identical formulas. Keep the local
  guided calculation for the repeated Euler steps; no new class is introduced.
- Token flow-matching tests: 4 passed. Touched-file Ruff checks pass. Network call
  order, guidance, Euler updates and densities are unchanged. The full repository
  clarity audit remains incomplete.

## Re-noise transition rejects nonfinite sigma before density computation

- Combine finite and positive sigma validation into the existing tensor predicate
  and host check. NaN and positive infinity previously escaped the positivity
  check; negative infinity was already rejected, now with the complete condition.
- Keep validation in the shared rollout/replay math function, without a helper,
  new class or upper-bound restriction. Preserve sampling/scoring equations and
  serialized-action dtype behavior.
- Six sampling/scoring diagnostic cases failed before the change. Re-noise and
  CausVid suites: 34 passed. Touched-file Ruff checks pass. The full repository
  clarity audit remains incomplete.

## Denoise math names computation precision and distinguishes sigma quantities

- Rename local md to computation_dtype in flow-matching and DDIM math. Clarify
  SDEStepResult.sigma's comment: sigma's [0, 1] range does not imply the same
  range for standard deviation or sqrt(-dt), although all use flow-domain math.
- Keep standalone step functions as the shared generation/replay mathematical
  boundary. No scheduler owner or parameter wrapper is introduced, and equations,
  public arguments, dtype choices and output fields remain unchanged.
- Flow-matching and DDIM suites: 17 passed. Touched-file Ruff checks pass. The
  repository-wide clarity audit remains incomplete.

## Quantization defaults and explicit targets share one profile conversion

- Convert the selected default-or-explicit target through LinearTargetProfile
  before traversal. An invalid class declaration previously bypassed conversion
  and failed later with a missing matches attribute instead of the profile error.
- Extend the existing no-mutation regression across both selection paths and
  both quantization schemes. The two default cases failed before this change.
  Keep the enum, scheme registry and common traversal; no new validator or
  fallback value is introduced, and valid target scopes remain unchanged.
- Quantization and tiny-model targeting suites: 65 passed, 16 dependency warnings.
  Touched-file Ruff checks pass and formatting is applied. The full audit continues.

## Linear target profiles own their path predicate

- Replace matches_linear_target(path, profile) with LinearTargetProfile.matches
  and update quantization traversal and targeting tests. The existing enum owns
  both the selected policy and its matching behavior; no new policy class is
  needed. Remove the old exported helper without a compatibility alias.
- Preserve matching rules and target scope. Keep exclusion/MLP constants in the
  isolated model-path taxonomy and drop_quantized_masters as a scheme-independent
  module-tree operation; neither belongs to an individual quantized scheme.
- Real tiny-model targeting and quantization suites: 63 passed, 16 dependency
  warnings. Touched-file Ruff checks pass and formatting is applied. External
  imports of the retired helper must migrate; the broader audit continues.

## Fused log-prob reports missing hidden feature dimensions

- Reject scalar hidden tensors before indexing their final dimension. The
  previous path raised an incidental tuple IndexError when token_ids was also
  scalar; the boundary now reports the expected [..., D] feature dimension.
- Keep the check in the existing public function, without a new validator or
  kernel change. A single-token hidden vector with scalar output remains valid
  and now has an eager-equivalence regression alongside existing empty batches.
- The malformed-scalar regression failed before the fix. Fused log-prob suite:
  26 passed. Touched-file Ruff checks pass. The wider clarity audit continues.

## Fused log-prob precision helper names its shared accumulation role

- Rename _norm_dtype to _accumulation_dtype: both normalization and weight-gradient
  accumulation use it. Correct the public docstring to state that fp64 inputs
  retain fp64 output instead of claiming all outputs are fp32.
- Keep Torch/Triton forward/backward helpers as backend boundaries and keep
  _CHUNK_ELEMENTS/_BLOCK_V as memory-chunk and kernel-launch dimensions. Preserve
  dtype decisions, autograd behavior and kernel selection; no owner class is added.
- Fused linear log-prob suite: 24 passed. Touched-file Ruff checks pass. This
  readability correction does not change math or finish the repository audit.

## Native attention uses the trunk's cache vocabulary directly

- Rename local kv/past and the private forward parameter to past_key_values,
  matching the HF trunk argument and output attribute. Pass sequence_states
  directly to ar_concat_rows, which already accepts a Sequence, without an
  unnecessary list conversion.
- Keep _forward shared by prefill/step and _last_token_hidden as the adapter for
  the two supported output formats. Preserve split/concat behavior, missing-cache
  errors and the cross-backend payload interface; no new cache class is needed.
- Native attention, Janus KV decode and Emu3 structural schedule suites: 11
  passed, 2 dependency warnings. Touched-file Ruff checks pass. The repository
  clarity audit remains ongoing.

## Decoder dimensions read fallback config only when needed

- Replace eagerly evaluated getattr defaults in the three head-count/dimension
  readers with explicit attribute access and fallback on AttributeError. An
  attention layer's own dimensions no longer require unrelated trunk.config
  fields to exist. Three regression cases failed before the fix.
- Preserve integer conversion, absent-attribute fallback and shared decoder
  methods. Keep the standalone RoPE helpers as mathematical operations; no new
  dimension resolver class or generic getter is introduced.
- Decoder-contract and token-AR binding suites: 19 passed, 16 dependency
  warnings. Touched-file Ruff checks pass. This fixes fallback evaluation order;
  the repository-wide audit remains incomplete.

## Shared-GPU installation documents the missing block-table dependency

- Confirmed uvloop was absent from the active environment while uv.lock already
  records 0.22.1. Installed only uvloop==0.22.1 with --no-deps; Torch and vLLM
  versions were unchanged. The previously failing real CUDA KV-cache test then
  passed. This resolves the import failure recorded below.
- Add uvloop to README's explicit shared-GPU installation command and matching
  pyproject comment. Describe the imported dependency closure rather than claim
  every package is a direct vLLM requirement. No production abstraction, new
  core dependency, lockfile change or skip rule is needed for this correction.
- Full tests/nn rerun: 172 passed, 16 dependency warnings in 4.65 seconds. Log:
  /tmp/vrl-nn-clarity-regression-restored.log. The overall repository clarity
  audit remains incomplete despite this now-green subsystem regression.

## Paged-kernel import diagnostics report evidence instead of an ABI guess

- Joint tests/nn regression produced 170 passes and one failure: installed vLLM
  could not import vllm.v1.worker.block_table because uvloop was absent. The
  adapter's message guessed a Torch/CUDA ABI mismatch despite the actual cause.
- Include the originating exception type and message in ARAttentionUnavailable,
  retaining exception chaining and the framework adapter boundary. No dependency
  changes, new helper, fallback backend or test-skip rule is introduced.
- Import-gate tests: 5 passed, covering both ABI-style ImportError and missing
  module diagnostics. Touched-file Ruff checks pass. Isolated real-kernel rerun
  still fails, now reporting ModuleNotFoundError: No module named 'uvloop'.
  Logs: /tmp/vrl-nn-clarity-regression.log and
  /tmp/vrl-paged-kernel-import-failure.log. Full NN regression is not green and
  real-kernel validation remains outstanding; the repository audit continues.

## AR cache splitting diagnoses scalar tensors at the row boundary

- Reject scalar tensors explicitly before indexing shape[0], reporting the
  missing leading batch dimension rather than an incidental tuple IndexError.
  Direct, nested and scatter regressions failed before the change; scatter
  coverage also confirms that a rejected value leaves the original row intact.
- Keep shared split/concat functions used by GLM, native attention and ARCacheRows,
  and retain HF cache conversion helpers as framework adapters. Do not force
  non-owning callers through a mutable cache container merely to remove helpers.
- Cache-row and native attention suites: 35 passed. Touched-file Ruff checks
  pass. Valid tensor slicing and cache reconstruction are unchanged; the wider
  repository clarity audit remains incomplete.

## AR backend factory name states that it constructs an object

- Rename resolve_attention_backend to build_attention_backend, updating the
  token executor, module exports and tests. It constructs a backend, not merely
  a resolved name. Remove the unrelated SGLang label from the name-reader doc.
- Keep both concrete builder functions and the shared _lm_trunk adapter check
  for consistent cross-family construction. Keep _ATTENTION_BACKENDS as the
  explicit protocol-name set. No additional factory class or legacy alias is
  introduced; backend selection and constructor arguments remain unchanged.
- Backend-selection and token-AR binding suites: 14 passed, 2 skipped, two
  dependency warnings. Touched-file Ruff checks pass. This rename updates repo
  callers; external imports must use the new name. The full audit continues.

## AR prefill validates the token budget before block reservation

- Use require_exact_int for ARAttentionPrefillInput.max_new_tokens, minimum one.
  A positivity comparison previously admitted floats and booleans, or emitted
  an unscoped comparison TypeError for strings. Three regression cases failed
  before the fix; nonpositive rejection remains covered.
- Keep validation in the existing input dataclass. Keep _require_embed_mask_batch
  shared between prefill and step inputs: neither payload should own the other's
  shape rules. Do not add an inheritance layer solely to remove that function.
- Input contract, decoder contract and native attention suites: 13 passed.
  Touched-file Ruff checks pass. These checks establish the input boundary, not
  real paged-kernel correctness; the wider repository audit remains incomplete.

## FP8 scalar and row scaling belongs to Fp8Linear

- Move _amax_scale into Fp8Linear as a private static method. Its only callers
  are weight requantization and the rowwise/tensorwise activation branches.
  Keep one shared formula rather than duplicating it across those branches.
- Preserve reduction dtype, floor, scaling, output handling and all GEMM recipe
  paths. Keep FP8_BLOCK as the kernel block dimension and FP8_E4M3_MAX as the
  numeric format boundary. No general scale utility or new owner is introduced.
- FP8 and shared quantized-linear suites: 33 passed, 14 dependency warnings.
  Touched-file Ruff checks pass and formatting is applied. This ownership cleanup
  does not establish a training-quality result or finish the repository audit.

## NVFP4 shape eligibility belongs to Fp4Linear

- Move _alignment_error into Fp4Linear as a static method, shared by its
  constructor and class-level can_replace check. Those are its only callers;
  no module alias or separate shape-policy class is needed.
- Preserve one alignment rule for direct construction and replacement traversal,
  along with existing messages and supported shapes. Keep format constants and
  standalone tensor-layout/quantization functions as the numeric representation
  boundary; do not reorganize them solely to minimize free-function count.
- FP4 and shared quantized-linear suites: 34 passed, 14 dependency warnings.
  Touched-file Ruff checks pass and formatting is applied. This ownership change
  does not alter quantization math or complete the wider repository audit.

## Drift-source enablement stays inside its sole diagnostic consumer

- Inline _sampling_enables into unguarded_drift_sources's traversal. The local
  value/enablement check replaces a standalone predicate with no other caller.
  Preserve truthiness behavior and mapping enabled defaults; this diagnostic
  does not replace TeaCacheConfig's strict parsing of user inputs.
- Keep unguarded_drift_sources as a shared cross-config check. Keep ROLLOUT_PASSES
  as the ordered optimization registry and REQUEST_SCOPED_DRIFT_SOURCES as the
  deliberately isolated diagnostic taxonomy. No class or extra schema is added
  merely to reduce free-function count.
- Optimization and config suites: 348 passed, 14 dependency warnings. Touched-file
  Ruff checks pass. Repository-wide clarity completion remains unproven.

## TeaCache descriptions follow the implemented signal and measured limits

- Replace promotional speed rankings and future-sprint prose with the actual
  latent-change accumulation, reuse and forced-forward rules. Retain upstream
  approach attribution and explain that skip counters alone do not establish
  training throughput or acceptable replay drift. Document the metric's host
  synchronization without claiming its cost is negligible.
- Rename the private _acc state to _accumulated_change. Keep the shared
  relative_l1_change function because runtime and offline analysis must measure
  the same signal. No new class, helper or policy table is introduced; threshold,
  cache behavior and correction-policy validation are unchanged.
- Denoise and rollout optimization suites: 82 passed, 14 dependency warnings.
  Touched-file Ruff checks pass. The repository clarity audit remains ongoing.

## Paged-attention adapter defaults only absent kernel block sizes

- Replace truthiness fallback for kernel_block_size with an explicit None check.
  An explicit zero previously became the configured block size before reaching
  the backend. Keep explicit values intact; backend validity is not established
  by the forwarding fake. The zero-forwarding regression failed before the fix.
- Keep VllmPagedAttentionKernels' thin methods as the external framework adapter
  and _REQUIRED_MODULES as its internal API dependency boundary. Do not move them
  into family code or duplicate backend validation in a new configuration class.
- Import-gate and Janus/NextStep binding suites: 5 passed, 2 skipped, 2 dependency
  warnings. Touched-file Ruff checks pass. This verifies argument forwarding,
  not a full real-kernel acceptance run; the repository clarity audit continues.

## Direct denoise configuration rejects unknown execution modes

- Use the existing DenoiseMode literal for DenoiseLoopConfig and validate it
  at construction. Unknown strings and None previously reached the loop's
  non-native branch and selected SDE execution instead of reporting an error.
- Keep the existing native/SDE loop branches and shared mode vocabulary. No
  separate constant list or validator helper is introduced. Normal parsed
  request behavior is unchanged; direct construction now has the same allowed
  modes. Four malformed-mode regressions failed before the fix.
- Denoise and full-sequence binding suites: 183 passed. Touched-file Ruff lint
  passes and formatting is applied. The overall repository audit is ongoing.

## Denoise configuration owns exact sample identity validation

- Validate sample_start as a nonnegative exact integer and sample_count as a
  positive exact integer in DenoiseLoopConfig using require_exact_int. Remove
  the downstream int(config.sample_count) coercion in buffer allocation.
- Eight malformed direct-construction cases failed before the change; also
  cover zero sample_count rejection. Normal executor construction already gets
  valid identity from GenerationSampleBatch, but direct loop configuration now
  enforces the same semantics rather than coercing or failing during sampling.
- Keep existing config/buffer owners and the probe's partial execution with full
  allocation. No extra helper or wrapper class is introduced. Denoise and full
  sequence binding suites: 179 passed. Touched-file Ruff checks pass. The full
  repository audit remains incomplete.

## Denoise loop local names follow execution order

- Rename latents_ori to latents_before_step and prev_latents to next_latents in
  the shared loop. The latter is the produced action and next state, despite
  the scheduler/SDE API calling that output prev_sample in diffusion time.
- Preserve external prev_sample/prev_sample_mean names and all sampling math,
  cache inputs, state assignment and trajectory writes. Keep run_denoise_loop
  as the shared execution entry and DenoiseTrajectoryBuffers as the storage
  owner; no new class or helper is needed for these local temporal names.
- Denoise-step and full-sequence binding suites: 170 passed. Touched-file Ruff
  checks pass. This is a local readability change; the full audit is ongoing.

## Weight manifest shape parsing rejects accidental scalar shapes

- Require list/tuple shapes with nonnegative exact integer dimensions in the
  receiver constructor. Previously an iterator was exhausted during validation
  and stored as a scalar shape; empty strings and mappings also became scalars.
  Report the weight name and shape field instead of an unscoped iteration error.
- Keep the existing staging owner and sender manifest function. Preserve empty
  list scalar shapes and zero-sized dimensions; add a scalar to the existing
  exact round-trip test. No new parser class or validation helper is introduced.
- Five malformed-shape regressions failed before the fix. Transfer and worker
  versioned-slot suites now pass: 31 tests. Touched-file Ruff checks pass. This
  checks shape representation, not every possible malformed manifest structure;
  the broader repository audit remains incomplete.

## Weight staging enforces its declared transfer identifier type

- Replace transfer-ID truthiness with an explicit nonempty-string check in
  StagedWeightTransfer. Previously integers, booleans, bytes and lists could
  enter worker staging despite the string protocol. Four worker-boundary
  regressions failed before the fix and now verify no staged/live state changes.
- Keep validation on the existing receiver owner and keep sender-side manifest,
  chunk and bucket functions separate: they implement shared transport data
  processing, not receiver lifecycle. No validator wrapper or new class is needed.
- Preserve valid IDs, exact policy-version checks, chunk ordering, receiver
  buffer ownership and complete-before-install behavior. Transfer and versioned
  worker suites: 26 passed. Touched-file Ruff checks pass. The repository clarity
  audit remains incomplete.

## Engine combination documentation states the single-rank boundary

- Correct RayGenerationEngine.remote's claim that combining one result always
  yields that result. The implementation skips combine for single-rank engines;
  document that fact and direct validation/transformation needed on every result
  to the caller after awaiting. Preserve the raw-ref fast path and timing.
- Keep uniform_rank_result as an aggregation-policy callback and rank_handles
  as the shared launcher/session lifecycle view. Neither needs an owning wrapper
  class solely to reduce module-level function count.
- Engine and weight-sync suites: 55 passed, one Ray dependency warning.
  Touched-file Ruff checks pass. This is a documentation correction, not a
  runtime behavior change or proof that the overall audit is complete.

## Ray driver device discovery belongs to its validation owner

- Move _get_device, _iter_parameter_devices and _cuda_device_index into
  RayGenerationConfig. Their only production entry is validate_driver_state;
  no external production or test caller imports these private helpers.
- Retain distinct helpers for property access, recursive traversal and CUDA
  index parsing. The traversal uses class dispatch for recursion and device
  access; the two independent transformations are static methods. No new class
  or generic device utility is introduced.
- Preserve property-error propagation, cycle protection, parameter inspection,
  current-device handling, lazy Torch import and ownership/memory guards.
  Runtime-config and trainer-memory-guard suites: 55 passed, 3 dependency
  warnings. Touched-file Ruff checks pass. The full clarity audit is ongoing.

## Generation and rollout joint regression after worker cleanup

- Reviewed sample_batches helpers against diffusion, token AR and chunk AR
  gatherer call sites. Keep the shared coverage ordering, row validation, replay
  merge and context comparison functions: these enforce cross-family batch
  semantics and prevent each binding from implementing different rules.
- Keep ContinuousRolloutSchedule's thin facade methods. Policy export occurs on
  the trainer thread; async admission and collection belong to the owner loop.
  Their small bodies express a thread boundary, not unnecessary indirection.
- Existing gatherer tests cover reordered batches, per-batch replay alignment,
  static/ragged value distinctions, dtype mismatch and context mismatch. No
  wrapper class or production change is justified by function count alone.
- Joint tests/generation and tests/rollouts regression: 1220 passed, 2 skipped,
  4 warnings in 46.17 seconds. Log:
  /tmp/vrl-generation-rollouts-clarity-regression.log. This verifies the combined
  tested paths after recent edits; it does not prove repository-wide clarity
  completion or an end-to-end training throughput improvement.

## Worker default device discovery preserves failures

- Remove the broad exception-to-CPU fallback in _executor_device. Torch import
  or CUDA discovery failures must propagate instead of inventing a CPU restore
  or profiling target. A regression reproduced a swallowed CUDA probe error.
- Keep the model-device preference, lazy Torch import and availability-based
  default when no device is supplied. Keep the shared worker method because
  sleep and profiling both use this decision. This does not require every model
  to declare a device or change property discovery semantics.
- Generation execution suite: 218 passed, including both default-device choices
  and error propagation. Touched-file Ruff checks pass. The wider repository
  clarity audit remains incomplete.

## Worker CPU transfer uses method and state shapes matching its dependencies

- Make GenerationWorkerCore._to_cpu a staticmethod: it consumes only its value
  argument, never the class. Replace its single-key mutable dictionary with a
  nonlocal boolean recording whether any CUDA copy was submitted.
- Keep this worker-owned transfer boundary and its local tensor callback. It
  completes copies before returning, unlike the pipeline enqueue helper; their
  stream synchronization and non-CUDA leaf handling remain distinct. No shared
  class or new file is introduced merely to unify similar-looking code.
- Preserve pinned allocation, detach behavior, lazy imports and synchronization.
  Generation execution tests: 215 passed. Touched-file Ruff checks pass. Device
  discovery fallback behavior is unchanged and still requires separate review;
  the repository audit remains incomplete.

## Pipelined CPU copy naming exposes submission rather than completion

- Rename the private synchronous helper _move_tree_to_cpu_async to
  _enqueue_cpu_copies and update both pipeline call sites and test references.
  Its docstring now states that returned CPU buffers require a completion wait,
  that the caller must establish producer ordering, and that non-CUDA tensors
  remain unchanged. There is no coroutine or implicit wait in this helper.
- Keep the helper: both pipeline teardown sites share pinned allocation,
  nonblocking copy and record_stream lifetime protection. Keep the schedule
  factory and topology guard as selection and cross-type validation boundaries;
  adding wrapper classes would not clarify those responsibilities.
- Preserve stream/event ordering, exception cleanup, lazy imports and result
  layout. Pipeline unit and CUDA test files passed: 12 tests. Touched-file Ruff
  checks pass; no old helper references remain in production or tests. This
  naming change does not establish a throughput improvement or finish the audit.

## Offline DPO projection consumes parsed actor scalars directly

- Remove redundant int/bool/str conversions of batch size, accumulation steps,
  optimizer selection, learning-rate scaling and prediction type in from_root.
  ActorSection already declares strict integer/boolean fields and a string
  prediction type; the production caller supplies the parsed RootConfig.
- Keep the local required-field reader for consistent actor path errors and
  keep optimizer compatibility checks. No new helper class, schema vocabulary
  or change to optimizer math is needed. Float normalization is outside this
  change, as are direct dataclass construction and bypassed schema validation.
- Existing offline builder and timestep/restore suites: 33 passed, two dependency
  warnings. Touched-file Ruff checks pass. The broader audit remains incomplete.

## Rollout selection owns its tensor callback

- Move the sole-use _select_tensor_tree callback into select_batch and traverse
  the complete extras dictionary once. The callback closes over the detached
  selector and batch size instead of forwarding both through a module helper
  for every extras entry. No public API or new class is introduced.
- Keep map_tensor_tree as the shared recursion boundary for generation copies,
  trajectory storage, rollout selection and weight snapshots. Keep the private
  tensor-only device mover shared by extras and context: replacing it with the
  duck-typed trajectory mover would broaden which objects receive to().
- Preserve leading-dimension selection, scalar and non-sample tensor handling,
  context ownership and trajectory reconstruction. Container support expansion
  and merging distinct transfer policies are outside this change; shared
  traversal consistency remains more valuable than removing every helper.
- Existing trajectory, online reward/update-flow and deferred replay suites:
  102 passed. Touched-file Ruff checks and git diff --check pass. No old helper
  references remain in production or tests. The repository audit is ongoing.

## Replay reference parsing stays with its sole consumer

- Inline reference qualification and splitting in replay_tensor_dict, removing
  _canonical_tensor_ref and the module-level _split_ref. The latter accepted a
  kind argument that was always "tensor". Reuse the qualified reference in axis
  slicing instead of reconstructing it after parsing.
- Keep tensor_ref as the shared reference-format boundary used by construction
  and validation. Keep axis slicing and cross-segment rejection unchanged;
  they express independent replay constraints. No new reference class or file
  is needed for this local string operation.
- Preserve local and qualified references, first-dot splitting, malformed-name
  errors and exception chaining. Schema format changes and merging validation
  with runtime slicing are non-goals; those boundaries remain useful.
- Existing trajectory and rollout replay suites: 140 passed. Touched-file Ruff
  checks and git diff --check pass. Neither removed helper has remaining
  production or trajectory-test references. The wider audit remains ongoing.

## Replay segment defaults distinguish omission from an explicit empty name

- Replace truthiness fallback in replay_tensor_dict with an explicit None check.
  An empty segment name previously selected the primary segment silently; it
  now raises the same unknown-segment error as any other absent name. Document
  the omission rule at the public method.
- Keep primary_trainable_segment_name as the shared checked accessor, and keep
  explicit production segment names and omitted-name algorithm calls unchanged.
  No validator class, helper function or constant is introduced. Axis behavior,
  reference parsing and cross-family replay interfaces are outside this fix.
- Added a public-boundary regression that failed before the change for the
  empty name, plus checks for unknown names and omitted/None equivalence.
  Trajectory, rollout replay and chunk-autoregressive binding suites: 167 passed.
  Touched-file Ruff checks and git diff --check pass. Wider audit remains open.

## Denoise recording casts at the preallocated destination

- Remove explicit dtype conversions immediately before copy_ for actions,
  log-probabilities, KL, previous means and reference predictions. Destination
  buffers already own their dtype; copy_ performs the conversion while writing.
  Preserve detach calls and the existing absolute-value operation for KL.
- Keep record_step as the transition-write boundary and _expand_timestep for
  shape normalization. No helper or class is added. Scheduler mathematics,
  TeaCache, buffer allocation and timestep placement remain unchanged.
- Added mixed-dtype recording coverage for float16, bfloat16 and float32 buffers
  on CPU and CUDA, including optional caches, exact cast values and no-grad
  outputs. All six cases ran. Denoise-step and full-sequence binding suites:
  189 passed. Touched-file Ruff and git diff --check pass. No training throughput
  improvement is claimed without a benchmark; the wider audit remains open.

## Denoise step telemetry reports execution rather than allocation

- Set diffusion_num_denoise_steps from num_steps_to_run. A probe limited to one
  of three scheduled steps previously reported three because the counter read
  the full replay-buffer width. Normal full executions retain their count.
- Keep full replay-buffer allocation and byte counters: memory probing needs
  that capacity even during shortened execution. Keep the common denoise loop
  and probe adapter; separating them would duplicate execution semantics. No
  new telemetry class or constant is needed. Probe output handling, scheduling
  mathematics and capacity calculations are outside this change.
- Added tests for truncated, oversized and omitted execution limits, checking
  that full capacity is retained. The truncated case failed before the fix.
  Denoise-step, full-sequence binding and batch-memory shadow suites: 219 passed.
  Touched-file Ruff and git diff --check pass. The repository audit remains open.

## Sample planning expresses batch boundaries directly

- Replace synchronized remaining/start mutation in GenerationSampleBatch.plan
  with range over sample starts. The final width is the smaller of configured
  width and samples still available at that start. Construct through cls, as
  the existing classmethod signature declares, rather than naming the base type.
- Keep strict positive-integer input checks, prompt-major ordering and short
  final batches. Preserve shared gather helpers and the OOM retry function:
  they enforce cross-family consistency and execution cleanup rather than
  belonging to the batch identity record. Moving them into a class is not a
  goal, and split behavior is unchanged.
- Existing generation execution and binding suites: 369 passed, 16 dependency
  deprecation warnings, process exit zero. Coverage includes prompt-major
  planning, invalid widths, OOM splitting and family gather paths. Touched-file
  Ruff and git diff --check pass. The repository-wide audit remains open.

## Rank rendezvous rejects non-string addresses before URL construction

- Require master_addr to be a non-empty string in RankGroupSpec. Previously,
  truthy integers, booleans, bytes and lists passed validation and were formatted
  into TCP URLs. Four constructor regressions failed before this fix.
- Keep init/destroy free functions as the process-group lifecycle and lazy
  torch.distributed import boundary. Backend names remain the supported protocol
  choices. No network validator abstraction or hostname restriction is added.
- Rename the CPU smoke test to all_gather, matching the actual collective, and
  remove its implication that it establishes NCCL/GPU behavior. Preserve real
  two-process rendezvous, RNG synchronization and teardown coverage.
- Rank-group, worker sleep and batch-memory shadow suites: 101 passed, including
  the real CPU collective smoke. Touched-file Ruff and git diff --check pass.
  Address reachability and NCCL verification are outside this change; the wider
  repository audit remains open.

## Schedule entrypoint uses direct resource predicates and domain names

- Rename requested_arm to reward_mode at schedule construction and remove
  redundant bool calls around topology predicates in if statements. Preserve
  enum conversion, branch order, diagnostics and GPU isolation requirements.
- Keep build_rollout_schedule as the runtime-selection factory and
  validate_rollout_schedule_topology as a guard spanning configuration and
  resolved resources. Keep coordinator query methods as a consistent lifecycle
  surface rather than removing an individual forwarding method in isolation.
- Reviewed the strict phase manager: it rethrows body failures after cleanup
  and combines cleanup failures explicitly. No changes to phase ordering,
  cancellation, weight synchronization or protocol shapes are part of this
  expression cleanup. No new class or constants are introduced.
- Existing orchestration suite: 272 passed. Touched-file Ruff and git diff
  --check pass. This is behavior-preserving cleanup, not evidence that the
  repository-wide audit is complete.

## Continuous settings documentation matches rejection and clock semantics

- Correct ContinuousRolloutSettings documentation: it rejects nonpositive policy
  windows; it cannot route callers to another schedule. State that mechanisms
  retain capacity/version validation rather than claiming all range validation
  exists exclusively at the configuration boundary.
- Clarify that completed_at requires a shared monotonic clock domain to compute
  age and is not a portable cross-machine timestamp. The queue remains local to
  its owner; no timestamp conversion or distributed clock protocol is added.
- Keep GeneratedRolloutCapacity as one owner of reservation/waiting/scoring
  accounting and StalenessPolicy as the shared producer/consumer version rule.
  Their methods are meaningful state operations and consistency boundaries, not
  candidates for flattening solely to reduce function count. Behavior is unchanged.
- Existing continuous orchestration suite: 209 passed. Touched-file Ruff and git
  diff --check pass. This validates the inspected subsystem, not completion of
  the repository-wide clarity audit.

## Continuous consumption derives version from selected receipts

- Rename _select_iteration to _take_ready_groups to expose its queue-removal
  side effect. Return only the selected receipts, removing the duplicate version
  tuple member and _build_iteration version argument. The builder reads the
  version from the first receipt after homogeneous-version validation.
- Keep ready-version validation, distinct-slot checks, prompt ordering and
  iteration construction as consumer-owned methods. The selected group set is
  the source of truth; no new selection wrapper or dataclass is introduced.
- Preserve queue ownership transfer, public collect_iteration arguments,
  backpressure, timeout behavior and metric values. Existing continuous suites:
  209 passed. Touched-file Ruff and git diff --check pass; no removed method name
  remains in production or tests. The broader repository audit is incomplete.

## Completed receipts keep queue identity and charged size stable

- Freeze ContinuousRolloutItem fields. Queue admission charges nbytes and removal
  reads the same field; consumer selection likewise relies on stable batch and
  policy identity. Production constructs each receipt once and does not reassign
  its fields. Document that referenced batch and stats objects remain mutable.
- Keep queue operations and capacity accounting separate: the former retains
  ready payloads, the latter tracks in-flight reservations. No new wrapper or
  custom assignment guard is introduced. Payload immutability, recalculating
  byte estimates and changing group-ID remapping are outside this change.
- Three public admission regressions showed that byte count, batch identity and
  policy version could be reassigned before this change. They now reject writes
  and verify removal restores zero charged bytes. Full orchestration suite:
  275 passed. Touched-file Ruff and git diff --check pass. Wider audit remains open.

## Generation and rollout integration after receipt and planning cleanup

- Correct the producer freshness comment: absent versions bypass version-gap
  gating; future versions pass the producer's too-stale check but are rejected
  by consumer validation. Do not claim both cases fail downstream.
- Keep production handoff and ready-queue admission separate from generated
  capacity accounting: their payload ownership and lifetimes differ. Retain
  the CPU placement target and shared retry backoff limit as concrete execution
  policy, without adding another configuration or wrapper object.
- Revalidated recent planning, denoise recording, rank-group, resolver and
  continuous-receipt changes together. Full tests/generation and tests/rollouts:
  1260 passed, 18 warnings in 49.63 seconds, process exit zero. Full output is in
  /tmp/vrl-generation-rollouts-clarity-followup.log. Touched-file Ruff and git
  diff --check pass. This is cross-layer regression evidence, not a complete
  repository audit or an end-to-end training throughput measurement.

## Shared config helpers document their actual conversion depth

- Correct plain_mapping documentation: OmegaConf conversion and Pydantic
  model_dump preserve their existing handling, while ordinary Mapping inputs
  receive only a top-level copy. Point recursive ordinary-container callers to
  to_builtin_deep without changing either function's behavior.
- Describe to_builtin_deep key/tuple normalization and unchanged arbitrary
  leaves; remove the unsupported claim that tuple handling matters only for
  tests and the implication that arbitrary payloads become serializable.
- Keep both helpers as shared conversion boundaries with different presence and
  recursion semantics. Keep JSON helpers and _write_atomically: they centralize
  atomic publication, exclusive creation and cleanup across writers. Adding a
  loader/writer class or merging these contracts is outside this documentation
  correction; no constants or new abstractions are introduced.
- Existing config-helper and JSON-file tests: 13 passed. Touched-file Ruff and
  git diff --check pass. Wider repository audit remains incomplete.

## Frame sampling validates explicit counts before the no-op branch

- Reuse require_exact_int for positive num_frames in sample_frames. Zero
  previously produced an empty selection, oversized fractions silently returned
  the input, and other malformed values failed through backend-specific errors.
  None still means no limit; valid short inputs retain object identity.
- Keep shared media functions as cross-reward/evaluation conversion boundaries.
  No sampler class or additional validator is introduced. Frame-index selection,
  image/video decoding and layout/range conversion remain unchanged.
- Six malformed-count regressions failed before the fix; added an ordered
  sampling and no-op check. Media and reward suites: 424 passed, 5 skipped,
  7 warnings. Touched-file Ruff and git diff --check pass. Skipped tests are not
  evidence of coverage, and the broader repository audit remains incomplete.

## GPU busy reports own their text and dictionary views

- Move format_report and report_to_dict onto the existing GpuBusyReport as
  to_text and to_dict. Update CLI and test consumers and remove obsolete exports;
  no compatibility forwarding layer or new report base class is introduced.
- Keep interval algebra and SQLite analysis functions: they compute across raw
  captures before a report exists. Keep nanosecond/percentage rendering helpers
  for consistent units and zero-denominator handling, and the API-version regex
  as an input-format normalization boundary.
- This supersedes the rendering-placement rationale in the historical homeless
  function sprint: current ownership is clearer on the report, and this module
  now lives under scripts/perf. Historical decisions are not current completion
  evidence. Query behavior, union math, text layout and JSON schema are non-goals.
- Existing Nsight report tests: 15 passed. A synthetic SQLite report's complete
  text and sorted JSON match saved pre-change outputs exactly. Touched-file
  Ruff and git diff --check pass; obsolete names are absent from Nsight code and
  tests. No real capture or GPU throughput benchmark was run. Wider audit remains open.

## Nsight analysis validates report limits before opening captures

- Require nonnegative integer top_gaps, top_nvtx and min_gap_ns at analyze's
  public boundary. Negative top values otherwise act as Python negative slice
  endpoints; booleans/fractions have inconsistent downstream behavior. Reuse
  require_exact_int rather than adding another validator abstraction.
- Preserve zero top limits, which omit corresponding rows, and zero minimum
  gap. Keep interval mathematics and device selection unchanged. Correct CLI
  help and analysis documentation to say the default device has the most
  kernels in the window, without claiming it has the highest busy fraction.
- Nine early-validation regressions failed before this change; a zero-limit
  test preserves the empty-list control. Nsight report suite: 25 passed.
  Touched-file Ruff and git diff --check pass. This does not validate a real
  Nsight capture or complete the broader repository audit.

## Nsight CLI validates milliseconds before integer conversion

- Validate --min-gap-ms before int conversion: a tiny negative value previously
  truncated to zero, while infinity and multiplication overflow escaped as
  OverflowError. The CLI now reports an argparse usage error naming the option
  when the value is negative or its nanosecond conversion is nonfinite.
- Keep conversion in main as the CLI adapter and retain analyze's integer
  boundary. No separate converter function or config object is introduced.
  Nonnegative finite conversion keeps its existing truncation behavior; report
  schema, selection rules and profiling collection are unchanged.
- Five invalid-input regressions failed before the fix. Nsight report suite:
  30 passed. A valid zero-threshold CLI call against the synthetic SQLite capture
  completed with text and JSON output. Touched-file Ruff and git diff --check
  pass. This is not real capture validation; the wider audit remains open.

## CountGD patch specifications own text replacement

- Move _replace_text_exact onto the existing _PatchSpec as apply_to_text. The
  patch already owns path and ordered replacements; callers now supply only
  source text. Remove the free helper and update its production caller and test.
- Preserve replacement-count drift checks, newline normalization, subsequent
  file hashing and patch order. Keep download/subprocess/file boundaries as
  functions; retain revision, digest and compatibility tables as reproducible
  installation inputs. No installer superclass or new configuration is added.
- Existing installer tests: 8 passed, including drift rejection and staged
  publication checks. Touched-file Ruff and git diff --check pass, and the old
  helper name is absent from production/tests. These tests use isolated fixtures;
  no dependency installation, download or real model smoke was run. The broader
  script and repository audits remain incomplete.

## CountGD archive URL is a fixed install input

- Replace the argument-free _source_archive_url function with
  _SOURCE_ARCHIVE_URL beside the source repository and digest. Download and
  manifest construction share the same revision-derived constant; URL text is
  unchanged. This constant represents a pinned artifact boundary, not a workflow
  business vocabulary or a dynamic runtime decision.
- Keep the isolated environment lock and its payload/digest helpers: lock
  identity includes target platform, bootstrap and all distributions rather than
  belonging to an individual LockedDistribution. No new owner class is needed.
- Existing installer suite: 8 passed. Touched-file Ruff and git diff --check
  pass; no old function references remain in production/tests. No downloads or
  environment changes were performed. The wider audit remains incomplete.

## SANA reward definitions own persisted report projection

- Move reward_model_record to RewardModelDefinition.to_report_record. Report
  production, provenance validation and tests call the existing data owner;
  remove the free projection function without a compatibility alias.
- Preserve every report field, device/dtype string conversion and provenance
  reference semantics. Keep protocol versions, seed grid and asset digests in
  this isolated report-contract module. Score aggregation and cross-config
  comparisons remain functions spanning collections rather than one definition.
- Existing aesthetic checkpoint evaluation and curve verdict suites: 45 passed.
  Touched-file Ruff and git diff --check pass; old projection references are
  absent from production and tests. No real checkpoint generation/scoring was
  run, and the wider repository audit remains incomplete.

## SANA sampling reads its declared protocol directly

- Remove resolve_sampling, which accepted no input and only copied the already
  declared OFFICIAL_SAMPLING_PROTOCOL. Generation explicitly copies that mapping;
  report validation compares it directly. Tests access the inference-owned
  protocol instead of the report module's incidental import.
- Preserve mutation isolation for generation and drift-test inputs. Keep seed
  grid and evaluation curve record helpers as shared persisted-schema builders,
  and keep the isolated sampling protocol constant as the authoritative identity.
  No sampling resolution class, compatibility alias or protocol change is added.
- Aesthetic checkpoint evaluation and curve verdict suites: 45 passed.
  Touched-file Ruff and git diff --check pass. This cleanup does not run real
  generation or complete the repository-wide audit.

## SANA autocast checks preserve the selected device and query failure

- Remove the sole-use _is_autocast_enabled wrapper and call the device-qualified
  torch API directly, consistent with vrl/models/precision.py. The wrapper
  previously swallowed TypeError and retried without a device, obscuring the
  original failure and potentially querying a different autocast domain.
- Keep scheduler configuration access as an adapter for mapping/attribute
  representations. No compatibility class or replacement fallback is added;
  scheduler identity, sampling protocol and active-autocast rejection remain.
- Added a regression that failed before the change, proving query-error identity
  and exactly one device-qualified call. SANA comparison, aesthetic evaluation
  and curve verdict suites: 71 passed. Touched-file Ruff and git diff --check
  pass. No real model generation was run; the repository audit remains open.

## SANA generation requires an explicit positive integer image count

- Replace the lower-bound-only num_images check with shared require_exact_int.
  Booleans and fractions previously passed that check, while strings failed
  through comparison TypeError. Invalid counts now fail before scheduler or
  pipeline access with a named argument error.
- Keep the shared inference function, scheduler adapter and protocol constants;
  no new validation wrapper is introduced. Valid sampling, seeds, pipeline
  placement and image conversion are unchanged.
- Five invalid-count cases cover the boundary; three failed before the fix and
  nonpositive integers already failed correctly. SANA comparison, aesthetic
  evaluation and curve verdict suites: 76 passed. Touched-file Ruff and git
  diff --check pass. Real model generation and wider audit completion remain
  outside this local verification.

## GEMM breakdown owns its textual view

- Move format_report onto the existing Breakdown as to_text. Update the CLI,
  embedded usage example and exports. No new rendering class or forwarding alias
  is introduced; the data object already supplies every input to this view.
- Keep classification, profiler instrumentation and fusion helpers separate:
  they consume models/events before a breakdown exists. Keep PROJECTION_ORDER
  as the shared category/display vocabulary. Measurement and percentage formulas
  remain unchanged in this ownership slice, including existing zero-total behavior.
- Four complete before/after text comparisons passed for synthetic CPU/CUDA
  report data with and without unclassified modules. These are rendering checks,
  not actual GPU profiling. Touched-file Ruff and git diff --check pass; the old
  formatter name is absent from this module. Wider audit remains incomplete.

## GEMM reports retain measured zero instead of a denominator sentinel

- Preserve the measured total in Breakdown.to_text. Previously `sum(...) or
  1.0` changed a zero measurement into a displayed 1.0 microsecond total and
  100 percent. Guard percentage division separately; zero time now displays
  zero total and zero percentage.
- Keep report ownership, category order, profiling hooks and measurement
  collection unchanged. No helper class or generic percentage utility is added.
- Two zero-total regressions failed before the fix; positive totals remain
  covered for both CPU/CUDA-labelled synthetic reports. GEMM and Nsight report
  tests: 34 passed. Touched-file Ruff and git diff --check pass. Synthetic CUDA
  labels are not evidence of GPU execution. Wider repository audit remains open.

## Projection profiling validates execution counts before warmup

- Reject noninteger/negative warmup and noninteger/nonpositive active counts
  before any forward executes. Replace max-based clamping with direct ranges
  after shared require_exact_int validation. Invalid active values no longer
  execute warmup before failing or silently become one measured iteration.
- Keep the profiler function and instrumentation context as model execution
  boundaries. No configuration class or new helper is introduced; valid default
  counts, measurement attribution and device synchronization stay unchanged.
- Seven invalid-count regressions failed before the fix. A real tiny CPU Linear
  profile verifies zero warmup and exactly two active calls. GEMM/Nsight suites:
  42 passed. Touched-file Ruff and git diff --check pass. This does not establish
  GPU profiling accuracy or complete the wider repository audit.

## Profiler field fallback distinguishes zero from absence

- Read the legacy device-time field only when self_device_time_total is absent
  or None. A present zero no longer falls through to a different legacy value.
  Keep _event_self_us as the profiler representation adapter; no new helper or
  compatibility class is introduced.
- Correct PROJECTION_ORDER's comment: it stabilizes counters and tied report
  rows, while classify_linear owns match precedence. Keep the category constant
  and classification behavior unchanged; remove unsupported blanket coverage
  claims from the comment.
- Three adapter cases cover zero, nonzero and None; zero failed before the fix.
  GEMM/Nsight suites: 45 passed, one profiler warning. Touched-file Ruff and git
  diff --check pass. This is synthetic adapter and CPU profiling evidence, not
  real GPU capture validation. Wider audit remains incomplete.

## Projection instrumentation restores model attributes across setup failures

- Include wrapper installation and module traversal inside the cleanup scope.
  Previously an exception before yield left already-installed wrappers active.
  Record whether forward was an instance override, restoring it when present
  and deleting the temporary attribute otherwise. Unwind in reverse order.
- Keep the context manager and local wrapper factory as the temporary model
  instrumentation boundary. No owner class is added. Category assignment,
  timing ranges and forward arguments remain unchanged.
- Three regressions failed before the fix: normal exit, body failure and partial
  traversal all left an instance attribute behind. Existing override restoration
  also passes. GEMM/Nsight suites: 49 passed, one profiler warning. Touched-file
  Ruff and git diff --check pass. Wider repository audit remains incomplete.

## Bottleneck CLI rejects invalid counts before model setup

- Validate --steps >= 1 and --warmup >= 0 immediately after argparse's integer
  parsing. Previously zero steps reached later per-step division, while negative
  warmup silently skipped execution. Report option-specific usage errors before
  configuration loading or model construction.
- Keep CLI orchestration and shared diffusion runtime functions as entrypoint
  and cross-profiler boundaries. No validation helper or config class is added.
  Kernel classification, monitor lifecycle and timing implementation remain
  outside this change and require their own review.
- Three early-boundary regressions failed before the fix. Bottleneck argument
  and shared diffusion runtime suites: 7 passed. Touched-file Ruff and git diff
  --check pass. No actual GPU profile was run; wider audit remains incomplete.

## Bottleneck profiling reaps its monitor on failure

- Enclose profiling and monitor collection in one exception-cleanup scope.
  Forward errors, profiler setup failures and communicate timeouts now kill and
  reap the CLI-owned dmon process before rethrowing the failure.
- Keep lifecycle handling inline with its single Popen owner; no monitor class
  or forwarding context manager is added. The monitoring command, profiling
  window, device counters and report schema stay unchanged.
- Three regression scenarios failed before the fix and now verify cleanup plus
  original error identity. Bottleneck and shared diffusion runtime suites:
  10 passed. Touched-file Ruff and git diff --check pass. Subprocess and GPU
  operations are mocked in these tests; real profiling was not performed.
  Wider repository audit remains incomplete.

## Diffusion performance measurements use an elapsed-time clock

- Replace time.time with time.perf_counter for full-generation latency and the
  bottleneck profile window. Rename t0 to started_at. Adjustable wall-clock
  changes no longer affect these elapsed durations.
- Keep shared execution functions, warmup counts, CUDA synchronization positions
  and median reporting unchanged. No timing class or helper is introduced; the
  measured scopes remain the same.
- An end-to-end timing regression failed before the fix and now verifies the
  performance clock, five total calls and median reporting from controlled
  ticks. Shared runtime and bottleneck suites: 11 passed. GPU operations are
  mocked. Touched-file Ruff and git diff --check pass. Wider audit remains open.

## Diffusion probe sampling preparation owns its inference gradient boundary

- `prepare_sampling_state` now disables autograd for prompt encoding and sampling
  preparation. Previously only subsequent denoising was guarded, so a model with
  trainable parameters could retain preparation graphs in the measured state.
- Keep this shared function: single-step, end-to-end and TeaCache probes consume
  the same preparation contract. Keep the registry builder's lazy import boundary
  and the isolated benchmark prompt constant. No profiler-specific class is added
  to the production model interface; reducing function count alone is not a goal.
- A CPU tensor regression with gradients enabled failed before the fix; it now
  verifies both preparation stages, graph-free output and restoration of the
  caller's gradient setting. The disabled-gradient control also passes. Shared
  diffusion runtime and bottleneck suites: 13 passed. Touched-file Ruff and
  git diff --check pass. Actual model/GPU memory savings were not measured.
  The repository-wide clarity audit remains incomplete.

## Continuous consumer timeout diagnostics follow prompt-batch ownership

- Report prompt_batch_id, target-batch ready_groups/expected_group_count and
  current_policy_version on timeout. Queue-wide occupancy includes prefetched
  batches and previously made an incomplete target look ready to consume.
- Inline the consumer's single-use _timeout_message into its timeout branch,
  where the requested batch identity already lives. Preserve queue occupancy and
  producer error counters in the message. Selection, receipt ownership, capacity
  and staleness behavior are unchanged.
- Keep the producer's separately shared _drain_timeout_message (two callers),
  queue/capacity ownership and the staleness policy. Eliminating every short
  method or merging independent mechanisms is not a goal of this cleanup.
- Two regressions failed before the fix: zero or one current-batch receipts with
  a complete prefetched batch. They verify batch-specific diagnostics and that
  timeout retains all receipts. Continuous orchestration suite: 214 passed.
  Touched-file Ruff and git diff --check pass. The full repository audit remains
  incomplete; these tests do not establish distributed/GPU runtime performance.

## Distributed placement treats engine identities as explicit strings

- Reject a bare str/bytes engine_ids argument before tuple conversion, and
  require every sequence element to be a non-empty string before uniqueness
  checking. A bare ID could previously become character-sized worker identities;
  non-string elements could reach unrelated width-resolution failures or an
  unhashable-type exception.
- Keep validation inline in DistributedExecutionPlanner; no separate validator
  or new carrier is needed. Keep EnginePlan separate from fleet placement, and
  retain shared replay/coverage helpers: direct and distributed execution plus
  different model gatherers depend on those consistent boundaries. The strategy
  Literal remains the protocol vocabulary rather than a duplicated name table.
- Five regression inputs failed before the fix and now report the engine-ID
  boundary before unresolved auto sizing. Generation execution and Ray OOM split
  suites: 251 passed, one Ray environment FutureWarning. Touched-file Ruff and
  git diff --check pass. No production fleet throughput claim follows from this
  change. The full repository clarity audit remains incomplete.

## Generation pipeline submits each batch's teardown in the same iteration

- Move copy submission directly after each successful forward and completion
  callback. The next forward still starts after the previous copy is enqueued,
  without waiting for that copy to finish. This removes prev_result, prev_idx,
  prev_done and the duplicated final-batch teardown branch.
- Append CPU result trees in submission order instead of maintaining indexed
  placeholders. Keep the input-list snapshot, recorded produce fences, source
  record_stream lifetime protection, pending-copy joins and failure stream
  synchronization. None results retain their prior behavior.
- Keep _enqueue_cpu_copies as the tensor-tree/stream boundary and the exported
  pipeline function as the binding adapter. No additional class, helper, policy
  constant or alternate execution path is introduced. CUDA ordering and cleanup
  consistency are more valuable than removing these necessary boundaries.
- Five relevant pipeline suites: 40 passed, including four real-CUDA checks for
  serial equivalence, order, typed CPU results and queryable produce fences.
  Existing tests also cover CPU completion callbacks and failure cleanup.
  Touched-file Ruff and git diff --check pass. Throughput was not measured;
  repository-wide completion remains unproven.

## AR request layout names sample batches consistently

- Rename validate_chunk to validate_batch and chunk_seed_offset to
  batch_seed_offset, updating the shared token executor, Janus R1 and NextStep
  execution paths. These methods consume GenerationSampleBatch, not a temporal
  chunk; the old names blurred sample batching with chunk-autoregressive video.
- Keep both methods on ARRequestLayout: three execution paths share the same
  request range check and prompt-major seed-offset formula. No forwarding alias,
  extra class or per-family copy is added. Shared gather helpers and driver-only
  gatherer modules also remain necessary for cross-family coverage consistency
  and model-free result assembly; reducing their count is not a goal.
- Existing AR scheduler/composition, NextStep request/runner and Janus R1 wiring
  suites: 86 passed. No new rename-only tests were added. Touched-file Ruff and
  git diff --check pass; source/tests have no old-name references. This does not
  establish all model backends or the whole repository as fully audited.

## AR cache concatenation makes dtype preservation explicit

- Reject tensor rows with different dtypes instead of allowing torch.cat to
  promote them silently. Report the offending row and expected dtype at the
  cache assembly boundary. Mixed tensor/non-tensor rows also receive an explicit
  type error before concatenation.
- Route HF DynamicCache key/value merging through ar_concat_rows, sharing the
  plain tensor check instead of bypassing it with separate torch.cat calls.
- Keep the free split/concat operations: GLM, attention backends and the token
  scheduler share them. Keep HF conversion adapters as a framework boundary.
  Do not import generation helpers into the lower-level NN package or introduce
  a new class solely to share a dtype check. Valid homogeneous cache assembly
  and row ordering remain unchanged.
- Three regressions failed before the fix (tensor, nested mapping and HF cache).
  Cache/token scheduling/runner suites: 84 passed. Direct attention backend
  suites: 6 passed. Touched-file Ruff and git diff --check pass. These checks do
  not establish production model performance or complete the repository audit.

## Prompt scoring retains its existing generation receipts through finalization

- Pass GeneratedPromptGroup directly through strict collection and continuous
  reward finalization instead of reconstructing (unscored, prompt_indices)
  tuples. Remove the unused scalar-remap alternative; both real callers already
  receive list-valued prompt indices from generation.
- Derive generation intervals from those receipts when reporting collection
  timing, removing the parallel mutable interval list. Scoring tasks, bounded
  overlap, cancellation cleanup, reward retries and prompt remapping retain
  their existing behavior.
- Keep generation/scoring/finalization as collector methods because both
  schedules consume these stages independently. Keep local scoring task helpers
  for call-owned timing and cleanup. No extra class or compatibility wrapper is
  added. The request builder's reflected denoise-field constant remains a schema
  boundary rather than a hardcoded business vocabulary.
- Rollout suite after receipt migration: 486 passed (three dependency warnings).
  After removing the redundant timing list, collector/orchestration suites:
  311 passed. Touched-file Ruff and git diff --check pass. No new tests merely
  mirror the representation change; existing tests exercise prompt identities,
  timing, overlap and cleanup. The full repository audit remains incomplete.

## Collector preserves the group-size value for request validation

- Remove int(group_size) before request construction. GenerationRequest already
  requires a positive exact integer; collector coercion silently accepted True,
  fractional values and numeric strings before that boundary could check them.
- Keep validation on GenerationRequest and the request builder as the adapter;
  no new helper or duplicate validator is added. Valid group sizes and runtime
  scheduling stay unchanged. Timing compatibility reads remain pending a fuller
  review of the simplified test-fake contracts rather than a partial deletion.
- Three regressions failed before the fix and now verify rejection before any
  generation request is submitted. Collector and orchestration suites: 314
  passed. Touched-file Ruff and git diff --check pass. Wider audit remains open.

## Reward artifact selection stays within one builder operation

- Consolidate _reward_view, _reward_output and _tensor_value_from_ref into their
  sole public consumer, TrajectoryRolloutBatchBuilder.reward_outputs. The method
  now reads in order: require one scoring view, resolve its artifact, normalize
  its declared range. Remove the three private method hops and unused type import.
- Preserve error messages, tensor-ref precedence, uint8 reconstruction and tanh
  conversion. Keep training pack methods, shared group-ID construction and primary
  trainable-segment validation separate from this reward-only operation. Keep the
  collector-owned reward group metadata key as a genuine schema constant. No new
  class, alias or generic resolver is introduced to replace the removed helpers.
- Existing collector and trajectory suites: 122 passed. These cover ambiguous
  views, unsupported output references, replay-storage independence and uint8
  reconstruction. Touched-file Ruff and git diff --check pass; removed helpers
  have no remaining collector/test references. Whole-repository completion is
  still unproven.

## Replay sample batches own balanced plan construction

- Move _balanced_training_sample_batches into the existing _ReplaySampleBatch
  as plan_balanced. Prompt-group slicing and zero-weight dummy-slot construction
  now live together; the trainer requests a plan without owning its assembly.
  Construct via cls and reuse cls.from_prompt_group. Remove the old free function
  and update the trainer and distributed regression callers, without an alias.
- Keep scalar collective helpers free: parity and initial-replay statistics also
  consume them, so they have no single batch owner. Collective ordering, smallest
  dummy-template selection, loss weights and empty-rank rejection stay unchanged.
  No new class or distributed abstraction is introduced for line-count reduction.
- Existing distributed agreement and advantage/metrics suites: 14 passed,
  including real two-process Gloo planning checks. Touched-file Ruff and
  git diff --check pass. These do not establish NCCL/FSDP performance or full
  repository completion; the audit remains open.

## Replay metric assembly exposes its shared denominator

- Compute the replay weight sum once per _ReplayMetrics.build call instead of
  recomputing it inside each of four weighted scalar reductions. Inline the
  single-use ordinary mean at the grad_norm field; empty gradients still yield
  zero. Remove that local forwarding helper without adding a replacement class.
- Keep the weighted_mean closure, which shares length validation and reduction
  across four fields. Preserve PolicyUpdateStats and LogprobMismatchStats
  reductions as their own semantic boundaries. PromptBatchSampler's separate
  sample/preview methods and lazy torch imports remain necessary for RNG state
  ownership and torch-free config loading; they are not cleanup targets merely
  because their entry methods are short.
- Existing advantage/metrics and distributed agreement suites: 14 passed,
  including uneven replay weighting and empty-update paths. Touched-file Ruff
  and git diff --check pass. No throughput benefit is claimed for this local
  simplification. Full repository completion remains unproven.

## Artifact reports reuse their validated training-manifest fields

- Build the training ArtifactManifestReport once in from_examples and use
  dataclasses.replace to add eval provenance and overlap warnings. Remove the
  duplicate training-field constructor list from the optional eval branch.
- Keep the frozen report, recursive eval validation, output schema and warning
  behavior. Preserve SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS as an ordered
  schema contract and DEFAULT_ARTIFACT_FIELDS as reflected schema fields.
  Artifact reading/value helpers keep format-specific checks out of iteration;
  no new helper, class or flexible keyword-field bag is introduced.
- Existing artifact and video-world manifest suites: 22 passed, including train/
  eval overlap reporting. Touched-file Ruff and git diff --check pass. The wider
  repository clarity audit remains incomplete.

## Required reference-image resolution names its mutation

- Rename validate_reference_images to resolve_required_reference_images_ and
  update its recipe caller, tests and checkpoint-test comment. The operation
  fills missing references from the dataset default, expands/resolves paths and
  writes them into PromptExample; its old validation-only name hid those effects.
- Keep it as a dataset-level function rather than adding a manager class or
  attaching collection/default-path responsibilities to one example. Keep path
  existence checks and in-place behavior unchanged. Remove the old exported name
  without a forwarding alias; correct the test comment claiming validation only.
- Existing data setup and video-world reference metadata suites: 18 passed.
  Source/tests contain no old-name references. Touched-file Ruff and
  git diff --check pass. The real-checkpoint test received only a comment change
  and was not executed. Repository-wide completion remains unproven.

## Weight-sync initialization preserves policy identity

- Replace int(current) with the existing exact-integer validator before
  allocating the next version. Explicit resume versions and runtime-provided
  versions must be non-negative integers; bool, fractional and string values
  must not silently become different policy identities.
- Preserve explicit-version precedence, absent-version initialization, CPU
  snapshot ownership, push locking and version publication. Keep shared state
  selection and snapshot helpers as cross-strategy/transport boundaries; no new
  validator class or helper is introduced.
- Eight invalid-source regressions failed before the fix. Weight-sync, rollout
  protocol and Ray lease/runtime suites: 88 passed, 15 dependency/profiler
  warnings. Touched-file Ruff and git diff --check pass. No production cluster
  performance claim is made; the repository-wide audit remains incomplete.

## Weight-transfer helper ownership review: retain transport boundaries

Reviewed vrl/generation/weight_transfer.py and its sender in
vrl/generation/ray/weight_sync.py against production callers and both transfer
suites. No implementation edit is justified by this ownership review.

- Keep weight_manifest: it validates the sender's CPU tensor payload and emits
  the shape/dtype wire schema consumed by the receiver. It has no receiver state.
- Keep iter_weight_chunks: cloned slices prevent serialization from retaining an
  entire source tensor's storage. Keep iter_weight_buckets separately: packing
  several small chunks enforces the per-transport-object tensor-byte ceiling.
- Keep StagedWeightTransfer as the receiver owner for buffers, offsets and ID
  checks. Moving stateless sender iteration into it would mix lifecycle owners.
- Keep GenerationWeightSync as the protocol boundary and RayGenerationWeightSync
  as the actor-dispatch implementation. Its ACK validator is shared by local,
  remote and bucketed paths; its local broadcast closure shares one transfer ID,
  expected version and bounded dispatch policy across transfer phases.
- No new wrapper, helper class or business-name table is needed. Shape/dtype keys
  are wire schema. The buffer setting bounds wire objects, not full sender or
  receiver residency, as the module already documents.

Validation: 56 transfer/Ray-sync tests passed with one Ray environment warning,
including real local Ray shared-state, ACK, cancellation and incomplete-bucket
commit checks. This supports retaining these boundaries; it does not establish
cross-node GPU throughput or finish the wider repository clarity audit.

## Generation launch validation uses one sequence traversal

- Merge identical list/tuple recursion in GenerationRuntimeLaunchContract while
  preserving index-qualified error paths and accepted values. Correct the mapping
  error to say mapping, matching the actual collections.abc.Mapping check.
- Keep the launch contract and RayGenerationLaunchInputs as separate process/wire
  boundaries. Keep the mapping-normalization helper and recursive validator: four
  fields share normalization, and nested data must report its exact path. Pickle
  validation remains at construction. No combined runtime/config class is added.
- Existing runtime config, rollout launcher and worker checkpoint identity suites:
  63 passed, three dependency warnings. Touched-file Ruff and git diff --check
  pass. This local simplification does not complete the full repository audit.

## Overlap benchmark run metrics own loading

- Move read_run_metrics into the existing RunMetrics as from_run_dir, constructing
  through cls. Update analysis and test callers and remove the free constructor
  without an alias. File checks, warmup filtering, missing-phase behavior and
  per-run measurements remain unchanged.
- Keep summarize_arm and acceptance calculations free: they combine multiple
  runs or arms, not one RunMetrics instance. Keep ARMS, acceptance thresholds and
  the Student-t critical-value table as the isolated benchmark protocol. No new
  report/loader class is introduced just to group functions.
- Existing overlap benchmark suite: 12 passed, exercising warmup exclusion,
  skipped collection rows and acceptance analysis. Touched-file Ruff and
  git diff --check pass; source/tests have no old constructor references. No
  training benchmark was launched. The wider repository audit remains open.

## Overlap analysis distinguishes missing timings from measured zero

- Require generation, reward and overlap phase fields on rows containing
  collect.wall. The production collector emits all three; substituting zero for
  absent fields hid incomplete measurements and could distort acceptance math.
  Errors now identify rollout_stats.jsonl, the step and missing field.
- Keep rows without collection excluded, and keep optional reward queue-wait
  defaulting to zero. Explicit zero overlap remains valid for serial arms. Checks
  stay in RunMetrics.from_run_dir without another schema class/helper. Benchmark
  thresholds and statistical reduction rules remain unchanged.
- Three missing-field regressions failed before the fix; the complete overlap
  benchmark suite now has 15 passing tests. Touched-file Ruff and git diff --check
  pass. Tests use synthetic run logs; no performance acceptance campaign was run.
  Repository-wide completion remains unproven.

## Algorithm/evaluator pairs own recipe construction

- Move build_algorithm_and_evaluator into the existing AlgorithmEvaluatorPair as
  from_configs, constructing return values through cls. Update online recipe,
  factory/lifecycle tests and the real-checkpoint test caller. Remove the free
  constructor/export without a compatibility wrapper or a new class.
- Keep algorithm/evaluator pairing branches in this one composition point for
  now; no registry or declaration carrier is added merely to relocate names.
  Keep reward function/runtime construction and cross-type parking validation
  separate from pair construction because their inputs/lifecycles differ.
  This ownership change does not claim to retire all algorithm name checks.
- Existing common factory and online lifecycle suites: 52 passed, three dependency
  warnings. Touched-file Ruff and git diff --check pass; source/tests have no old
  function references. Real-checkpoint test calls were migrated but those tests
  were not run. The full repository audit remains incomplete.

## GRPO-family composition constructs the selected class once

- Separate concrete algorithm class selection from the identical constructor
  invocation in AlgorithmEvaluatorPair.from_configs. GRPO, FlashGRPO, FlowDPPO
  and GRPOGuard now share one config/advantage-estimator call instead of four
  repeated argument blocks. Constructor signatures and inheritance were checked.
- Preserve the explicit selection branches, diffusion objective vocabulary and
  evaluator/chunk-replay compatibility checks. Do not introduce a separate backend
  table, registry or forwarding helper for the same choice. This is a local
  construction simplification, not a claim that all algorithm facts are relocated.
- Existing common factory suite: 25 passed, two dependency warnings. Touched-file
  Ruff and git diff --check pass. No new tests merely mirror the refactor.
  Repository-wide completion remains unproven.

## SFT latent helpers retain distinct identity, persistence and recipe boundaries

Reviewed trainers/data/sft_latents.py, its encode_targets producer, the recipe
loader and online regularizer consumer. This pass does not justify merging them
into a single additional object.

- Keep CleanTargetRef.from_source: producer and consumer share the exactly-one
  target identity rule across PromptExample and rollout metadata representations.
- Keep save_sft_latents/load_sft_latents: they are the shared tensor-persistence
  boundary, with call-time torch imports. SFT_LATENTS_SCHEMA_VERSION is an actual
  file-schema version and should remain explicit.
- Keep _load_sft_latents_from_config as the recipe adapter: disabled SFT does not
  read a configured shard, and enabled SFT supplies family/model provenance.
  Moving this logic into CleanTargetRef would give one target dataset/config
  responsibilities unrelated to its identity.
- Do not treat the documented [C,T,H,W] shape as a validated load guarantee:
  load_sft_latents currently validates schema, provenance and target keys but not
  tensor rank. Shape requirements require a separate producer/consumer analysis
  before tightening this persistence boundary; this remains an audit limitation.

Existing shard, recipe-loader and regularizer suites: 32 passed. No production
model encode was run and no implementation change was made in this review.
The broader repository audit remains incomplete.

## SFT geometry is checked where rollout shape is known, before staging

- Resolve the prior SFT shape audit question: encode_targets removes the source
  batch dimension, while the trainer compares the expanded target shape against
  rollout observations with the transition dimension removed. The I/O contract
  should describe unbatched model latents, not impose a universal rank of four.
  Update shard documentation accordingly, retaining video [C,T,H,W] as an example.
- Read the target once into clean_latents, calculate the required expanded shape
  and reject a mismatch before expansion/device conversion. Use the observation
  tensor's dtype directly. The trainer owns this cross-input geometry check;
  persistence helpers cannot know the run's rollout geometry.
- Keep the existing save/load functions, schema version and CleanTargetRef owner.
  No shape-policy class or generic conversion helper is added. Model noising,
  replay execution and valid-target loss behavior remain unchanged.
- The strengthened mismatch regression failed before the fix by reaching
  Tensor.to. SFT regularizer, shard and recipe-loader suites: 32 passed. Touched-
  file Ruff and git diff --check pass. No real model encoding or GPU performance
  campaign was run. Wider repository completion remains unproven.

## Rollout group splitting uses one ordered deduplication operation

- Replace the parallel seen set and ordered_ids append loop with dict.fromkeys
  over normalized group IDs. Preserve first-occurrence order, int conversion,
  per-group selection and the zero/one-group identity return.
- Keep rollout device movement distinct from move_value_to_device: the former
  deliberately matches torch.Tensor, while the trajectory helper accepts
  callable .to leaves. Direct substitution would change metadata behavior.
  Keep shared tensor-tree traversal and trajectory rebuild helpers; they centralize
  container and sample-axis behavior rather than merely forwarding one call.
- Existing collector, orchestration and trajectory operation suites: 333 passed.
  Touched-file Ruff and git diff --check pass. No speedup is claimed for the
  deduplication cleanup. The whole-repository audit remains incomplete.

## Reward service identity validation belongs to its existing protocol type

- RewardServiceInfo now rejects non-string model_name and model_version at
  construction. Previously truthy numbers, booleans and containers could pass
  info_from_wire and reach client identity comparisons as supposedly typed data.
  Model names remain non-empty; an empty string remains a valid model version.
- Keep wire encode/decode functions as a shared client/server protocol boundary.
  Their matching shapes make the envelope contract easy to locate. Keep the
  common envelope/object/unknown-field checks, WIRE_VERSION and derived artifact
  schema fields. Moving HTTP envelopes into inference dataclasses or adding a
  state-free codec class would add coupling without simplifying this boundary.
- Ten malformed identity cases cover both fields; nine failed before the fix.
  All 74 reward service tests pass, as do touched-file Ruff and diff checks.
  No GPU throughput claim is made. The broader repository audit remains open.

## Name shared batch-context gathering for its returned value

- Rename require_matching_batch_context to gather_batch_context across diffusion,
  chunk denoise, token AR, Janus and NextStep gatherers and the existing tests.
  The operation validates nested values and returns the shared context mapping;
  its name now expresses that result, consistent with gather_replay_tensors.
  Remove the old export without introducing a forwarding alias.
- Keep the shared free functions: recursive tensor-safe comparison and replay
  gathering remove real cross-family complexity. Keep family gatherer files as
  driver/model boundaries and preserve their uniform gather_batches interface.
  Class proliferation and line-count reduction are not goals of this change.
- Generation suite: 781 passed with 17 dependency warnings. NextStep runner and
  Janus rollout wiring: 25 passed. Touched-file Ruff and diff checks pass; no old
  code/test references remain. This is a naming change, with no scheduling,
  memory-admission or merge-semantics change. Repository-wide audit remains open.

## Validate prefetch input before starting the continuous pipeline

- Move the duplicated empty-next-prompts check to next_iteration's input
  validation, before pipeline startup, weight synchronization or batch consumption.
  Previously both split and combined collection could push initial weights before
  rejecting an empty prefetch; combined collection could also finish the current
  iteration first. Remove the redundant prefetch metric assignment: its value
  already derives from whether next_prompts was supplied.
- Keep valid-request prefetch ordering, prompt identity matching, capacity owners
  and terminal cleanup. Keep the schedule facade for trainer-thread weight export
  and the cross-type topology guard; neither is an ownerless helper. This change
  does not consolidate concurrency boundaries merely to reduce file count.
- Both regression arms failed before the fix because initial weights were pushed.
  All 216 continuous orchestration tests pass. Touched-file Ruff and diff checks
  pass. No production throughput measurement; broader audit remains incomplete.

## Reuse exact-integer validation at token scheduling boundaries

- TokenLoopInit row_count/step_count, TokenStepBatch position and the token loop's
  optional scheduler_batch_size now call the existing require_exact_int helper.
  Remove four handwritten bool/type/lower-bound checks while retaining the same
  accepted values and ValueError behavior. Diagnostics now include the value and
  distinguish an invalid type from an invalid lower bound.
- Keep row-list emptiness, uniqueness and row-routing validation with their
  existing owners. Keep cache gather/scatter and the token-step protocol types:
  these represent actual family-neutral state-routing boundaries. No new helper,
  wrapper class or family import is introduced; config utilities remain a leaf.
- Existing token loop, scheduler batching and NextStep runner tests: 33 passed.
  Touched-file Ruff and diff checks pass. No performance claim; broader audit
  remains incomplete.

## Let TeaCacheConfig own enabled option construction

- Replace the hand-selected threshold/warmup_steps mapping with direct dataclass
  construction after removing the parser-only enabled flag. Enabled mappings no
  longer silently discard misspelled keys, and added dataclass fields do not need
  a second forwarding list. Preserve disabled-input short circuiting and defaults.
- The drift probe now constructs TeaCacheConfig(threshold=threshold) directly;
  its already-known numeric input does not need a bool/mapping parser round trip.
- Keep relative_l1_change as the shared runtime/offline metric boundary. Keep the
  existing state machine and denoise-loop function; their roles remain separate
  from config parsing. No new schema table, wrapper or numeric algorithm change.
- Unknown-option regression failed before the fix. Denoise-step tests: 80 passed;
  config drift guards and diffusion layout: 117 passed. Touched-file Ruff and diff
  checks pass. No real-model drift-probe run or speedup claim. Broader audit open.

## Execution progress and probe counts use the shared integer boundary

- BatchProduceFence.completed_batches, BatchSizeProbeTrial.n and the probe
  result's sample count/budget now use require_exact_int. Their former lower-bound
  comparisons accepted True, fractional counts and NaN; the existing owner types
  now reject those values before progress tracking or planning can consume them.
- Keep the fence's query method as the nonblocking device-completion boundary,
  and keep probe records separate from worker/runtime implementations because
  they cross that interface. Preserve probe fitting, confirmation and event-query
  behavior. No new validation wrapper, taxonomy or class is introduced.
- Twelve invalid-count regressions failed before the fix. Memory probe, pipelined
  execution/progress and runtime-config suites: 122 passed with three dependency
  warnings. Touched-file Ruff and diff checks pass. Production probe performance
  was not measured; broader repository audit remains incomplete.

## Keep latent decode batching visible at its execution point

- Remove ChunkedLatentDecoder._chunks, a sole-use manual slice-list builder.
  The decode entrypoint now uses Tensor.split for bounded row batches and keeps
  the original tensor when batching is disabled or the limit covers all rows.
- Keep LatentDecodePlan and the shared decoder: Flux, Wan and other families
  supply distinct preparation/postprocess callbacks through this common boundary.
  Keep _decode_chunk for that ordered transformation pipeline and the layout
  normalization. Cross-family consistency matters more than removing these types.
- Extend the existing decode test across uneven batches, exact/oversized limits,
  and disabled None/zero/negative limits. Decode, layout parity and VAE memory
  suites: 29 passed with two dependency warnings. Touched-file Ruff and diff
  checks pass. Tests use synthetic decode models; no throughput claim. Audit open.

## CFG packing enforces the equal branch shapes required by splitting

- Extend the existing pair validator to include the batch dimension. Packing
  previously allowed one conditional row and three unconditional rows, while
  split_batched_cfg_output unconditionally divides an even output into halves.
  Reject unequal branch row counts for primary and extra tensor kwargs at packing.
- Keep shared pack/split/combine functions: they centralize the uncond-first
  ordering and guidance math across family runners. Keep the pair validator
  because both primary and extra tensor packing use it. No new wrapper or
  guidance-formula change; separate-CFG execution remains unchanged.
- All four unequal-row regression cases failed before the fix. Shared denoise
  model tests: 102 passed with 16 dependency warnings. Touched-file Ruff and diff
  checks pass. This validates synthetic/model-component contracts, not a full
  production training run. Broader repository audit remains incomplete.

## Forward prepared backbone branches through one argument-conversion boundary

- DiffusionBackboneCaller now passes DiffusionBranch directly to _forward_branch.
  Move the repeated as_transformer_kwargs conversion from four CFG/no-CFG call
  sites to this existing execution boundary. Remove the old _call_transformer
  name without a forwarding alias; transformer invocation order is unchanged.
- Keep runner branch construction, postprocessing and finalization hooks, which
  preserve cross-family execution shape. Keep current output-format adaptation;
  this review does not establish that all model callers use a single return type.
  No new facade or numeric/output compatibility change is introduced.
- Existing CFG and backbone contract tests: 11 passed, covering batched, separate
  and disabled CFG invocation counts and results. Touched-file Ruff and diff
  checks pass; no old local references remain. Broader audit remains incomplete.

## Preference image splitting uses its declared RGB-pair layout

- PreferenceBatch.split_winner_loser now checks [B, 6, H, W] and explicitly
  selects the winner's three channels and loser's three channels. The previous
  shape[1] // 2 inferred a layout even for four/eight channels or invalid ranks.
  Keep this check with the existing layout operation, before trainer VAE staging.
- Keep collate as the DataLoader adapter and stacked_winner_then_loser as the
  shared DPO image-order contract. Keep trainer checks for encoder output counts,
  which involve different inputs. No new class, helper or channel-count table.
- Four malformed-layout regressions failed before the fix; valid block ordering
  is covered explicitly. Preference data and offline DPO tests: 31 passed with
  two dependency warnings. Touched-file Ruff and diff checks pass. No real dataset
  download or production training run; broader repository audit remains open.

## Prompt sampler and manifest adapters: retain their actual boundaries

- Reviewed PromptBatchSampler sample/preview/_sample_with. Keep the common draw
  implementation: preview clones RNG state while sample advances the authoritative
  generator, and every rank slices an identical global draw. Existing tests verify
  preview purity, distributed slices and restoration from checkpointed RNG state.
- Keep path loading and JSONL bytes parsing separate: both file loading and the
  JSONL Dataset use the same parser, with UTF-8/row diagnostics and metadata rules.
  The bytes API can accept an authenticated snapshot, but current production call
  sites inspected here read files; this review does not prove authenticated-snapshot
  use throughout training. Keep image-manifest loading as a public API facade.
- Keep the small Dataset __len__/__getitem__ implementations. A new inheritance
  layer solely to remove those repeated framework methods would obscure ownership.
  No implementation change was justified in these two reviewed files.
- Prompt parser and sampler suites: 61 passed. Further inspection found duplicated
  image-loader config projection in DatasetProvenanceSpec.load_manifest and
  load_prompt_examples_from_config; review their distinct single-manifest versus
  mixture semantics before consolidating. The broader audit remains incomplete.

## Image-caption Dataset owns config-based construction

- Add ImageCaptionPromptDataset.from_config(data, path=...) and route training
  and provenance single-manifest loading through it. The existing Dataset now
  owns the formerly duplicated image/caption/default-task option projection.
  Require an explicit path so provenance can load evaluation data without
  accidentally selecting the training manifest from config.
- Preserve mixture sampling in the training entrypoint and separate train/eval
  loading in provenance. Keep load_prompt_image_manifest as the explicit-options
  public facade for callers without DataConfig. No new loader class or registry.
- Existing prompt parsing, data setup, dataset provenance and VideoPhy I2V suites:
  63 passed. Touched-file Ruff and diff checks pass. Projection defaults and list
  return semantics remain unchanged. Broader repository audit remains incomplete.

## EMA swap ownership rejects overwriting an unrestored snapshot

- copy_ema_to now rejects a second swap while temp_stored_parameters exists.
  Previously the second call overwrote the original training weights with a copy
  of already-installed EMA weights, so copy_temp_to could no longer restore them.
  Reuse the existing snapshot as the authoritative state; add no duplicate flag.
- Keep the swap/restore methods and checkpoint rank agreement. They own actual
  reversible weight mutation and distributed failure coordination. Preserve normal
  swapping and rollback after a failed copy; no optimizer or EMA formula change.
- Regression failed before the fix and now verifies snapshot identity plus exact
  original-value restoration after a rejected second swap. EMA/checkpoint suites:
  124 passed with 14 dependency warnings. Touched-file Ruff and diff checks pass.
  No new distributed training campaign; broader repository audit remains open.

## Optimizer construction depends only on optimizer configuration

- build_optimizer accepts OptimConfig instead of TrainerConfig; both online
  trainer call sites now pass config.optim. Read the declared optim_8bit field
  directly and share AdamW hyperparameter construction across both backends.
  Compute fused eligibility only for standard torch AdamW.
- Keep the free factory as backend-selection/lazy-import boundary and retain
  FP32MasterWeightOptimizer's source/master ownership. No optimizer class, bucket
  mechanism or wrapper compatibility path is added; parameter identity/order and
  backend-specific fused handling are preserved.
- Optimizer and online state restoration tests: 41 passed, two skipped. This
  environment has no bitsandbytes, so actual 8-bit validation was unavailable.
  A standard AdamW constructor check and a clearly isolated fake 8-bit constructor
  check verified non-default hyperparameter forwarding. Touched-file Ruff and
  diff checks pass. Broader repository audit remains incomplete.

## Parameter diagnostics use one result schema for empty inputs

- Remove parameter_state_summary's shortened early-return payload. Missing a
  callable named_parameters now selects an empty iterator and uses the normal
  eight-field result construction, including all four trainable statistics.
  This preserves existing empty-count behavior without requiring readers to
  infer whether omitted trainable fields mean zero or a different schema.
- Keep the diagnostic free functions as shared observation/serialization APIs.
  Keep _requires_fp32_master_weights: both optimizer initialization and checkpoint
  restore use the same predicate, so inlining it would duplicate a correctness
  decision. No new diagnostics class or cached parameter collection is introduced.
- Empty-input regression failed before the fix. Diagnostic utility and online
  diagnostics suites: 21 passed, including real local Gloo/DTensor digest checks.
  Touched-file Ruff and diff checks pass. Whole-repository completion unproven.

## Express rollout phase success through control flow, not duplicate state

- Replace rollout_memory_released with try/except/else: restore trainer state
  only in the successful offload branch and only when it was parked. Preserve
  root/cleanup exception precedence and the rule that failed offload leaves
  trainer state parked for terminal cleanup.
- Remove the strict schedule's None initialization and post-collection assertion;
  its phase manager propagates errors, so successful exit guarantees batch
  assignment. No alternate missing-batch state exists on that path.
- Keep coordinator phase methods, cleanup-error wrapper and cross-layer cause
  inspection. They own GPU handoff sequencing and restart-policy semantics, not
  redundant forwarding. Do not replace them with another state-machine class.
- Strict failure, orchestration and frozen-driver-offload tests: 34 passed. Tests
  cover collection failure, offload failure, no-parking and restoration failure.
  Touched-file Ruff and diff checks pass. Broader audit remains incomplete.

## Reward launch parsing preserves explicit parking types

- RewardRuntimeLaunchContract.from_component_config now requires a boolean
  sleep_offload and uses require_exact_int for the nonnegative residual-byte
  limit. Previously bool("false") enabled parking and int(1.5) truncated its
  budget. Keep absent-field defaults and the open model-factory parameter bag.
- Keep build_reward_scorer as transport-selection/lazy-import boundary. Keep
  _build_prepared_model_in_pool: its separate frame allows failed lazy model
  preparation references to be cleared before pool cleanup. Removing that helper
  would change failure ownership, not merely reduce navigation.
- Six invalid-config regressions failed before the fix. Runtime factory,
  in-process scorer and service tests: 95 passed, one skipped. Touched-file Ruff
  and diff checks pass. No new GPU parking benchmark; broader audit incomplete.

## Reward runtime reads launch configuration from its existing owner

- Remove the runtime's duplicate component_config, sleep_offload and residual
  limit attributes. Read the parsed RewardRuntimeLaunchContract directly at model
  construction, parking and shutdown. Update configuration-inspection tests to
  inspect that owner; no forwarding properties or aliases are added.
- Remove score-map float/key conversion before RewardInferenceResult construction:
  its existing post-init already normalizes scores and checks finiteness. Retain
  the explicit plugin Mapping check and shared per-artifact result construction.
- Keep model/pool/preload baseline as actual runtime state, plus the isolated
  failed-build frame and transport factory. No scoring or parking policy change.
- Inference, multi-reward and service suites: 135 passed, one skipped. Touched-file
  Ruff and diff checks pass. No production GPU performance claim; audit remains open.

## Reward result schema rejects malformed mappings before conversion

- RewardInferenceResult validates scores and timing_ms as mappings before using
  .items(). Null/list/scalar wire fields previously leaked AttributeError past
  the wire parser's typed protocol-error conversion. Both direct and HTTP inputs
  now receive field-specific errors through the same existing schema owner.
- Remove the second float conversion from the timing validation loop; values
  were already normalized by the preceding comprehension. Keep wire error
  wrapping and request result-order validation as actual transport/identity
  boundaries. No new parser helper or result wrapper.
- Twelve direct/wire malformed-field regressions failed before the fix.
  Inference and service suites: 126 passed, one skipped. Touched-file Ruff and
  diff checks pass. Broader repository audit remains incomplete.

## Artifact geometry is checked before device-to-host staging

- DiskRewardArtifactStore._write_one checks output type, nonemptiness and media
  rank before detach().cpu(). These are tensor metadata checks; an invalid shape
  need not trigger device transfer before reporting its input error.
- Keep materialization, hashing and owned-path cleanup unchanged. Keep release
  and retain as distinct terminal ownership operations, and the provenance filter
  as the scalar wire boundary. No new staging helper or artifact type is added.
- Strengthened bad-shape test failed before the change by reaching Tensor.cpu.
  Artifact store and disk reward-function suites: 13 passed. Touched-file Ruff
  and diff checks pass. CPU regression proves call ordering, not measured GPU
  transfer savings. Broader repository audit remains incomplete.

## Generation batch identity and retry boundaries: retain shared ownership

- Re-read GenerationSampleBatch, ordered_covering_batches, the local OOM retry
  driver and BatchExecutorBase. Keep the batch owner for range/key/splitting,
  the common gather coverage checks and the executor facade used across bindings.
  Moving these into individual family classes would duplicate their contracts.
- Keep retry's exception-frame cleanup: its regression uses weak references to
  prove failed forward locals are released before cache cleanup. Terminal failure
  tests separately prove diagnostic traceback locals remain available when no
  split retry will happen. This helper removes real failure-handling complexity.
- Keep EnginePlan as the request-level width-resolution API. Its small size is
  not evidence that direct and Ray planning should implement defaults separately.
  No implementation changes were justified in this review slice.
- Sample-batch and gatherer suites: 70 passed, including ordering, per-batch
  replay alignment, dtype preservation and retry/terminal failure behavior.
  These are component checks, not a complete distributed generation campaign.
  Whole-repository completion remains unproven.

## Engine aggregate uses asyncio's existing awaitable adapter

- Remove EngineCallRef._gather's local wait_one coroutine, which only awaited its
  argument. Pass each rank ref to ensure_future directly. Inspected the installed
  Python implementation: ensure_future already wraps generic __await__ objects.
- Keep EngineCallRef, uniform_rank_result and rank_handles as aggregate,
  cross-rank agreement and lifecycle-view boundaries. Preserve sibling task/ref
  cancellation and rank-order result selection; no replacement adapter is added.
- Engine and weight-sync suites: 55 passed with one Ray dependency warning,
  including custom awaitable refs and a real Ray shared-object weight-sync check.
  Touched-file Ruff and diff checks pass. No cross-node performance claim;
  broader repository audit remains incomplete.

## Dispatcher fleet identity is explicit before admission state exists

- Reject bare string/bytes worker_ids before tuple conversion, and reject empty
  or non-string members before building slot and waiter maps. Previously a bare
  string became a fleet of characters; malformed members could enter state or
  fail incidentally during hashing. Keep existing nonempty/unique fleet checks.
- Keep worker availability, active refs and per-worker admission waiters separate:
  they encode different scheduling facts. Keep run/run_one and cancellation
  boundaries; no alternate dispatcher or validation class is introduced.
- Five malformed-identity cases plus existing deterministic batch dispatch tests:
  28 passed. Touched-file Ruff and diff checks pass. These tests use controlled
  awaitable refs, not a new real-Ray fleet run. Broader audit remains incomplete.

## Dispatcher result tasks no longer return duplicate reference identity

- Remove the local await_ref wrapper returning (ref, result). The waiters mapping
  already owns task-to-ref identity; both normal harvesting and cancellation
  harvesting now read task.result() directly after looking up the ref once.
  ensure_future handles generic awaitables, including EngineCallRef and Ray refs.
- Keep spawn/finish_success as reused registration and telemetry operations.
  Keep run_one's wait_for_result closure: it defers custom waiter invocation
  inside a task, so removing it would change synchronous exception handling.
  Preserve deadline, admission and cancellation state machines.
- Batch dispatch and real actor-pool tests: 37 passed with one Ray warning;
  deadline/cancellation tests: seven passed. Touched-file Ruff and diff checks
  pass. No throughput claim; broader repository audit remains incomplete.

## Actor admission waits check state once before yielding

- Remove the duplicate open/candidate/availability checks after Event.clear in
  RayActorDispatcher._wait_for_available. The dispatcher mutates its admission
  state on one event loop; no await separates the initial checks, clearing the
  event and entering Event.wait. The repeated check cannot observe new state.
- Keep the loop's checks after every wakeup and its cancellation cleanup. Keep
  _can_acquire separate from _try_acquire: readiness does not reserve a slot.
  Queue helpers preserve per-worker FIFO identity and shared run/run_one
  behavior; reducing their count is not a goal. No new classes or constants.
- Batch dispatch and real actor-pool suites: 37 passed, one Ray dependency
  warning, including shared admission and middle-waiter cancellation. Ruff and
  diff checks pass. This simplifies control flow without a throughput claim;
  the broader repository audit remains incomplete.

## Health monitoring uses the existing runtime fleet view directly

- Remove RolloutWorkerHealthMonitor._owned_ranks and its type-only-purpose
  actor-handle import. Both consumers now read runtime._owned_ranks directly.
  Runtime already handles an absent session, and session.rank_handles builds a
  fresh flattened list; the monitor wrapper added only another list copy.
- Keep the runtime adapter, session view and shared rank_handles lifecycle
  helper: they hide session/engine structure from consumers. Keep probe and
  terminalization boundaries, pause/resume epoch checks and shutdown join grace.
  The grace constant represents a bounded shutdown wait, not domain vocabulary.
  No new classes, public API, or reduced cross-family consistency is intended.
- Health-monitor suite: 12 passed with one Ray dependency warning, including
  a real blocked actor whose timeout kills the fleet and unblocks an outstanding
  driver call. Touched-file Ruff and diff checks pass. Broader audit incomplete.

## Pipelined execution declares its existing generation contracts

- Replace Any on forward_batches_pipelined's executor, request and batch
  sequence with existing GenerationBatchExecutor, GenerationRequest and
  GenerationSampleBatch types. Name family-owned results with BatchPayload and
  pending copy fences with torch.cuda.Event. Type-only imports preserve lazy
  loading. The production caller supplies EnginePlan.sample_batches, a tuple;
  iterate it directly instead of copying the plan into another list.
- Keep the pipeline function and CUDA tree-copy helper as execution and tensor
  lifetime boundaries. Preserve record_stream, ordered results, produce fences
  and exception-path copy synchronization; no new protocol or wrapper class.
  Family payloads intentionally remain opaque through the existing alias.
- Pipeline CPU/control-flow and real-CUDA suites: 12 passed, including four
  CUDA tests for bit-exact results, ordering, CPU payloads and completion fences.
  Touched-file Ruff and diff checks pass. No throughput measurement claimed;
  broader repository audit remains incomplete.

## Sample-batch methods no longer use ambiguous chunk names

- Rename full-sequence _forward_chunk to _forward_batch, the reference-image
  hook to _reference_image_for_batch, and the worker profiling wrapper to
  _profile_forward_batch. Update Cosmos's shared conditioning call and binding
  tests. All consume GenerationSampleBatch; they do not represent temporal
  autoregressive chunks or weight-transfer chunks. No compatibility aliases.
- Keep forward_batch/forward_probe_batch as distinct plugin entry points sharing
  the same implementation, and keep reference loading and profiler scope methods
  as reused conditioning and instrumentation boundaries. Do not rename actual
  weight chunks or chunk-autoregressive bindings. Keep the engine.forward_chunk
  profiler label used by historical performance records; trace format changes
  are outside this naming cleanup. No constants or classes added.
- Full-sequence binding and worker debug/versioned-slot suites: 126 passed.
  Touched-file Ruff and diff checks pass; no old private method references remain
  in vrl or tests. Broader repository audit remains incomplete.

## Storage policy parsing uses its dataclass schema directly

- Replace manual device/dtype extraction and str conversion in
  TrajectoryStoragePolicy.from_config with cls(**value) after config-container
  normalization. Dataclass defaults own omitted fields and its constructor
  rejects unknown fields rather than silently dropping a misspelled setting.
- Keep from_config as the configuration adapter, Literal-derived valid-value
  constants as schema boundaries, tensor-tree conversion and byte estimation as
  distinct operations. Keep the report's complete-field check: it requires an
  exact persisted shape, which is stricter than allowing config defaults.
  No generic parser or new owner class is introduced.
- Two unknown-field cases failed before the fix. Trajectory and binding storage
  tests: 88 passed; SANA curve-verdict tests: 13 passed. Touched-file Ruff and
  diff checks pass. Broader repository audit remains incomplete.

## TeaCache reuse requires a below-threshold measurement

- Express TeaCacheState's decision as not(change < threshold) for a fresh
  forward. Previously change >= threshold classified NaN as permission to skip,
  and a NaN accumulator could keep skipping until a forced step. A forward now
  resets that accumulator; two subsequent finite signals can resume normal reuse.
- Keep relative_l1_change as the shared runtime/offline metric, including its
  raw nonfinite measurement, and keep decision/counter ownership in TeaCacheState.
  No additional validator, class or vocabulary table is introduced. This does
  not repair nonfinite model state; it prevents the cache from hiding it.
- NaN and infinity signal regressions both failed before the change. Denoise
  suite: 82 passed. Touched-file Ruff and diff checks pass. No training-quality
  or throughput claim; broader repository audit remains incomplete.

## Rollout stats expose a metric mapping, not only phase times

- Rename RolloutStats.as_phase_dict to as_metrics_dict and update all Python
  callers/tests. The output includes summed phases, counters, peak gauges and
  reward latency percentiles; the old name incorrectly implied every value was
  a duration. Sink locals now call that full mapping metrics too.
- Keep separate accumulation methods and sink classes because reduction and
  output ownership differ. Preserve all output keys, units, phase percentage
  calculations and historical log/JSON formats. No alias or new abstraction;
  downstream metric-schema field renaming is outside this interface cleanup.
- Stats, collector, continuous orchestration and trainer reward-update tests:
  281 passed. Touched-file Ruff and diff checks pass; no old method references
  remain in vrl/tests Python sources. Broader repository audit incomplete.

## Bulk metric recording delegates validation without coercing names

- Require nonempty string names at add_phase/add_counter/observe_gauge, and
  remove str/float conversion from add_phases/observe_gauges. Previously bulk
  recording turned None and numeric keys into plausible strings while singular
  calls treated them differently; numeric values were also converted twice.
- Keep bulk methods as shared merge entry points and _sum_optional as the shared
  absent-versus-zero reduction. Preserve dynamic metric names and the distinct
  sum/peak semantics; no metric-name vocabulary table or validation class.
- Twenty boundary cases cover singular and bulk entry points; nine failed before
  the fix. Stats, collector, continuous and trainer reward-update suites: 301
  passed. Touched-file Ruff and diff checks pass. Broader audit incomplete.

## Collector reads the timing fields its receipt actually owns

- Read UnscoredRollout.phases and reward_timing_ms directly when finishing scored
  groups. Both dictionaries are declared with empty defaults on the production
  dataclass; missing attributes should expose a contract violation rather than
  silently erase measurements.
- Keep simplified scheduling payload compatibility in PromptCollectionFake's
  finish adapter. It supplies timing-only views for those fakes before invoking
  production accounting/remapping, without mutating their pending payloads.
  This test boundary adds lines deliberately; production should not guess fields
  solely because a test omits full request/output construction.
- Keep generation receipts, scoring and identity-remapping boundaries. No new
  production class, constant or alternate statistics calculation is introduced.
- Collector, orchestration and trainer reward-update suites: 329 passed.
  Touched-file Ruff and diff checks pass. Broader repository audit incomplete.

## Reward timing consumers use the normalized output contract

- Remove repeated str/float conversions when MultiReward sums RewardOutput
  timings and when the collector forwards extra timing fields into RolloutStats.
  RewardOutput.__post_init__ already normalizes names and values and rejects
  nonfinite/negative timings. Remove the intermediate timing dict copy before
  update; the destination still owns an independent mapping.
- Keep RewardOutput normalization, standard timing-key selection and first-group
  accounting. These are type/schema and call-level ownership boundaries. Empty
  unscored input still exits before indexing; no new helper or class is needed.
- MultiReward, collector and prompt collection suites: 80 passed, including
  child timing aggregation and per-call timing ownership. Touched-file Ruff and
  diff checks pass. Broader repository audit remains incomplete.

## MultiReward component names identify one contribution each

- Validate nonempty string and unique component names in MultiReward.__init__.
  The config dictionary naturally has unique keys, but direct construction
  previously allowed two contributions with the same name: totals included both
  while the component dictionary retained only the last. Reject that ambiguity
  before scoring instead of attempting to repair names downstream.
- Keep the named registry and lazy builtin loader as configuration lookup and
  import boundaries. Keep component tuples, scoring order, weighted reduction
  and retryable child teardown; no wrapper class or new vocabulary table.
- Four constructor regressions failed before the fix. MultiReward and collector
  tests: 62 passed. Touched-file Ruff and diff checks pass. Broader repository
  audit remains incomplete.

## Core gather and rank-lifecycle boundaries retained after review

- Re-read sample_batches.py coverage, concatenation, replay and context helpers,
  their full-sequence/chunk-autoregressive/token/Janus/NextStep consumers, and
  rank_group.py with worker load/release ownership. No implementation change is
  justified by function count: these operations are shared across families or
  bridge the worker lifecycle to lazily imported torch.distributed.
- Keep SampleAlignedValues to distinguish ragged sample rows from static Python
  sequences. Keep dtype agreement before concatenation, per-batch row checks,
  exact prompt-major coverage and nested context equality; one total row count
  cannot replace these distinct invariants. Keep RankGroupSpec as the serialized
  rendezvous contract, with init/destroy outside the data object.
- The nccl/gloo choices are the supported process-group protocol boundary, not
  an algorithm vocabulary. No new classes, flattened family implementations or
  moved ownership are proposed; cross-family consistency outweighs LOC reduction.
- Rank-group, gatherer and sample-batch suites: 94 passed, including a two-process
  Gloo rendezvous/all-gather/teardown check and strict replay/dtype tests. These
  are component evidence, not proof of full training or repository completion.

## Required chunk-trajectory fields are validated before concatenation

- Remove the second None scan from ChunkAutoregressiveDenoiseGatherer._cat_field.
  Its only callers follow _ordered_batches, which validates every trainable
  result's required shape prefixes. Missing values already fail there before
  any concatenation. Keep validation on each batch, not just the final total.
- Keep _cat_field as the repeated named-field projection and shared strict-dtype
  concatenation adapter. Keep _cat_optional_field's all-or-none presence rule:
  per-result validation deliberately permits absent KL, so cross-result presence
  still needs checking. Keep family-specific consistency checks and the neutral
  shared coverage helper; no new classes/constants or public API changes.
- Chunk binding and shared gatherer suites: 69 passed, including trajectory-axis,
  replay-axis and dtype rejection cases. Touched-file Ruff and diff checks pass.
  Broader repository audit remains incomplete.

## AR runner construction is explicit in the shared hook name

- Rename _ar_runner to _build_ar_runner across the shared token executor,
  Janus/NextStep consumers, test overrides and sampling-schema reader comment.
  The hook constructs a new runner with a selected attention backend; it does
  not retrieve a cached runner. No compatibility alias or lifetime change.
- Keep shared embedding and tokenizer-padding adapters for cross-family
  consistency. Keep native-runner backend rejection and family declarations;
  backend configuration validation is not changed by this naming slice.
- Token binding and NextStep/GLM/LlamaGen model-construction/parsing suites:
  84 passed with 16 dependency deprecation warnings. Touched-file Ruff and
  diff checks pass. Broader repository audit remains incomplete.
