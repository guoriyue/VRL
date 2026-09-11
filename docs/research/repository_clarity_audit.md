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
