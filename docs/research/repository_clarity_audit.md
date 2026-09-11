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
  candidates for a separate ownership review.
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
