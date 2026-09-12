# Rollout batch and collector configuration review

Read all three `vrl/rollouts/batch` modules and `rollouts/collector/config.py`.
Followed collector group assembly, continuous producer CPU storage, trainer
microbatch/zero-advantage selection and deferred replay transfer. Read the
denoise-options projection to establish where nested sampling blocks go.
The larger collector/producer/trainer modules remain pending full review.

## Retained responsibilities

- `core.py`: keep the dependency-light `RolloutBatch` data contract. The payload
  estimate delegates to the shared traversal, accounting for shared references
  in one call. It is an estimate, not a GPU allocator reading.
- `__init__.py`: keep the small public facade; importing a batch type must not
  eagerly import torch-dependent operations.
- `select_batch`: shared by trainer sample planning, zero-advantage filtering and
  group splitting. Selection keeps rewards, group IDs, extras and trajectory
  samples aligned. The inner selector captures the selected rows and batch size;
  it is not an independently stateful object needing another class.
- `nonzero_advantage_mask`: pure numerical operation shared across tensor ranks;
  it consumes advantages, not a batch instance. Preserve its sum-of-absolute-values
  reduction rather than changing the numerical filtering policy during cleanup.
- `split_batch_by_group`: preserve first-seen group order and the no-copy
  single-group path. Collector remaps local prompt groups before calling it.
- `remap_group_ids_`: preserve the clone-before-assignment behavior, so remapping
  one local group to another group's original numeric ID cannot cascade.
- `move_training_batch_to_device`: preserve its explicit deferred-replay option.
  Rewards/group IDs move while stored replay payloads can remain on CPU until
  the evaluator selects a denoise step. The sentinel regression confirms no
  full replay tensor move occurs on that path.
- `_move_tensor_tree`: keep the torch-only leaf adapter over the shared walker.
  Replacing it with duck-typed `move_value_to_device` would also invoke unrelated
  objects' `to()` methods. Repeating the walker arguments at both call sites or
  adding a class would obscure that shared contract.
- `RolloutCollectorConfig.from_root`: keep projection on the existing owner.
  `_DENOISE_OPTION_FIELDS` derives from the denoise dataclass. Nested TeaCache
  mappings are handled by `DenoiseRequestOptions.from_sections`, not silently
  lost merely because the scalar projection excludes dictionaries. Collector
  policy and request fields are intentionally separate.

There is no business vocabulary ALL_CAPS table in batch operations. Export lists
are API declarations. No production change is warranted merely to reduce free
function counts; the batch operations are shared across the actual pipeline.

## Follow-up constraints

Legacy `extras` selection uses leading-dimension equality to identify per-sample
tensors; trajectory selection uses explicitly declared axes. Do not generalize
that heuristic to arbitrary metadata or replace the declared-axis implementation
with it. A future removal needs an explicit extras producer/consumer contract.

The collector config's `train_segments` lookup applies only to algorithms that
declare that optional feature. This is not checkpoint-style fallback inference.
The root/algorithm objects are mutable, so duplicate coefficient validation must
be judged with their direct construction paths, not just type annotations.

## Validation

76 tests passed: collector runtime, deferred denoise replay slicing, online
reward/update flow and trajectory selection operations, with CUDA disabled.
Checks include group-local training and one optimizer step across streaming
accumulation. No new tests or production edits were needed for retained behavior.

Previous review commit: `0ca846310` (model/sampling boundaries).
