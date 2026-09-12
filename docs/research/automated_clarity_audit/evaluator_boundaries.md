# Evaluator interface and trajectory signal ownership

Reviewed complete evaluators/base.py, trajectory.py and __init__.py, all five
concrete evaluator call sites, replay-model guard implementation and signal-batch
validation. Previous audit commit: 4f7b88443.

## Change

Correct trajectory.py's module description: the builder selects recorded facts
and moves them to the replay device, while TrajectorySignalBatch owns shape
validation. The implementation already centralizes those checks; the old prose
incorrectly attributed them to the builder. No execution behavior changes.

## Retain and why

- Evaluator is a structural consumer interface; ReplayEvaluatorBase is an
  implementation base whose abstract evaluate method prevents accidentally
  inheriting a no-op protocol stub. They serve different purposes and should not
  be merged solely because their method signatures agree.
- _require_models centralizes the same model/reference-model contract check and
  evaluator-specific diagnostic across five implementations. The replay guard
  checks protocol members and callable methods, not tensor values. Removing it
  requires proving all entry points validate these mutable model objects earlier;
  the current direct evaluator API does not provide that guarantee.
- TrajectorySignalBuilder is already the shared owner for recorded old log-probs,
  masks, declared-axis selection and device movement. single_segment packages the
  common one-segment case; segment_signal supports multi-segment assembly. Their
  similar signatures reflect genuinely shared output fields and avoid duplicating
  batch assembly in every evaluator.
- _mask_from_trajectory first honors the requested named mask if it has mask role,
  otherwise requires the unique mask role. This supports named and role-based
  trajectory access; it does not guess from tensor dimensions or retry failures.
- _select_denoise_step finds the axis by its declared kind. A tensor without a
  denoise axis remains intact; a tensor with multiple such axes is ambiguous and
  rejected. This preserves token masks and allows denoise axes in either physical
  position. Do not replace the declaration with fixed-dimension indexing.
- group_ids exposes the batch-owned IDs; context returns a shallow dict copy.
  These accessors are shared by both single- and multi-segment consumers and do
  not need a separate context or metadata wrapper.
- The package initializer deliberately exports nothing and documents direct
  submodule imports. There is no ALL_CAPS workflow vocabulary to extract here.

Non-goals: remove necessary model/trajectory boundary checks, change reference
model selection, add a validator class, flatten shared signal assembly, or move
training-signal fields into the generic trajectory storage representation.

## Validation and limits

60 tests passed on CPU across test_evaluator_contract.py,
test_replay_result_signals.py and test_replay_model_contract.py. Coverage includes
structural replay boundaries, real signal construction, mask-shape mismatch,
unknown segments, canonical trajectory requirement, both declared denoise-axis
positions and rejection of token-axis guessing. This is not pretrained/GPU replay
parity evidence. No new test was needed for the documentation correction.

The builder validates neither a standalone SegmentSignal nor all optional signal
fields. Batch construction compares log_prob against old_log_prob and mask, and
does not establish arbitrary reference/intermediate shape correctness. Its
shape helper skips values without shape attributes. Those existing schema limits
must not be described as complete tensor validation. The context copy is shallow,
and group IDs remain shared rather than a deep immutable snapshot.
