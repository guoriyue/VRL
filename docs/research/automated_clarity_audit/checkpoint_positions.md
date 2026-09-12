# Checkpoint position access (scoped review)

Following 874a269e7, inspect TrainingCheckpoint loading/accessors, TrainingResumeConfig,
the schema-root validator, restore entry point and Wan offline position writer.
checkpointing.py remains pending its complete module review; this does not cover
the publication, adapter export and distributed agreement implementations.

Change: read progress once in _resume_position rather than evaluating its property
twice. Correct next_step's docstring: it is a loop position defined by the writer,
not universally an optimizer-step count. Wan increments it once per preference
microbatch, while OfflineDPOTrainer updates the optimizer only at accumulation
boundaries. Neither saved values nor resume behavior change.

Retain and why:

- next_epoch/next_step delegate to the same explicit progress accessor. It does
  not inspect checkpoint directory suffixes, trainer counters or metadata as
  fallback. A new generic position resolver would reintroduce unnecessary layers.
- Accessor validation is not all duplicated by load: _validate_checkpoint_payload
  validates schema/model roots, while progress is separately read and may also
  be supplied through direct TrainingCheckpoint construction. Keep the existing
  explicit field requirement; do not add another global validation pass.
- Schema version, checkpoint filenames and strict-default constants represent
  serialized/public protocol boundaries. trainable_state is a compatibility
  property projecting the schema-specific model state, not a guessed field value.
- TrainingResumeConfig owns only selected path and strictness. CheckpointTarget
  serves evaluation selection, including optional digest and completed epoch;
  it does not replace resumable trainer state.

Non-goals: alter supported schemas, progress units, missing-value behavior, digest
cost, checkpoint serialization or add new checks/tests. Serialized mutable dicts
are not a deeply immutable snapshot merely because the outer dataclass is frozen.

22 existing selected checkpoint tests passed, covering explicit saved positions,
absence of old-field/name inference, resume config and metadata projection. Ruff
check and format check passed for checkpointing.py. No new tests or hypothetical
examples were added. Whole checkpoint publication/restoration is not established
by this focused run.

Follow-up for the offline lifecycle review: the Wan writer's checkpoint interval
is expressed in loop steps, and optimizer gradients are not serialized. Therefore
an interval not aligned with accumulation can save an incomplete accumulation
window. The current trainer intentionally restarts the window on restore; do not
advertise exact mid-window continuation or infer the missing gradients from the
step number. Inspect actual recipe intervals before proposing a change or gate.
