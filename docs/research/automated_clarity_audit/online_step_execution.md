# Online collection and update execution (scoped review)

Following 397cfc097, read replay loss, gradient clipping/stepping, the step entry
point, reward component extraction, collect/filter, timestep/replay selection,
streaming finish and the full-batch PPO loop including optional debug recording.
Together with online_replay_metrics.md, source inspection now reaches the end of
train_on_rollout_batch. Stats/event methods, precision/parity enforcement, SFT
computation and trainer state restoration remain to complete the module review.

Change: inline _step_impl's two statements into step's existing profiler scope.
The only production call was step, with no overrides or independent test callers.
The profiler still encloses collection and training; next_prompts still reaches
collect_training_batch unchanged. Update the existing forwarding test's docstring
to describe the shorter call chain and use prefetch terminology.

Retain and why:

- collect_training_batch and train_on_rollout_batch are public execution seams
  used independently in tests and by orchestration. They return/consume the
  existing TrainingBatch owner. Removing _step_impl does not collapse that seam.
- _compute_replay_loss is shared by both update paths and separates evaluator
  replay from evaluator-less objectives. Their precision boundaries differ.
  Per-segment parity observes the actual replay result, not algorithm-reported
  defaults. Keep those responsibilities together for now.
- _clip_and_step centralizes master-gradient preparation, unscale, strategy-aware
  norm computation, nonfinite handling and scaler skip detection. These real
  optimizer lifetimes justify a helper. Even max_norm disabled requires a global
  FSDP norm, so the strategy call cannot be replaced by a local tensor reduction.
- Timestep selectors implement real strided/random/stratified sampling. Recorded
  SDE windows are generation facts, not trainer guesses. Trajectory evaluators use
  one invocation instead of flattening chunk/transition axes into unrelated steps.
  Keep the typed axis lookup and window agreement checks at that consumer seam.
- Streaming begin/backward/finish and full-batch PPO replay have different data
  lifetimes. Both share _run_replay_pass but retain their optimizer orchestration.
  Scaler-skipped updates must not run EMA/algorithm hooks or publish new weights.
- Reward component conversion adapts extras from tensor/sequence producers and
  preserves pre-filter diagnostic values. A wrapper class would not eliminate
  this boundary or its sample alignment requirement.
- Debug event names and field names here are serialized evidence keys, not a
  business vocabulary requiring an extracted configuration table.

Non-goals: alter training math, sampler RNG, reference policy, diagnostic schemas,
optimizer attempt counters or move all methods into another class. No new tests
or validation gates were introduced.

Follow-up: optional first-step debugging invokes evaluator index 0 directly,
whereas actual replay uses selected indices. Check its behavior with a nonzero
recorded SDE window before treating that probe as evidence for the trained step.
The mandatory full-update gate is separate; no failure is claimed without tracing
the evaluator/window representation and its existing coverage.

Validation: 66 existing step-split, GradScaler, trajectory-granularity, DanceGRPO
and V-GRPO tests passed on CPU, with 12 warnings. Ruff check and format check passed
for both touched Python files. Existing tests cover step/split equivalence,
next-batch forwarding, synchronization placement and real scaler skipping. No GPU
or multi-rank coverage is claimed for this run. Module coverage remains 190/499.
