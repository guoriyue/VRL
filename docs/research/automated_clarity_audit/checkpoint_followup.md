# Checkpoint ownership follow-up

Reviewed the complete `vrl/trainers/activation_checkpointing.py` module and its
tests, plus its compile-matrix and family-registry callers. Read the trainer's
state serialization/restoration and successful/skipped optimizer hook branches,
and `restore_training_checkpoint`'s model-then-trainer restore sequence. The much
larger checkpointing, model identity and online trainer modules remain pending;
selected sections are not marked as complete source coverage.

## Fixed: selective checkpointing swallowed model errors

The selective setup previously caught every TypeError from
`enable_gradient_checkpointing(gradient_checkpointing_func=...)`, interpreted it
as an unsupported argument and called the same method again without arguments.
This hides an internal model setup failure, and can retry after partial mutation.

Bind the keyword against the callable's signature before invoking it. A binding
TypeError selects the existing logged full-checkpointing fallback. The actual
model method runs outside that catch, so an internal TypeError reaches the
caller and is not retried. No new wrapper class or capability table.

The regression first failed on the original implementation: a method accepting
the keyword raised internally, but setup logged a fallback and returned success.
It now raises the original error and is called exactly once. Existing tests
retain support for kwargs-based methods and legacy no-keyword methods.

Compatibility: supported repository adapters and installed Diffusers methods
are inspectable Python callables. A custom opaque callable with no inspectable
signature now fails during inspection rather than using an exception-driven
trial call. A kwargs wrapper that forwards to a legacy implementation must
expose the actual capability in its signature or handle forwarding itself;
errors inside the wrapper are no longer classified as unsupported arguments.

## Keep

- `_GRADIENT_CHECKPOINT_SAVE_OPS` is the deliberately isolated SAC policy table
  of torch operator identities. It determines saved versus recomputed outputs;
  replacing it with algorithm/family names or moving it to YAML would obscure
  that policy and change the dependency boundary.
- `_selective_checkpoint_policy` has the torch callback signature. The public
  `selective_checkpoint_func` is the Diffusers adapter also used by the backward
  performance probe. The small lambda supplies torch's zero-argument context
  factory. These are framework boundaries, not incidental forwarding functions.
- `_normalize_gradient_checkpointing` owns existing bool/string compatibility;
  `resolve_gradient_checkpointing_mode` is shared by setup, compile conflicts
  and the NextStep family gate. Mode resolution must remain consistent across
  those consumers. Do not change accepted spellings in this refactor.
- Preserve explicit rollout-versus-replay compile scope and per-module fallback
  warnings. No changes to the save-ops policy, recomputation math or model hooks.

## Verified open issue: V-GRPO resume noise counter

Strengthened the previous source finding with an actual OnlineTrainer state
roundtrip, using the existing CPU resume fixture and a VGRPO instance on each
trainer. The source objective received `after_optimizer_step(..., global_step=7)`
with a no-op adapter-sync hook and had counter 8. Saved/restored trainer counters
were step=8/global_step=8. Strict trainer restore succeeded, but the fresh
objective's counter stayed 0. With identical latents, group IDs and replay index,
the next generated noise differed. No model forward or full disk checkpoint
roundtrip was claimed by this focused reproduction.

This confirms the runtime state is omitted, rather than merely absent from one
search result. The fix still needs an algorithm-state persistence contract at
the checkpoint/trainer boundary. Restoring from global_step would be guessing:
the trainer increments global_step even when GradScaler skips an optimizer
update, while the objective hook is skipped. Calling that hook during restore
would also resynchronize/decay model adapters. Do neither as a shortcut. Keep
this actionable issue open for the full checkpoint owner review; strict and
non-strict handling of older snapshots needs an explicit decision and tests.

## Validation

30 activation-checkpointing and config drift/compile tests passed after the fix.
An installed Diffusers ModelMixin subclass accepted the production selective
function and completed a CPU backward pass. Ruff check/format and diff whitespace
checks passed on the two changed Python files. An initial test command used a
nonexistent drift test filename and collected no tests; the corrected command
used `tests/config/test_rollout_drift_guard.py` and completed successfully.

No GPU performance or memory claim. Previous isolated audit commit: `276c598d4`.
