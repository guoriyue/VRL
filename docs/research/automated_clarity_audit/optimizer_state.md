# Optimizer state and trainer configuration ownership

Reviewed complete optimizer.py, core/types.py and core/__init__.py, with online
optimizer selection, clipping/scaling and state restore/export callers, typed
public configuration sections, and optimizer/GradScaler/continuous contract
tests. Checkpointing.py and the full online trainer remain pending; their scoped
call-site inspection does not constitute complete module coverage.
Previous audit commit: 2089eb608.

## Changes

- Remove the Tensor assertion after _validate_master_state has already checked
  every low-precision source's saved master tensor, dtype and shape. Keep the
  checkpoint boundary validation itself. This does not change accepted state.
- Correct the optimizer module's LoRA claim: selection follows actual trainable
  dtypes, not the LoRA flag. FP32 adapters avoid duplication; low-precision
  adapters still need master residuals.
- Correct OptimConfig's 8-bit comment: quantized optimizer state can change
  weight updates and subsequent logprobs. It does not itself select a different
  forward precision for rollout and replay. No optimizer backend was changed.

## Retain and why

- build_optimizer is a backend factory with a lazy bitsandbytes import. It
  shares OptimConfig with the trainer and returns standard AdamW by default.
  A stateless factory class would not improve ownership. No independent bucket
  setting or replacement optimizer is introduced.
- FP32MasterWeightOptimizer owns source/master bindings and gradient lifecycle.
  Its state_dict extension preserves sub-ULP residuals; reconstructing masters
  from rounded model parameters on resume would lose real training state.
- Factory identity/order checks protect positional source/master publication.
  Shape, dtype, backend and master-presence checks protect an external saved
  state boundary. Gradient preparation, closures and dynamic group insertion
  have explicit supported lifetimes. These are not generic tensor assertions
  scattered through ordinary forward computation.
- _make_master expresses FP32 source reuse versus low-precision copying;
  _copy_masters_to_sources serves both stepping and restore. _optimizer_type
  supplies the same checkpoint identity on save and load. Keep these owned
  operations, not a second utility namespace or a monolithic restore method.
- Wrapper param_groups/state track the wrapped optimizer's live objects,
  including after load. This lets schedulers, GradScaler and clipping operate
  on the parameters actually stepped. Zeroing resets preparation after a
  skipped scaled update as well as after a successful update.
- Trainer configuration dataclasses are the schema consumed by public Pydantic
  sections. Constructor checks for continuous capacity/version bounds and
  finite timeouts precede projection; they are not duplicated rollout defaults.
  TrainState carries counters, not algorithm-specific resume reconstruction.
- Schedule and reward-arm strings are deliberately isolated config vocabulary
  avoiding a torch-heavy rollout import. Empty core package exports keep this
  configuration import path light. Checkpoint extension keys and __all__ are
  serialization/public boundaries. There is no workflow business table to move.

Non-goals: changing training math, replacing standard AdamW, removing FP32
masters, moving every helper into a class, accepting arbitrary optimizer
parameter-group reordering, or changing schema/version compatibility.

## Follow-up and limits

The PrecisionDriftGuardConfig finding is closed by the follow-up recorded in
precision_guard.md: thresholds are normalized once, NaN is rejected, positive
infinity remains supported, and check counts use the shared integer boundary.
No repeated validation was added to the measurement loop.

Master checkpoint state_dict returns detached tensors sharing master storage,
like ordinary torch state dictionaries; it is not an immutable asynchronous
snapshot. Snapshot ownership belongs to strategy/checkpoint export. Restore
prevalidates master metadata but is not a transaction across wrapped optimizer
loading and all tensor copies. No stronger atomicity claim is made.

## Validation

101 optimizer, online GradScaler and continuous contract tests passed, one
optional bitsandbytes test skipped and one GPU test deselected. The existing
checkpoint roundtrip compares invisible master residuals, momentum and later
updates; CPU GradScaler tests exercise unscale, clipping and skipped updates.
Ruff passed for both changed production modules. No new test was added for the
redundant assertion removal or comment corrections. GPU FSDP and 8-bit update
behavior are not established by this CPU run.
