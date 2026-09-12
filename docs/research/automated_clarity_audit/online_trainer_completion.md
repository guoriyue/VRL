# Online trainer: diagnostics, regularization and restore

Following ed6a3d5ad, read the remaining stats/events, precision/parity, SFT,
checkpoint and precision metadata methods. Together with online_replay_metrics.md
and online_step_execution.md this completes the online/trainer.py source review.

Change: the optional first-step evaluator probe uses train_indices[0] instead of
hardcoded index 0. Random/stratified subsets and a recorded SDE window can start
later in the denoise schedule. SDELogProbEvaluator uses the supplied index for
both replay_forward and trajectory axis selection, so index 0 does not stand for
the first selected step. Trajectory evaluators still receive their existing 0
sentinel. Training loss, selected indices, RNG draws and full-update parity
enforcement do not change. This closes the optional probe follow-up in
online_step_execution.md for evaluator-backed replay; the evaluator-less algorithm
invariant remains its separate existing protocol.

Evidence: a one-off CPU run reused tests/trainers/online/test_step_split.py's
_build_trainer fixture with first-step debugging, random selection, fraction 0.5
and torch seed 1. Before the change, evaluator calls were (0, no_grad), (1, grad),
(1, grad); afterward all three used index 1. The two training calls are the
fixture's PPO epochs. This exercises a supported selection, not malformed data.
No new repository test or runtime gate was introduced.

Retain and why:

- _step_stats merges the schedule's owned accumulator with trainer timings.
  _write_phase_events drains the timer's events. Keep these distinct from reward
  metrics aggregation; they represent different output schemas and lifetimes.
- Precision drift measurement's local callback captures the batch and model for
  an existing measurement API. A class would add state without simplifying that
  boundary. Role metadata describes observations; its dtype fallbacks are not
  checkpoint position inference or authoritative model construction.
- Mandatory exact parity is reset on resume and checked before the first actual
  optimizer update. Optional debug evidence cannot replace it. Intentional
  precision correction instead uses its existing drift policy.
- SFT resolves a clean target through recorded reward metadata and uses the
  evaluator's scheduler and model replay hook. Geometry checks compare separately
  encoded data to rollout geometry, a real external-data boundary. Keep them.
  Shared rank timestep selection and per-slice normalization remain unchanged.
- state_dict/checkpoint_state_dict share implementation but have different FSDP
  retention policies. Keep the two public facades and collective export seam.
- The optimizer parameter manifest binds positional optimizer slots to named
  trainable tensors. FP32 master residuals, scaler history and EMA shadows are
  actual saved state and cannot be reconstructed from rounded model weights.
- Successful restore resets rollout initialization, parity/drift status and the
  schedule. Existing tests cover rechecking at nonzero progress and publication
  of restored weights. These are runtime lifetimes, not persisted success flags.
- Event keys, checkpoint fields and optimizer manifest fields are serialized
  schemas. No workflow ALL_CAPS vocabulary requires moving into a config table.

Limits retained / deferred:

- Phase-event writing catches every Exception and stays best-effort; events clear
  only after a complete successful loop, so a partial append followed by retry can
  duplicate events. Deferred as an output-policy decision: replacing this with the
  diagnostic append helper would change failure propagation and per-call I/O.
  This review does not claim durable or exactly-once trace recording.
- Trainer restore is not transactional across counters, optimizer, scaler and
  EMA. Non-strict mode can continue after a failed component load; strict mode can
  raise after earlier components changed. No rollback infrastructure was added.
- The SFT rank-shared timestep does not by itself establish full multi-rank RNG
  resume equivalence; per-rank noise and other RNG consumers still exist.
- V-GRPO algorithm-private counter persistence remains the earlier independent
  finding; trainer counter fields are not a substitute for that algorithm state.

Non-goals: redesign diagnostics, change optional inference fallback APIs, add
checker classes, change training mathematics, or claim GPU behavior from CPU
tests. The full repo audit remains incomplete even though this module is reviewed.

Validation: 82 existing diagnostics, step-split, state-restore, SFT and Flash-GRPO
tests passed; two tests skipped. The one-off before/after run above also passed its
expected-call assertion. Ruff check and format check passed for trainer.py.
Coverage is now 191 reviewed and 308 pending baseline modules.
