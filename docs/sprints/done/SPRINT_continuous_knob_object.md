# SPRINT: Freeze continuous rollout settings at the runtime boundary

**Date:** 2026-07-18  **Status:** PLANNED

This follows the policy-axis layout work and the args/settings audit. The goal
is to replace repeated seven-argument plumbing with one immutable runtime
snapshot while preserving the trainer-to-rollout import boundary.

## Current evidence

The seven continuous rollout settings are `max_inflight_groups`,
`max_ready_groups`, `max_ready_bytes_mb`, `max_stale_policy_versions`,
`wait_timeout_s`, `queue_poll_interval_s`, and `fail_fast_errors`.

- `vrl/trainers/core/types.py::ContinuousRolloutConfig` owns the user-facing
  defaults and validates the typed training config.
- `vrl/rollouts/orchestration/schedule.py::_build_continuous_schedule` reads
  and converts every field before expanding seven keyword arguments.
- `vrl/rollouts/orchestration/continuous/schedule.py::ContinuousRolloutSchedule`
  repeats the seven-argument signature and forwards it again.
- `vrl/rollouts/orchestration/continuous/owner.py::ContinuousRolloutOwner`
  captures opaque `**runtime_kwargs`; `_ContinuousOwnerRuntime` then repeats the
  full signature a third time.
- `ContinuousRolloutConsumer.__init__` still carries a second
  `fail_fast_errors=3` default even though the trainer config is documented as
  the default source.

`max_stale_policy_versions >= 1` is checked at three boundaries today: the
trainer config, the rollout factory, and the schedule facade. The trainer check
is a legitimate input-boundary validation; the two rollout-layer copies are the
duplicate runtime check to consolidate.

## Planned change

1. Add a frozen `ContinuousRolloutSettings` dataclass to the existing
   `vrl/rollouts/orchestration/continuous/types.py` module. Its seven fields
   have no defaults. Do not create another thin module and do not place the type
   in `continuous/schedule.py`, which would invert the schedule-to-owner import
   direction.
2. Keep `_build_continuous_schedule`'s explicit `getattr`-style projection from
   trainer config. The rollout package must not import `vrl.trainers`. Construct
   the immutable settings snapshot at that boundary.
3. Pass one `settings` object through `ContinuousRolloutSchedule`,
   `ContinuousRolloutOwner`, and `_ContinuousOwnerRuntime`. Remove the owner's
   opaque `**runtime_kwargs` storage.
4. Consolidate the two rollout-layer staleness checks in the settings snapshot.
   Keep `ContinuousRolloutConfig.__post_init__` because it validates the
   independent user-facing config boundary.
5. Make `ContinuousRolloutConsumer.fail_fast_errors` required and pass
   `settings.fail_fast_errors` explicitly. The consumer remains a focused queue
   component and does not need the entire settings object.

Every snapshot field must retain a non-logging consumer in queue capacity,
admission, staleness, timeout, polling, or failure control flow.

## Non-goals

- Do not move the runtime snapshot into `vrl.trainers` or pass
  `ContinuousRolloutConfig` into the rollout package.
- Do not group per-request collector inputs such as `group_size`,
  `runtime_debug`, and `policy_version`; they do not share this run-level
  lifecycle.
- Do not reshape `OnlineTrainer` dependency injection or local timing wrappers.
- Do not add another defaults table. User-facing defaults remain solely on
  `ContinuousRolloutConfig`.

## Verification

- Run the full `tests/rollouts/orchestration` suite and compare behavior with
  the baseline.
- Run Ruff only on touched Python files using the repository's required
  fix/format/check sequence.
- Confirm the seven-field forwarding signatures and owner `**runtime_kwargs`
  are gone.
- Confirm `fail_fast_errors=3` exists only at the trainer config source and all
  snapshot fields have behavior consumers.
