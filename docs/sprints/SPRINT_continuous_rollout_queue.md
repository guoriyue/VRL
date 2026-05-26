# Sprint: Continuous Rollout Queue (async / off-policy producer→buffer→consumer)

状态：proposed (decision + scoped execution)

> Cross-ref: `SPRINT_continuous_rollout_from_slime.md` is the detailed 7-phase blueprint
> (class names, config keys, metric names). This sprint is the **decision + single-GPU-safe
> scoped execution** of that blueprint's Phases 1–4 (its Phase 7 = this sprint's Phase D).
> If keeping both feels like sprawl, fold this into that doc as its top "Decision" section.

## 1. 结论 / Decision

Question asked: should we build a "workload queue to throw rollout AND reward tasks to,"
slime-style?

Research reframed it:
- **Slime has no unified rollout+reward queue.** Reward is computed **inline** with
  generation (`generate_and_rm` / `generate_and_rm_group`, same `RolloutManager` actor).
  Slime's async-ness = an asyncio pending-task pool + `RolloutDataSourceWithBuffer`
  (a buffer for completed/partial rollout *groups*) + batch-level pipelining.
- **vrl already splits reward** into a transport (`make_reward_runtime` →
  `LocalRewardRuntime` in-process *or* `RayRewardRuntime` actor pool). Reward "thrown to a
  pool" already exists.
- The missing piece vs slime is continuous *production* with a bounded ready buffer +
  staleness — already blueprinted in `SPRINT_continuous_rollout_from_slime.md`.

**Decision: build the continuous rollout queue** (producer → bounded ready buffer →
consumer), **NOT** a generic "one queue for both rollout and reward." Reward stays its own
transport; the queue carries **completed rollout items at the OUTPUT**. Land it **staged,
default-off** (behavior-unchanged) so single GPU works today and multi-GPU drops in with no
contract rewrite.

## 2. 底层逻辑 / Honest caveat — where the payoff actually is

On a **single GPU**, rollout and reward **time-share** the device
(`_should_release_runtime_before_reward_model` in `vrl/rollouts/collector/core.py`), so a
queue's concurrency win is a **multi-GPU** payoff (producer generates while reward scores
while trainer trains). On one GPU it buys only cadence decoupling; the target path is
separate trainer and rollout GPU ownership before the hardware.

## 3. Reuse (do not rewrite) — with paths

- `RolloutLifecycle`, `record_phase`, `current_policy_version()`, `sync_weights_after_train`,
  `runtime_is_colocated()` — `vrl/rollouts/orchestration/lifecycle.py`
- `RolloutIteration`, `RolloutScheduleState`, `RolloutScheduleMode`, `build_rollout_iteration`,
  `annotate_batch_context` — `vrl/rollouts/orchestration/types.py`
- `RolloutSchedule` protocol + `build_rollout_schedule()` factory —
  `vrl/rollouts/orchestration/schedule.py`
- `collect_prompt_batches()` (producer body; routes reward via the existing transport) —
  `vrl/rollouts/orchestration/prompt_collection.py`
- `RolloutCollector.collect` + `_should_release_runtime_before_reward_model()`
  (gives correct single-GPU time-share for free) — `vrl/rollouts/collector/core.py`
- Reward transport `make_reward_runtime` (`local`/`ray`) — `vrl/rewards/runtime.py`

## 4. 目标结构 / New files — `vrl/rollouts/orchestration/continuous/`

- `__init__.py` — export `ContinuousRolloutSchedule`
- `types.py` — `ContinuousRolloutItem` (wraps a `RolloutBatch` + `policy_version` +
  `prompt_key`), `ContinuousRolloutProducerState`
- `queue.py` — `ContinuousRolloutQueue`: bounded by max_items/max_samples/max_bytes;
  `put()` (backpressure when full), `drain_for_iteration(group_size, current_policy_version,
  staleness)` returning a **single homogeneous policy version**, `pause_admission()` /
  `resume_admission()`, `drain_inflight()`, `stats()`
- `staleness.py` — `StalenessPolicy(max_stale_policy_versions, drop_too_stale)`;
  `admit(item, current_version) -> bool`
- `producer.py` — `ContinuousRolloutProducer`: in-process `asyncio` loop calling
  `collect_prompt_batches(collector=lifecycle.collector, ...)`, stamping
  `lifecycle.current_policy_version()`, `await queue.put(...)`. (Ray-actor variant = Phase D,
  same contract.)
- `consumer.py` — `ContinuousRolloutConsumer`: `drain_for_iteration` →
  `build_rollout_iteration` + `annotate_batch_context`
- `schedule.py` — `ContinuousRolloutSchedule` (implements the protocol): `next_iteration`
  starts producer if idle then drains; `after_train_step` runs barrier
  (`pause_admission` → `drain_inflight` → `sync_weights_after_train` → `resume_admission`);
  `reset` cancels producer task

## 5. Edits (small, additive)

- `vrl/rollouts/orchestration/types.py`: add `CONTINUOUS = "continuous"` to
  `RolloutScheduleMode`
- `vrl/rollouts/orchestration/schedule.py`: factory branch for `CONTINUOUS`; relax the
  `max_pending_rollouts == 1` guard to allow >1 only for continuous
- `vrl/rollouts/orchestration/__init__.py`: export `ContinuousRolloutSchedule`
- `vrl/trainers/core/types.py` `RolloutOrchestrationConfig`: allow `mode="continuous"`; add
  `max_stale_policy_versions:int=0`, `drop_too_stale:bool=True`, `queue_max_items`,
  `queue_max_samples`, `queue_max_bytes_mb`, `min_prompt_groups_per_iteration`; allow
  `weight_sync_barrier="pause_admission_and_drain_inflight"` for continuous.
  **Default mode stays `strict_on_policy`.**
- No `trainer.py` edits — it already drives `next_iteration` / `after_train_step` against the
  protocol (≈ lines 294, 772).

## 6. Off-policy correctness risks + mitigations

- **Mixed policy versions in a batch** breaks PPO ratio/advantage → `drain_for_iteration`
  returns one homogeneous version; `annotate_batch_context` already stamps
  `rollout_policy_version`.
- **Stale gradients / IS divergence** → bounded by `max_stale_policy_versions`
  (default **0** = on-policy-equivalent); `drop_too_stale` discards rather than trains past
  the bound.
- **Weight-sync race** (item straddling a version bump) → the `pause→drain→sync→resume`
  barrier.

## 7. Single-GPU now vs multi-GPU later (no contract rewrite)

- Now: producer/queue/consumer/staleness all in-process; reward via `LocalRewardRuntime`;
  `collect()`'s time-share keeps gen+reward from colliding on one device.
- Later (Phase D): swap `ContinuousRolloutProducer` to a Ray actor + `make_reward_runtime("ray")`
  reward pool. Items still arrive as completed `RolloutBatch`; queue/consumer/`RolloutIteration`
  contract unchanged.

## 8. 分阶段计划 / Phasing (incremental, behind config)

- **Phase A (ship first):** new dir + types/queue/staleness/consumer/schedule (in-process
  producer) + enum/config/factory edits. `max_stale=0`, queue depth 1 ⇒ output-equivalent to
  strict. Default-off.
- **Phase B:** enable `max_stale>=1` + bounded prefetch (queue depth >1) — first real
  off-policy/throughput behavior (most visible on multi-GPU).
- **Phase C:** byte-cap + queue-wait metrics into `phase_times`.
- **Phase D:** Ray-actor producer + `ray` reward pool (multi-GPU / multi-node payoff).

## 9. 验证矩阵 / Verification

- New unit tests under `tests/rollouts/orchestration/continuous/`: `test_queue.py` (bounded
  admission, same-policy drain, `drop_too_stale`), `test_staleness.py`, `test_schedule.py`
  (barrier ordering pause→drain→sync→resume; `max_stale=0` matches strict output).
- Factory test: `mode:continuous` builds `ContinuousRolloutSchedule`; `strict_on_policy` is
  still the default.
- Parity: run the existing online-trainer integration test under `mode:continuous, max_stale=0`
  and confirm identical output to strict (proves Phase A is behavior-equivalent).
- `ruff check vrl tests`.

## 10. 非目标 / Non-goals

- No generic "rollout+reward input task queue" — reward stays a transport; the queue is at
  the output (completed rollouts).
- No change to default behavior (default mode stays `strict_on_policy`).
- No Ray-actor producer in the first landing (Phase A is in-process).
- No PPO/algorithm semantic change beyond bounded staleness gated by `max_stale_policy_versions`.
