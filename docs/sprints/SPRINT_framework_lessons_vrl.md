# SPRINT: Framework lessons → vrl architecture improvements

Source material: the five architecture readings under `docs/sprints/reading/`
(cosmos-rl, vllm, sglang, sglang-omni, slime), cross-checked against the
current vrl code. Every gap below cites both sides: where vrl stands today
(path:lines in this repo) and the prior art (path:lines inside the reading
docs / the studied repos).

## Current vrl spine (verified by reading, 2026-06-11)

```
vrl/scripts/train.py            YAML trainer.entrypoint dispatch (89 lines)
  └─ scripts/common/online.py   run_online_recipe: flat epoch loop (299-352)
       └─ trainers/online/trainer.py  OnlineTrainer.step: collect → advantage
          → filter → grad-accum (364+)
            └─ rollouts/orchestration/continuous/   producer + queue + consumer
               (asyncio, version-stamped groups, StalenessPolicy)
                 └─ generation/ray/                 Ray actors; per-request
                    EnginePlanner static plan (execution/planner.py)
```

- Weight sync: trainer flattens trainable state to CPU
  (`vrl/trainers/weight_sync.py:56-62`), pushes the dict through Ray to every
  worker (`vrl/generation/ray/weight_sync.py:34-48`), worker calls
  `model.load_trainable_state(state_ref)`
  (`vrl/generation/execution/worker.py:70-81`).
- Sync barrier: `pause_admission → drain_inflight → sync → resume`
  (`vrl/rollouts/orchestration/continuous/schedule.py:110-120`); the producer
  drains rather than cancels so no request straddles a version bump
  (`producer.py:121-131`).
- Version hygiene already half-built: every group is stamped at submit
  (`producer.py:165-177`), consumer enforces `StalenessPolicy`, and workers
  refuse chunks whose `request.policy_version` mismatches the loaded version
  (`worker.py:115-129`).
- Colocated mode: `ReleasableRayGenerationRuntime` kills and relaunches the
  generation actors every release/ensure cycle — full model reload
  (`vrl/generation/ray/runtime.py:95-175`).
- Generation planning: `EnginePlanner.build()` produces one immutable
  per-request plan with fixed-size sample chunks
  (`vrl/generation/execution/planner.py:205-258`); there is no cross-request
  batching or budget-driven admission.

---

## P0 — directly blocking the full-parameter cosmos direction

### P0-1. Weight-sync data plane: CPU state dict over Ray object store

**vrl today.** Every push detaches the full trainable state to CPU and ships
it through Ray's object store to each worker
(`vrl/trainers/weight_sync.py:56-62, 182-195`;
`vrl/generation/ray/weight_sync.py:34-48`). Fine for LoRA-sized payloads; the
repo just moved cosmos predict2/2.5 to full-parameter training (commit
`f9b7f9a`), which makes this a multi-GB pickle + host-RAM round trip per
training step, serialized once per worker.

**Prior art.**
- slime ships two GPU-direct paths: CUDA-IPC tensor handles for colocated
  (`slime: update_weight/update_weight_from_tensor.py:122-128, 209-267`) and
  bucketed NCCL broadcast per PP group for disaggregated
  (`update_weight_from_distributed.py:62-140, 234-248`), with
  pause→flush→chunked-send→continue choreography
  (`update_weight_from_tensor.py:138-181`). See `reading/slime.md` §4/§9.
- cosmos-rl precomputes a per-rank slice plan (`WeightSyncInstruction`) from
  DTensor metadata and executes raw NCCL unicast — the training model is never
  resharded (`cosmos-rl: dim_slice_info.py:211-240,
  parallelism_map.py:1152-1166`). See `reading/cosmos-rl.md` Part II §10.

**Move.** Keep the Ray-object-store path as the LoRA/small-payload fallback;
add an NCCL broadcast group (trainer rank 0 → rollout workers) with bucketing
for the full-parameter path. vrl is single-trainer-rank today, so the
cosmos-rl slice algebra is not needed yet — slime's bucketed broadcast is the
right size of solution.

### P0-2. KL reference model without LoRA: CPU swap, not a second model

**vrl today.** `default_reference_model` returns the LoRA policy itself (LoRA
disabled at ref time) or `None`
(`vrl/scripts/common/online.py:74-80`). With LoRA dropped for cosmos, KL
either silently disappears (`None`) or would require a full second model in
GPU memory.

**Prior art.** cosmos-rl never materializes a second model: a CPU-resident
state dict is swapped into the live sharded model for the REF_COMPUTE phase
and swapped back, zero extra GPU memory
(`cosmos-rl: grpo_trainer.py:1955-1970`). slime generalizes the same trick to
four logical policies via pinned-CPU `TensorBackuper`
(`slime: utils/tensor_backper.py:54-74, actor.py:435-448`). See
`reading/cosmos-rl.md` Part II §10, `reading/slime.md` Part II §10.

**Move.** Snapshot the initial policy state to pinned CPU at startup; for KL
evaluation, swap in / compute ref logprobs / swap back inside
`OnlineTrainer._step_impl`'s replay phase. This restores KL for full-param
runs at the cost of two H2D/D2H copies per step.

---

## P1 — throughput on the existing architecture

### P1-1. Non-draining weight-sync barrier

**vrl today.** `after_train_step` pauses admission and **drains every
in-flight generation request** before syncing
(`schedule.py:110-120`). With video clips at ~5-6 min each
(`project_wan_i2v_14b_inference`), the barrier can stall training for minutes
per step.

**Prior art.** cosmos-rl avoids the global drain with three cooperating
mechanisms (see `reading/cosmos-rl.md` §3/§8):
1. prompts stamped with a *predicted* weight version at admission plus an
   `allowed_outdated_steps` soft window;
2. late rollouts dropped on arrival (and the worker re-stamps the actual
   generating version, `rollout_control.py:1893-1902`);
3. weight sync executed asynchronously into a buffer model on a side CUDA
   stream, swapped at generation boundaries (`weight_sync.py:16-50`), with
   mid-generation command drains every N engine steps
   (`vllm_rollout.py:77-119`).

**Move.** vrl already has the hard parts: version stamping at submit
(`producer.py:165-177`), consume-side `StalenessPolicy`, and chunk-level
version refusal (`worker.py:115-129`). Replace the global drain with:
sync pushes the new state to workers *while generation continues*; each worker
swaps weights at its next chunk boundary (chunks are already the version
checkpoint); the consumer's existing staleness window absorbs the mixed-version
tail. `max_stale_policy_versions=0` keeps today's strict mode available.
The one invariant to preserve: a single denoise trajectory must never mix two
policy versions — the swap point must stay at request/chunk boundaries,
exactly where `worker.py:115-129` already checks.

### P1-2. Colocated mode: sleep/wake instead of actor teardown

**vrl today.** `ReleasableRayGenerationRuntime.release_memory()` shuts the
runtime down and `_ensure_runtime()` relaunches actors from scratch — a full
model load from disk every collect/train cycle
(`vrl/generation/ray/runtime.py:150-175`).

**Prior art.** slime keeps engines alive and stages memory:
release KV → offload weights, then on resume onload weights *first*, push new
weights, only then re-allocate KV/graphs (`slime: train.py:91-97,
ray/rollout.py:326-346`), with `flush_cache` refusing unless idle and weight
updates asserting the flush (`sglang: scheduler.py:3229-3254`). See
`reading/slime.md` §8.10, `reading/sglang.md` Part II.

**Move.** Add `sleep()/wake()` to the generation worker: drop CUDA caches and
move the executor's weights to pinned CPU instead of killing the actor. The
launch contract, capability negotiation, and compile state all survive, and
wake + `update_weights` replaces a cold reload. This also softens the
"continuous rollout requires separate GPUs" restriction
(`schedule.py:199-209`) for the single-node debug case.

### P1-3. Straggler control for video rollout groups

**vrl today.** The producer keeps `max_inflight_groups` collect jobs running
and the consumer waits for `min_groups` full groups with a 300 s timeout
(`producer.py:151-157`, `schedule.py:97-105`). One slow clip holds back its
whole group.

**Prior art.** slime over-provisions sample groups, collects with
`asyncio.wait(FIRST_COMPLETED)`, aborts the tail via the engine's
`/abort_request`, and resumes partial rollouts from tokens
(`slime: sglang_rollout.py:345-426, 191-218`). See `reading/slime.md` §8.9.

**Move.** Submit ~1.2-1.5× the needed groups, take the first N complete,
cancel the rest at the executor level (the Ray chunk envelope already carries
`request_id` to target). Partial-rollout resume is diffusion-unfriendly
(latent trajectories are not resumable across versions) — skip that half.

---

## P2 — cheap engineering borrows

- **Cost-budget batching for VAE encode/decode.** Chunk size today is a fixed
  `max_samples_per_chunk` (`planner.py:245-258`). sglang-omni batches
  non-AR stages by (max size, max wait, per-request *byte-cost* budget with a
  measured activation multiplier) (`sglang-omni: simple_scheduler.py:101-130`)
  — the right shape for resolution-heterogeneous video batches.
- **`async_*` naming contract.** slime documents one prefix = "returns
  futures" (`slime: actor_group.py:13`); vrl mixes sync/async surfaces
  (e.g. collector vs runtime). Adopt the convention, no behavior change.
- **Metrics row derivation.** `_prepare_metrics_csv`/`_write_metric_row`
  hand-maintain a 20-column positional CSV
  (`vrl/scripts/common/online.py:375-464`) — the exact "hand-maintained
  duplicate of a typed structure" AGENTS.md warns about. Derive header and row
  from `TrainStepMetrics` field names + the reward component list.
- **Per-request stat accumulators on the sample.** slime accumulates engine
  stats on the `Sample` object itself across partial rollouts
  (`slime: types.py:53-120`); vrl carries per-iteration stats in
  `iteration.phase_times` dicts (`schedule.py:167-197`). For per-request
  denoise stats (steps, cache hits, per-phase latency), the request object is
  the better home.
- **Attributed, measured comments.** sglang-omni's `Note (name):` comments
  with measured numbers (600× GIL, bs=1 regression) are the cheapest
  provenance system for tuning constants — apply to vrl's batch-size /
  compile-mode constants already documented in project memory.

---

## P3 — the big one, as direction not as a rewrite

### Continuous batching at denoise-step granularity

**vrl today.** Generation is request-scoped: `EnginePlanner.build()` makes an
immutable plan per request (`planner.py:205-243`) and workers execute its
chunks to completion. Two requests never share a forward pass; admission is
"a worker is free", not "the step budget has room".

**Prior art (all three serving engines).**
- vLLM has no prefill/decode phases at all — every request is a
  `num_computed_tokens` deficit catching up to its target, and chunking is
  `min(deficit, budget)` (`vllm/v1/core/sched/scheduler.py:321-331`); admission
  and preemption both hang off one signal, `allocate_slots() -> None`
  (`reading/vllm.md` §3/§8).
- sglang's overlap loop pipelines CPU scheduling of step N+1 with GPU forward
  of step N via a result-queue deque + FutureMap placeholders
  (`reading/sglang.md` §3).
- sglang-omni's launch/resolve split (`model_runner/base.py:20-43`) is the
  same idea packaged per stage, with two pinned ping-pong host buffers and a
  `min_batch_size` gate (`reading/sglang-omni.md` §8.2).

**Mapping to vrl.** `DenoiseLoopState`-style `current_step/total_steps` (the
old engine design, and what `axis kind="timestep"` in
`planner.py:277-279` still models) *is* the counter-deficit shape. A
denoise-step-batched engine would: keep per-request step counters; each engine
tick, group compatible requests (same family/resolution — today's
`batch_group_key`, `planner.py:382-389`) into one forward; admit new requests
whenever the latent-memory budget allows (`allocate() -> handle | None` as the
single backpressure bit).

**Why not now.** This rebuilds the generation executor while the RL loop is
the current bottleneck consumer (GRPO groups arrive as whole requests anyway;
batching within a request already exists via chunks). It becomes worth it when
(a) serving-style mixed workloads matter, or (b) GPU utilization on
small-batch families (wan sbs=1, AR — `project_cross_model_smoke_20260609`)
must come up without resolution-homogeneous groups. Track it as the
`docs/NEURAL_ECS_ENGINE_DESIGN.md` evolution; do not bolt it onto the current
executor piecemeal.

---

## What NOT to change (the readings argue *for* current vrl choices)

- **YAML `trainer.entrypoint` dispatch** (`vrl/scripts/train.py:28-55`) is
  exactly slime's `*_path` + `load_function` plugin pattern
  (`slime: misc.py:9-17`) — zero import-time coupling. Keep.
- **The flat driver loop** (`online.py:299-352`) matches slime's "keep the
  driver loop flat and sacred" (`reading/slime.md` §8.1). Resist layering it.
- **Producer/queue/consumer split with version stamping** is structurally the
  same design slime and cosmos-rl converged on. Extend it (P1-1), don't
  replace it.
- **Protocol-based interfaces** are the side sglang's own 11-mixin Scheduler
  argues for (`reading/sglang.md` §8 anti-pattern). Keep.
- **No Ray-removal and no deeper-Ray.** The readings show Ray's defensible
  scope is exactly what vrl uses it for — placement and actor lifecycle —
  with the data plane elsewhere (`reading/sglang.md` §4: "Ray is used for
  process lifecycle; ZMQ handles communication"). P0-1 moves the *weight*
  data plane to NCCL but keeps Ray for control, which is the slime topology.

## Suggested order

1. P0-2 (KL ref CPU swap) — small, unblocks full-param KL immediately.
2. P0-1 (NCCL weight path) — required before full-param sync frequency hurts.
3. P1-2 (sleep/wake) — removes the per-cycle model reload in colocated runs.
4. P1-1 (non-draining barrier) — biggest steady-state throughput win for
   video; builds on machinery that already exists.
5. P1-3 / P2 items opportunistically.
6. P3 only as a deliberate engine project.
