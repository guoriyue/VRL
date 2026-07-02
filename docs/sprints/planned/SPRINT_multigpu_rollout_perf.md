# SPRINT: Multi-GPU rollout perf — per-worker pipeline + async weight-sync

> **Two multi-GPU rollout levers, both hardware-gated (cannot be validated on 1 GPU).**
> A rollout-dedicated GPU wastes any idle. On multi-GPU the denoise already
> data-parallels across workers (near-linear, each card MFU-bound), so the remaining
> idle is: (1) the per-worker between-chunk orchestration bubble, and (2) the
> weight-sync barrier. This sprint documents the design + a **measurement gate** so
> the work is built only where a real run shows the idle is worth it. Grounded in
> this session's code reads (file:line) + `[[project_real_run_profiling]]` (~36%
> rollout idle, ~33% between-chunk orchestration on 1 GPU). Status: **DESIGN ONLY —
> build when a multi-GPU box is available and the gate below fires.**

## Why this is not "class-ify" or single-GPU work

- Single-GPU levers are DONE: max `samples_per_chunk` (~1.95x) + the chunk-pipeline
  (single-worker, `executor.py` `len(self.workers)==1` gate) hide the 1-GPU copy.
- Multi-GPU denoise scaling is DONE: `run_actor_jobs` distributes chunks across
  workers = data-parallel; each card is MFU-bound. Adding cards ≈ near-linear.
- What's LEFT on multi-GPU is only the two idle sources below. **Cross-worker overlap
  (worker A's copy ∥ worker B's denoise) is a NON-GOAL — different GPUs already run in
  parallel; overlapping them shortens no worker's own critical path.**

## The gate — measure per-worker GPU-idle FIRST, build only if it fires

Build a lever only if a real multi-GPU run shows the idle it targets is material.

```
# Per-worker GPU-busy% during a rollout phase (one sample per GPU, 1s cadence):
nvidia-smi dmon -s u -d 1 -o T > dmon.log        # while a multi-GPU rollout runs
# → per-GPU sm% column. A rollout-worker GPU sitting well below ~90% busy DURING
#   generation is reclaimable idle. Cross-check which gap it is:

VRL_PROFILE=1 nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true \
    python -m vrl.scripts.train ...              # fork flag = captures Ray workers
# → judge GPU-idle by kernel-interval UNION over wall (NOT nsys projection).
```

Gate thresholds (from the 1-GPU baseline, re-measure on the real box):
- **Per-worker between-chunk idle ≳ 20–30%** (expected when `samples_per_chunk`=1 /
  chunks-per-worker > 1) → build **Lever 1**.
- **Weight-sync barrier ≳ 10% of step wall** (bigger for full-param than LoRA) →
  build **Lever 2**.
- Idle ~few % (max `samples_per_chunk`, ~1 chunk/worker) → **do nothing**; it is the
  MFU-bound floor.

## Lever 1 — per-worker pipeline (ungate the chunk-pipeline)

**Problem (verified):** the chunk-pipeline that hides chunk N's teardown behind chunk
N+1's denoise is gated to a single worker — `vrl/generation/ray/executor.py`:
`if self.pipelined and len(self.workers) == 1: ...`. With ≥2 workers the code falls to
per-chunk dispatch (`run_actor_jobs`), so **each worker runs its chunk subset serially
with unhidden between-chunk bubbles** (~33% orchestration gap when chunks-per-worker > 1).

**Fix:** let each worker pipeline ITS OWN chunk subset (not the whole request on one
worker). The mechanism already exists and is bit-exact tested — it just needs to run
per-worker in the multi-worker path.
```
① planner:  in multi-worker, assign each worker its chunk subset as a batch
            (not one-chunk-per-dispatch)
② worker:   run the existing forward_chunks_pipelined / execute_request_pipelined
            (vrl/generation/diffusion/pipeline.py) on the worker's subset, reusing the
            already-tested version-safety path (execution/worker.py execute_request_pipelined)
③ executor: replace the len(workers)==1 gate with "per-worker when chunks-per-worker > 1"
verify:     each rollout card's GPU-busy% rises (bubble hidden); gather bit-exact vs serial
```
No extra memory (worker-internal pipeline depth 1, holds ~1 extra trajectory), unlike
oversubscription (2 workers/GPU = 2× model+activations).

## Lever 2 — async weight-sync (hide the barrier)

**Problem (verified):** the continuous schedule's weight sync is a barrier —
`vrl/rollouts/orchestration/continuous/schedule.py` pause→drain→sync→resume. Every step,
all rollout workers pause while the trainer broadcasts new weights. This cost does NOT
scale with card count (it's a fixed per-step stall); small for LoRA, large for full-param.

**Fix direction:** overlap the sync with generation — start generating the next group
under the current weights while the new weights stream in, and swap at a version boundary
(the workers already refuse version-mismatched chunks: `execution/worker.py` version
guard). Requires: non-draining sync path (the launcher already computes
`supports_non_draining_weight_sync` when every worker keeps versioned slots —
`ray/launcher.py`), plus a double-buffered weight slot so in-flight chunks finish on the
old version. Design-only until measured; this is the larger multi-GPU win when
full-param sync dominates.

## Non-goals

- Cross-worker copy overlap (different GPUs already parallel) — useless.
- Oversubscribing workers per GPU to hide the bubble — costs 2× memory; Lever 1 is
  cheaper.
- Building either lever speculatively on 1 GPU — cannot validate the win; the whole
  point is the per-worker/ barrier idle that only appears with ≥2 workers.

## Status

DESIGN + GATE only. No code — both levers need a multi-GPU box to validate, and the
gate decides whether each is worth building there. Single-GPU perf is already at the
MFU-bound floor.
