# SPRINT: Multi-GPU rollout perf — per-worker pipeline + async weight-sync

> **DONE — negative gate result (2026-07-19).** Two- and three-worker L4 runs
> gave every worker exactly two chunks per request. Per-worker between-chunk
> bubble was 0.62-0.76%, and weight sync was at most 3.49% of exact update wall.
> Neither gate fired, so this sprint intentionally adds no rollout optimization.
>
> **Two multi-GPU rollout levers, both hardware-gated (cannot be validated on 1 GPU).**
> A rollout-dedicated GPU wastes any idle. On multi-GPU the denoise already
> data-parallels across workers (near-linear, each card MFU-bound), so the remaining
> idle is: (1) the per-worker between-chunk orchestration bubble, and (2) the
> weight-sync barrier. This sprint documents the design + a **measurement gate** so
> the work is built only where a real run shows the idle is worth it. Grounded in
> this session's code reads (file:line) + `[[project_real_run_profiling]]` (~36%
> rollout idle, ~33% between-chunk orchestration on 1 GPU).

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

## Final gate measurement (2026-07-19)

Both runs used SD3.5 OCR continuous LoRA on L4s, `128x128`, four denoise steps,
`samples_per_chunk=1`, one request at a time, one prompt per update, and four complete
updates. One trainer used physical GPU 0. Round-robin assignment was deliberately exact:

- two workers: four samples -> `2/2` chunks on physical GPUs 1/2;
- three workers: six samples -> `2/2/2` chunks on physical GPUs 1/2/3.

The busy metric is the union of CUPTI kernel `[start,end]` intervals per physical GPU,
clipped to complete two-chunk request spans. Ray exposes every actor's card as local
`cuda:0`; the analysis therefore maps `(PID, local CUDA id)` through Nsight's
`TARGET_INFO_CUDA_DEVICE` table before unioning. Never merge actor-local `deviceId=0`
rows directly.

| workers | physical GPU | kernel busy | kernel idle | between-chunk bubble |
|---:|---:|---:|---:|---:|
| 2 | 1 | 34.10% | 65.90% | 0.739% |
| 2 | 2 | 33.86% | 66.14% | 0.758% |
| 3 | 1 | 33.91% | 66.09% | 0.693% |
| 3 | 2 | 34.43% | 65.57% | 0.730% |
| 3 | 3 | 33.30% | 66.70% | 0.616% |

The large total idle is mostly *inside* the tiny acceptance-scale chunk (prompt encode,
host launch/synchronization, and decode), not between its two worker-local chunks. Lever 1
only targets the latter, which is 26-49x below its 20-30% gate.

`trainer.optimizer_update` encloses collect, replay/backward, optimizer step, and post-step
weight sync, so its NVTX wall is the exact denominator rather than an estimate:

| workers | sync / all update wall | sync / steady update wall | worst update |
|---:|---:|---:|---:|
| 2 | 2.85% | 3.28% | 3.49% |
| 3 | 2.23% | 2.50% | 2.75% |

All updates reported `continuous_weight_sync_barrier_mode=1`: the existing versioned
trainable-state slot path from `SPRINT_shadow_model_weight_sync` was active. This sprint
does not extend it into a new double-buffered transport because even the measured pause is
well below the 10% gate.

Nsight emitted one late `engine.forward_chunk` range without temporally matching kernels
per worker. Each is excluded, along with the otherwise valid unpaired range from that
request; three complete two-chunk request pairs remain on every GPU. The launch logs and
resolved configs still verify all four requests had exactly two assigned chunks per worker.

Evidence:

- `outputs/multigpu_rollout_perf_2w_20260719/rollout_perf_validation.json`
- `outputs/multigpu_rollout_perf_2w_20260719/profile.nsys-rep`
- `outputs/multigpu_rollout_perf_2w_20260719/train/metrics.csv`
- `outputs/multigpu_rollout_perf_3w_20260719/rollout_perf_validation.json`
- `outputs/multigpu_rollout_perf_3w_20260719/profile.nsys-rep`
- `outputs/multigpu_rollout_perf_3w_20260719/train/metrics.csv`

Decision: neither lever fires. Record the negative result, write no optimization code,
and close the sprint.

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

**Candidate cost:** continuous weight sync still pauses new admission while workers install
the payload and return ACKs. The capability fallback also drains in-flight work; the
versioned-slot path active in these runs safely skips that drain. The remaining synchronous
install/ACK pause does not scale away with card count; it is small for LoRA and may be larger
for a future full-parameter workload.

**Fix direction:** overlap the sync with generation — start generating the next group
under the current weights while the new weights stream in, and swap at a version boundary
(the workers already refuse version-mismatched chunks: `execution/worker.py` version
guard). The launcher already enables the safe non-draining versioned-slot path when every
worker supports it. A further transport lever would need a true double-buffered install so
transfer itself overlaps useful generation, with the new version visible only after complete
fleet ACK. The measured gate did not justify building that extension here.

## Non-goals

- Cross-worker copy overlap (different GPUs already parallel) — useless.
- Oversubscribing workers per GPU to hide the bubble — costs 2× memory; Lever 1 is
  cheaper.
- Building either lever speculatively on 1 GPU — cannot validate the win; the whole
  point is the per-worker/ barrier idle that only appears with ≥2 workers.

## Status

DONE — NEGATIVE RESULT. The required two- and three-worker measurements both stayed far
below their decision gates. Lever 1 and the proposed Lever 2 extension were not
implemented. Re-open a new sprint only if a different representative workload crosses a
gate; do not infer that from total within-chunk GPU idle alone.
