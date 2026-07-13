# READING: CUDA MPS + same-GPU data parallelism — and what it fixes in our profiling

> Original note: 《10 行代码把小模型吞吐提升 200%》 (attributed to Jiaxin Deng,
> LinkedIn Pulse). The original URL was not captured and could not be independently
> recovered on 2026-07-12. The general MPS mechanism is corroborated by
> [Databricks' scaling study](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps),
> but the Higgs/MOSS measurements below remain source notes, not independently
> verified project evidence. Recover the original URL before using those numbers in
> an implementation decision.
> **Why it matters to us:** the article's diagnosis is *exactly* our measured rollout
> pattern (GPU idle despite a full queue), and its fix is a lever we have **never tried**
> and currently **cannot even express** (the `rollout_gpus_per_worker` validation
> in `vrl/ray/resources.py` hard-blocks it).
> It also upgrades our profiling methodology, which is the thing that has misled us
> repeatedly. Read 2026-07-02.

## What the article found

- Migrating a small TTS model (Higgs) H20 → H100, throughput **plateaued** even with
  **saturated request queues** (62/64 running). Device profiling: **SM Active ≈ 29%** —
  the GPU was **idle ~71%**.
- **The lesson:** *"serving throughput plateau ≠ GPU saturation"*, and **"queue is full"
  ≠ "GPU is busy"**. One serving process could not keep the SMs fed.
- **The fix (the "10 lines"):** **CUDA MPS** + run **2–3 replicas of the model on the
  SAME GPU** (same-card data parallelism). MPS makes the processes' kernels run
  **concurrently** instead of being **time-sliced** by the driver.
  ```bash
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER-gpu0/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER-gpu0/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  CUDA_VISIBLE_DEVICES=0 <serve> --port 8801 &
  CUDA_VISIBLE_DEVICES=0 <serve> --port 8802 &
  ```
- **Measured (1×H100):** 1 replica 21.7–22.1 qps → **DP2+MPS 31.5–37.7 (1.4–1.7×)** →
  DP3+MPS 39.9–46.9 (1.8–2.1×). DP3 peaks higher but is less stable; **DP2 is the sweet spot**.
- **Caveats the author is explicit about:**
  1. The single replica must ALREADY be tuned (concurrency, CPU affinity) — otherwise you
     are just papering over a bad baseline.
  2. It **does not save VRAM — it recovers idle compute.** Weights must fit N× in VRAM.
  3. **Counterexample:** MOSS-TTS-Local was already at **~81% SM Active** → **DP gave nothing**
     (compute-bound). **DP only pays when the GPU is under-fed.**
  4. Enabling MPS ≠ clients attached — verify the PIDs appear in the MPS client list.
  5. `--mem-fraction-static` grabs from *currently available* memory → replicas launched
     sequentially get **unequal** KV allocations.

## Why this lands directly on us

Our own numbers ([[project_real_run_profiling]]) are the same shape as the article's:
```
cosmos rollout:  64% GPU-busy / 36% IDLE   ← ~33% is between-sample orchestration (Ray/Python/transfer)
the denoise loop itself: 96-98% GPU-bound  ← the kernels ARE efficient
```
So: **our kernels are fine; our GPU is under-fed.** That is precisely the article's
29%-SM-Active situation, just less extreme. We have been trying to fix it with *code*
(the per-worker chunk-pipeline design, [[SPRINT_multigpu_rollout_perf]]). The article
says there is a **config-only** lever we never tried: **put a second rollout worker on the
same GPU and let MPS interleave them.**

### The subtlety that reconciles this with our "overlap is NEUTRAL" finding
We previously measured (controlled 2-stream micro-benchmark): **two concurrent cosmos-2B
DiT forwards on one GPU = 1.1% ≈ NEUTRAL** — tensor-core-bound work does not overlap.
**That is still true, and it does not contradict this.** The two mechanisms are different:

```
❌ concurrent COMPUTE overlap  — two denoises running at once → serialize (tensor cores saturated). NEUTRAL. (our measurement)
✅ IDLE-GAP FILLING            — while replica A sits in its Python/Ray/copy gap (GPU idle),
                                 replica B's denoise kernels run. The denoises INTERLEAVE into
                                 each other's holes. This is what MPS+DP buys — and it is exactly
                                 the 36% idle we measured.
```
So the expected win is bounded by **the idle fraction (~36%)**, not by concurrency of compute.
Ballpark: filling most of that idle ≈ **up to ~1.3–1.5×** rollout throughput — a config change,
no new code. (The article's 1.4–1.7× came from a *71%* idle, so our smaller idle → smaller win.)

## The blocker in our code (concrete)

We **cannot express this today**. Ray already supports fractional GPUs
(`ray.remote(num_gpus=0.5)` → 2 actors per GPU; `vrl/ray/actor_group.py:52` passes it through),
but VRL hard-blocks it:
```python
# vrl/ray/resources.py:208-213
rollout_gpus_per_worker = float(config.rollout.gpus_per_worker)
if rollout_gpus_per_worker not in {0.0, 1.0}:
    raise ValueError(
        "distributed.resources.rollout.gpus_per_worker currently supports 0 or 1, ...")
```
**The lever = relax this guard to allow fractional values** (e.g. `0.5` with
`rollout.num_workers: 2`), then launch under MPS. The field is already typed `float`.

## Known hazard (be honest)

We have already been burned by same-GPU multi-process in this stack: the
`reward.resident_overlap` experiment (reward model resident on the rollout GPU) **HUNG —
a Ray deadlock**, not an OOM ([[project_real_run_profiling]]). MPS *may* be exactly what
that experiment was missing (it coordinates concurrent execution instead of letting the
driver time-slice), or it may not. **This must be probed, not assumed.**

Second hazard: **memory.** Two replicas = 2× weights + 2× activations. Cosmos-2B at max
`samples_per_chunk` already peaks ~25GB of 32GB → **DP2 will NOT fit at max chunk size**.
DP2 requires *shrinking per-replica chunk size*, which costs per-replica MFU — so the net
win is (idle recovered) − (MFU lost to smaller chunks). **This trade is the whole question
and must be measured, not argued.**

## The profiling upgrade (the real takeaway)

This is the part that answers "my profiling is a big issue". Our profiling has misled us
repeatedly (nsys projection counting async-launched kernels → we read GPU>wall and drew 3
wrong conclusions). The article's method is **cheaper and harder to fool**:

```
1. PRIMARY METRIC = SM Active %  (device-level), NOT throughput, NOT "queue is full",
   NOT nsys projected time.        →  nvidia-smi dmon -s u   /  DCGM  /  ncu
2. THE DECISIVE EXPERIMENT (no code):
      run a SECOND replica on the same GPU.
      aggregate throughput goes UP  → you had recoverable idle (under-fed GPU)
      aggregate throughput flat     → you are genuinely compute-bound; stop optimizing here
3. Only AFTER 1+2 say "under-fed" do you spend engineering on pipelining/prefetch/etc.
```
**Rule to adopt:** never start a rollout perf sprint without an SM-Active number and the
2nd-replica A/B. It would have short-circuited several of our past investigations.

## Proposed next steps (gated, cheap-first)

```
P0  Measure SM Active% during a real cosmos rollout (nvidia-smi dmon -s u), confirm the
    64/36 split at the device level. No code.
P1  2nd-replica A/B WITHOUT touching VRL: launch two independent rollout processes on GPU0
    under MPS (small shape so both fit) and compare aggregate samples/s vs one process.
    → This alone answers "is the 36% idle recoverable?" with no VRL change.
P2  If P1 is positive: relax the rollout_gpus_per_worker guard in vrl/ray/resources.py to allow fractional
    gpus_per_worker, add the MPS launch env, and A/B DP2 (2 rollout workers × half chunk)
    vs DP1 (1 worker × max chunk). Win = idle recovered − MFU lost to smaller chunks.
P3  If P1 is negative (throughput flat): we are compute-bound → the 36% "idle" is NOT
    reclaimable this way; fall back to the per-worker chunk-pipeline design
    ([[SPRINT_multigpu_rollout_perf]]) or accept the floor.
```

## Non-goals / what NOT to conclude

- **Do not** conclude MPS "makes denoises overlap" — it does not (our 2-stream measurement
  stands). It fills *idle gaps*.
- **Do not** deploy DP2 without the memory trade measured — a smaller per-replica chunk can
  cost more MFU than the idle it recovers.
- **Do not** skip P0/P1 and jump to code. That is the exact mistake this article is warning about.
