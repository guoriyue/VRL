# SPRINT: Diffusion Within-Chunk GPU Idle

**Status:** DONE — closed 2026-08-16 with a **documented negative result**. No
optimization code was written.

The `~65%` within-chunk idle **does not reproduce at a production-like shape**.
It is an artifact of the tiny acceptance workload, exactly as the Origin section
suspected. See "Measured result" below.

## Origin

`SPRINT_multigpu_rollout_perf` closed with a negative result for both proposed
multi-worker levers:

- between-chunk bubble was only `0.616%–0.758%`, and
- steady-state weight sync was only `2.50%–3.28%` of update wall time
  (`3.49%` maximum).

However, kernel-interval union over the same complete rollout request windows
showed only `33.30%–34.43%` GPU busy, leaving `65.57%–66.70%` idle. Since the
chunk-boundary bubble was below 1%, the next question is where the idle time
inside each chunk comes from.

This observation is currently limited to the SD3.5 OCR acceptance workload:
`128x128`, four denoise steps, and one sample per chunk. Historical traces at
larger, production-like shapes showed `95%–99%` kernel busy in dense denoise
segments. This sprint must therefore establish whether the idle reproduces at
a representative shape before treating it as a general rollout bottleneck.

Source evidence:

- `outputs/multigpu_rollout_perf_2w_20260719/rollout_perf_validation.json`
- `outputs/multigpu_rollout_perf_3w_20260719/rollout_perf_validation.json`
- `docs/sprints/done/SPRINT_multigpu_rollout_perf.md`
- `docs/sprints/info/SPRINT_rollout_performance.md`

## Goal

Attribute kernel-idle time within a rollout chunk to an exact stage and decide
whether a narrow optimization is justified.

The sprint is measurement-first. It must close with a documented negative
result if no stage crosses a decision gate.

## Measurement Boundary

The primary interval is one steady-state `engine.forward_chunk` occurrence on
one physical rollout GPU. The interval includes:

1. prompt encoding,
2. latent and scheduler preparation,
3. the denoise loop,
4. VAE decode when enabled, and
5. output and trajectory assembly performed by the chunk executor.

It excludes:

- `RolloutWorker._to_cpu(output)`,
- Ray serialization and driver gather,
- reward computation and training,
- weight synchronization, and
- time between consecutive `engine.forward_chunk` occurrences.

Those exclusions are deliberate. Worker finalization and between-chunk overlap
already belong to separate completed or planned sprints.

GPU busy is the union of CUDA kernel intervals for the matched process and
physical GPU, clipped to the chosen NVTX interval. GPU idle is the complement
inside the same interval. Kernel launch count, launch-gap distribution, CUDA
API time, and stage wall time are supporting measurements; none may replace
kernel-interval union.

## Questions

Answer these in order:

1. Does the low busy ratio persist after warmup and without an unmatched or
   truncated terminal NVTX occurrence?
2. Does it reproduce at a representative production-like resolution, denoise
   step count, precision, and model configuration?
3. Which typed stage owns the idle: prompt encode, prepare, denoise, decode, or
   output/trajectory assembly?
4. Within denoise, is the issue kernel gaps, or are kernels continuously active
   but individually underutilizing the GPU?
5. Is the gap launch-bound and removed by the existing compiled path?
6. Is request-local chunk batching sufficient to amortize it, without a new
   cross-request scheduler?

## Profiling Matrix

Run on one fixed GPU type with fixed seeds and identical model/precision:

| Dimension | Required points |
|---|---|
| Shape | current `128x128`, 4-step acceptance case; one representative production-like recipe that fits the same GPU |
| Samples per chunk | `1`, `2`, `4`, or the largest OOM-safe subset of those points |
| Execution | current eager/default path; existing compiled path when supported |
| Repetitions | at least two measured steady-state chunks after warmup per point |

For the production-like case, use an existing runnable recipe rather than a
synthetic kernel microbenchmark. Record any shape or step-count reduction
required by memory limits.

Profiled wall time must be compared with an otherwise equivalent unprofiled
run so profiler overhead is visible.

## Required Attribution

For each measured chunk, report:

- total chunk wall time,
- kernel-union busy time and idle time,
- busy and idle percentages,
- wall, kernel union, and idle for each typed stage,
- launch count and the `p50`, `p90`, `p99`, and maximum inter-kernel gap,
- total time in blocking or synchronization CUDA APIs,
- peak allocated and reserved GPU memory, and
- output and trajectory equivalence against the current executor.

The stage decomposition must cover the chunk interval without overlap or
double counting. Any unattributed remainder must be reported explicitly.

Within the denoise stage, separately report:

- `generation.denoise_forward`,
- scheduler step,
- latent snapshot/write,
- trajectory buffer write, and
- remaining denoise-loop overhead.

If denoise kernels are nearly continuous but have low Compute(SM), use Nsight
Compute on a representative hot kernel and compare it with an appropriate
same-machine baseline. Do not call low per-kernel occupancy or throughput
"GPU idle."

## Measured result (2026-08-16, RTX 5090, bf16)

Real `nsys` capture of `engine.forward_chunk` on the production
`GenericDiffusionBatchExecutor` path, analysed with this repo's own
kernel-interval-union primitives (`vrl/utils/nsys_report.py`). Two steady-state
chunks per shape; warmup chunks emit a different NVTX name and are excluded by
construction.

| shape | chunk wall (ms) | kernel busy | idle |
|---|---|---|---|
| 128x128, 4 steps, 1 sample (acceptance) | 224 / 230 | **47.2% / 48.2%** | ~52% |
| 512x512, 10 steps, 1 sample | 675 / 710 | **80.7% / 81.4%** | ~19% |
| 512x512, 10 steps, 2 samples | 1080 / 1089 | **87.3% / 87.1%** | ~13% |
| 512x512, 10 steps, 4 samples | 1941 / 1951 | **92.6% / 91.8%** | ~8% |

### Stage attribution at the production shape (512px / 10 steps / 1 sample)

| stage | % of chunk wall | busy within stage | idle as % of chunk wall |
|---|---|---|---|
| `generation.denoise_step` | 92.81 | 81.3 | 17.39 |
| `generation.prompt_encode` | 6.26 | 76.3 | 1.49 |
| `generation.decode_latents` | 0.43 | 77.1 | 0.10 |
| `generation.prepare_sampling` | 0.23 | 78.3 | 0.05 |

Latent snapshot/write and trajectory buffer write together are `<0.15%` of chunk
wall. **No stage owns a reclaimable `10%`.**

### Mechanism

Launch-bound, not a synchronization point. At the tiny shape the median kernel
is `1.89us` while the median inter-kernel gap is `2.21us` — kernels are shorter
than the gap between them. At the production shape the median kernel grows to
`3.26us` and the median gap falls to `1.18us`. Blocking/sync CUDA API on an idle
GPU is only `2.1-3.8%` of tiny-shape chunk wall, so synchronization is not the
cause.

### Gate outcomes

- **Gate A (stage-local gaps): NOT crossed.** `denoise_step` is 92.8% of wall but
  already 81-83% busy; its idle is spread across ~50,000 sub-2us gaps, not a
  removable sync point.
- **Gate B (compiled path): NOT EVALUATED.** `torch_compile` was forced off for
  every run so all points share one execution path. This question is genuinely
  still open.
- **Gate C (request-local batching): crossed on wall time.** Per-sample wall at
  512px/10 steps: 728-763ms (1) -> 593-598ms (2, **-20%**) -> 529-530ms (4,
  **-27%**). Absolute idle stays flat at ~110-160ms per chunk regardless of chunk
  size, i.e. a fixed per-chunk cost that larger chunks amortize. Peak allocated
  barely moves (16746 -> 16784 MiB; the resident T5-XXL dominates). **This needs
  no new code:** `samples_per_generation_batch` already exists and
  `online_grpo_ocr` already sets it to 16.
- **Negative exit: satisfied.** Representative-shape busy is 81-93%, and at 4
  samples/chunk it clears the 85% bar outright.

### Measurement hazards worth remembering

Two effects would each have produced a wrong answer:

1. **GPU contention.** The card was shared with other sessions. The same code and
   shape ran 1225ms/chunk under contention vs 728ms on a clear card — a 40%
   swing — and the contended trace showed 78% of its idle in 50 gaps of 1-4.4ms,
   a time-slice signature rather than a VRL bubble. All numbers above were taken
   with the GPU verified idle (0% util, ~31GB free).
2. **Profiler overhead is shape-dependent.** Interleaved A/B: tiny shape
   155-163ms unprofiled -> 212-230ms profiled (~40% overhead); production shape
   739-816ms unprofiled -> 697-770ms profiled (no overhead). Per-launch CUPTI
   cost dominates only when kernels are ~2us — itself independent evidence that
   the tiny shape is launch-bound. Corrected for this, the tiny shape's true
   unprofiled busy is ~68%, not 47%.

### Caveats on this result

- **bf16, not fp32.** Free memory did not permit fp32 SD3.5. The prior
  `65.57-66.70%` evidence is likely fp32. bf16 halves memory traffic and roughly
  doubles kernel duration, which *helps* busy — so the production-shape number is
  plausibly optimistic. The tiny-vs-production **contrast** is measured at
  identical precision, so the gate answer stands; the absolute production number
  should be re-confirmed in fp32 on a free card.
- **Reached 4 samples/chunk, not 16.** The trend is monotone but 8 and 16 were
  not measured.
- **Single in-process executor, no Ray** — matching the sprint's stated boundary.

## Decision Gates

Apply the gates only after the representative-shape run is available.

### Gate A: Stage-local host or launch gaps

Implement a narrow fix only if one stage:

- is at least `10%` of chunk wall time,
- is at least `20%` kernel-idle within its own envelope, and
- explains at least `10%` of total chunk wall time as reclaimable idle in two
  repeated steady-state measurements.

The implementation must target the measured stage. Examples include extending
existing compile coverage or removing a measured synchronization point. Do not
add a general scheduler for a local stage gap.

### Gate B: Existing compiled path

Prefer the existing compiled path when it:

- reduces chunk wall time by at least `10%`,
- materially reduces launch count or kernel gaps,
- preserves output and trajectory equivalence, and
- does not materially increase peak memory or warmup cost at the intended
  rollout lifetime.

If compile already resolves the issue, fix configuration or compile coverage
instead of creating a second execution architecture.

### Gate C: Request-local batch amortization

Adjust request-local chunk sizing only when a larger chunk:

- improves steady-state wall time per sample by at least `10%`,
- fits the declared memory budget without abnormal OOM splitting, and
- leaves ordering and OOM-split semantics unchanged.

The previously negative cross-request step-batching result remains in force.
Reopen that design only with new same-shape evidence that ordinary
request-local batching cannot capture and with multiple simultaneously ready
requests demonstrated in the production workload.

### Negative exit

Close without optimization code when any of the following holds:

- representative-shape kernel busy is at least `85%`, showing that the issue is
  confined to the tiny acceptance workload;
- no single stage explains at least `10%` reclaimable chunk wall time;
- gains disappear outside profiler runs or after warmup; or
- the trace shows high kernel busy but low per-kernel utilization, which belongs
  in a separate kernel/precision optimization sprint.

## Correctness and Resource Invariants

Any accepted implementation must preserve:

- serial bit-exact outputs and gathered trajectories,
- policy version and weight-version visibility,
- sample and chunk ordering,
- deterministic seeds and scheduler behavior,
- current OOM split and retry behavior,
- prompt/conditioning alignment, and
- cancellation and exception propagation.

Peak allocated and reserved memory must not increase materially. A proposed
lever that needs a durable extra model, latent, or trajectory-sized buffer
requires a separate explicit memory decision.

## Non-Goals

- Per-worker chunk pipelines or between-chunk overlap.
- Non-draining asynchronous weight synchronization.
- Worker `_to_cpu`, Ray serialization, or driver gather optimization.
- Reintroducing the deleted stage-queue architecture.
- A cross-request step scheduler without new evidence crossing Gate C.
- Treating tiny-workload idle as proof of a production bottleneck.
- Treating low Compute(SM) during continuously active kernels as GPU idle.

## Deliverables

1. A machine-readable report containing the full profiling matrix and
   per-occurrence stage attribution.
2. Trace artifacts and the exact command/config for each retained point.
3. An update to this sprint with the gate decision and evidence paths.
4. Optimization code and focused tests only if a gate is crossed.
5. A negative conclusion with no optimization code if all gates remain closed.

## Acceptance Criteria

- Every reported interval is matched to one rollout process and physical GPU.
- Warmup and incomplete terminal occurrences are excluded explicitly.
- Stage wall accounting reconciles with total chunk wall time.
- Kernel busy uses interval union rather than summed kernel duration.
- The acceptance workload and a representative production-like workload are
  both measured.
- At least two steady-state occurrences support each decision-critical result.
- Profiled and unprofiled wall times are compared.
- The selected gate is supported by saved artifacts and reproducible commands.
- No speculative optimization is merged when no gate is crossed.

## Related Work

- `docs/sprints/done/SPRINT_multigpu_rollout_perf.md`
- `docs/sprints/parked/SPRINT_diffusion_rollout_stage_pipeline.md`
- `docs/sprints/parked/SPRINT_diffusion_stepwise_batching_probe.md`
- `docs/sprints/parked/SPRINT_cross_request_step_scheduler.md`
- `docs/sprints/planned/SPRINT_rollout_finalize_overlap_ga.md`
- `docs/sprints/info/SPRINT_rollout_performance.md`

