# DECISION: GPU saturation evidence and colocation policy

Status: **DONE / accepted, 2026-07-12**. This is the canonical product decision. The
measurement archive remains in
`docs/sprints/info/SPRINT_gpu_saturation_and_mps_colocation.md`.

## Decision

VRL will not productize CUDA MPS, fractional rollout GPU reservations, or
multiple resident rollout replicas on one physical GPU.

The supported scheduling matrix is:

| Trainer and rollout placement | Schedule | GPU ownership |
| --- | --- | --- |
| Same physical GPU | `strict_on_policy` | Full time-shared phase lease: park trainer, run rollout, release rollout, restore trainer |
| Disjoint physical GPUs | `strict_on_policy` | Resident is allowed; policy execution remains serial |
| Disjoint physical GPUs | `continuous` | Resident generation and training with real wall-clock overlap |
| Same physical GPU | `continuous` | Rejected before actor launch |

Keep the rollout `gpus_per_worker` validation at exactly `0` or `1` in
`vrl/ray/resources.py`. Do not add an MPS daemon to a trainer launcher, restore
role-level `memory_fraction`, or reintroduce `require_separate_gpus` as an escape
hatch. A remote transport is also not proof of GPU isolation; any future overlap
must be gated by resolved local ownership plus an explicit, fail-closed service
capability for accelerators outside VRL's resource plan.

This keeps two explicit scheduling semantics instead of hiding a phase-serial
branch inside `continuous`:

- one shared GPU means capacity sharing over time;
- disjoint GPUs are required for concurrent tensor-heavy execution.

## Evidence

### Real homogeneous rollout

Two SD3.5 rollout forwards on one RTX 5090 produced only `1.03x` aggregate
throughput at the production chunk size (`8`). Reducing the chunk size to `2`
raised the result to `1.11x`, but that is not the production operating point.
The synthetic positive control reached `1.45x` for a small GEMM, while an already
saturated large GEMM remained at `1.01x`. The synthetic result therefore does not
transfer to the real rollout.

### Corrected heterogeneous arithmetic

The archived DiT plus VAE table originally reported `1.14x` normalized effective
work under MPS. That arithmetic was incorrect. Normalized effective work is the
sum of each colocated rate divided by its solo rate:

```text
no MPS = 1.733 / 3.765 + 2.549 / 5.289 = 0.942x
MPS    = 0.902 / 3.765 + 4.199 / 5.289 = 1.034x
```

The corrected aggregate result is therefore approximately `1.03x`, not `1.14x`.
The critical-path regression remains real:

```text
DiT slowdown = 3.765 / 0.902 = 4.17x
```

Even a small aggregate gain would not justify making the denoise critical path
more than four times slower.

### MPS priority is not a critical-path guarantee

NVIDIA MPS exposes client-priority controls, but the priority is a driver hint
that may be ignored or overridden. It is not hard QoS, preemption, or a latency
SLA. The product decision is therefore not based on the inaccurate claim that
MPS has no priority API; it is based on the absence of an enforceable scheduling
guarantee plus the measured critical-path regression.

## Correct interpretation of profiling signals

Low SM occupancy is not evidence of unused throughput. Tensor-core GEMMs often
have low occupancy because registers and shared memory limit resident warps. In
the real MPS run, occupancy increased while throughput improved only `3%`.

Use each measurement only for the question it answers:

- NVML/GPM utilization, occupancy, tensor activity, and DRAM activity explain
  what ran; none proves recoverable throughput by itself.
- Batch scaling is a useful screening measurement for amortization at the tested
  shapes. It is not a decisive saturation test because it changes kernel shape,
  activation pressure, and batching overhead together.
- A real-workload co-run A/B directly tests whether another workload can recover
  aggregate throughput on that device.
- Nsight Compute tensor-pipe SOL compared with a same-machine square-GEMM
  baseline tests kernel saturation. It answers a different question from the
  real-workload co-run.
- Kernel-union busy time from `vrl/utils/nsys_report.py` identifies timeline gaps.
  Process-local pipelining or resident actors, not MPS, are the relevant response
  to host-orchestration gaps.

## Evidence boundary

The one-shot MPS and single-GPU overlap harnesses are not retained as production
tools. Their answers are recorded in this decision and the measurement archive;
the disposable scripts and tests are removed under the repository's one-shot
artifact lifecycle rule. The repository also does not retain the standalone real
SD3.5/heterogeneous benchmark harnesses, raw samples, or a complete run manifest.
The exact numbers are historical decision evidence, not a currently reproducible
performance acceptance test.

This boundary does not weaken the current product decision: the reported best
production-shape result is too small, the heterogeneous arithmetic correction
makes the result weaker, and the critical-path regression is large. It does mean
future hardware or workload claims must start with a new, explicitly named
one-shot probe and a provenance-rich real-workload A/B. Record the answer in a
decision document, then remove the probe when that question is closed.

## Reopening gate

Do not reopen the product path for a synthetic result. A future proposal must:

1. record GPU UUID and model, driver, CUDA/Torch versions, model/config revision,
   and Git commit;
2. verify the exact benchmark client PIDs are attached to MPS and clean up the
   per-user daemon afterward;
3. compare the production baseline and proposal over repeated steady-state runs;
4. show at least `1.10x` end-to-end throughput with a positive lower confidence
   bound;
5. keep critical-path p95 regression within `5%` and preserve safe memory
   headroom; and
6. beat the existing single-GPU phase-lease design without fractional GPU or
   launcher-owned daemon state.

Kill the proposal if any of these gates fails.

## Implementation cleanup

The repository now carries only the execution paths that have a production
consumer:

- The speculative physical-stage contract, serial runner, Ray stage worker, and
  Ray stage runner were deleted. They had no config, launcher, runtime, registry,
  or dotted-string consumer, and their diffusion adapter duplicated
  `DiffusionExecutor.forward_chunk_plan`.
- Test-only `produce_fn` / `teardown_fn` hooks, worker property forwards, derived
  reward-overlap forwards, and flat Ray resource mirrors were deleted. Tests now
  exercise the canonical executor, reward primitives, and
  `ResolvedDistributedResources` directly.
- One-call timing, prompt-key, shutdown, queue-drop, config-parser, and placement
  helpers were merged back into the decisions they had split. Their names did not
  represent independent protocol, ownership, or algorithm boundaries.
- Dead result extras and duplicate stats fields were removed. Runtime-debug data
  remains under the versioned `runtime_debug` payload, while chunk memory drift
  remains log provenance rather than a second output schema.

The following thin boundaries stay deliberately:

- `forward_chunks_pipelined` owns the CUDA event/copy-stream lifetime algorithm;
- Ray runtime/worker methods are actor RPC adapters, including string-dispatched
  methods;
- reward client, server, and protocol modules separate transport, service
  ownership, and wire contracts;
- `StatsSink` and performance CLI facades are public instrumentation boundaries;
- `METRICS`, wire names, environment-variable names, and typed sentinels remain
  module constants because they define real schemas or protocol boundaries.

Non-goals: do not flatten actor, HTTP, callback, lock, or CUDA ownership seams for
line-count reduction; do not invent a shared `execute(payload)` model service;
and do not reintroduce a physical stage runtime until its profiling gate passes.
Cross-family consistency is more valuable than removing those justified facades.

## What stays unchanged

- `vrl/ray/resources.py` remains the resource-policy boundary and keeps the `0/1`
  rollout GPU guard.
- `vrl/rollouts/orchestration/strict_on_policy.py` remains the explicit shared-GPU
  phase-handoff schedule.
- `vrl/rollouts/orchestration/continuous/schedule.py` remains the disjoint-resource
  continuous schedule and retains its fail-fast lifecycle checks.
- `vrl/scripts/perf/gpm_sampler.py` remains a thin NVML instrumentation adapter.
  Its module-level `METRICS` table is a real metric/CSV schema boundary and should
  not be flattened or duplicated.
- `vrl/scripts/perf/nsys_gpu_busy.py` remains a thin CLI facade over the shared
  interval analysis in `vrl/utils/nsys_report.py`.

These thin modules are kept for protocol and instrumentation boundaries, not to
minimize line count. This decision does not authorize a broader profiling-tool
rewrite or cleanup of unrelated historical sprint artifacts.

## References

- Measurement archive: `docs/sprints/info/SPRINT_gpu_saturation_and_mps_colocation.md`
- Superseded reading note: `docs/sprints/reading/SPRINT_mps_same_gpu_dp.md`
- Phase-lease and continuous design: `docs/sprints/SPRINT_miles_phase_lease_and_one_continuous.md`
- Resource guard: `vrl/ray/resources.py`
- Schedule topology validation: `vrl/rollouts/orchestration/schedule.py`
- External reward isolation contract: `docs/sprints/done/SPRINT_reward_service.md`
- NVIDIA MPS documentation: <https://docs.nvidia.com/deploy/mps/>
