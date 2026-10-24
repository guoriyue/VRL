# SPRINT: PT26-11 — CUDA graphs, fusion and kernel tuning for measured hotspots

Status: **planned — P3, conditional: a fresh representative trace identifies a material launch or kernel bottleneck.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

Use `vrl/utils/profiling.py` and external GPU traces. Host stage timers are not exclusive CUDA timings, and the deleted `trace_attribution.py` is not a runnable dependency. Existing [throughput](SPRINT_rollout_reward_throughput_pt26.md) and [Triton](SPRINT_triton_map_and_perf_ledger.md) measurements are historical; previous production CUDA-graph measurements did not justify adoption. Reopen only for a changed measured workload.

## Deliverables

- [ ] Capture an actual production recipe and rank critical path, including reward/service waits, data movement and denoise/replay. Select one hotspot and estimate its maximum end-to-end contribution.
- [ ] For material launch gaps, isolate one fixed-shape CUDA-graph candidate and verify RNG across replays, changing timestep/guidance values, shape bounds, hot weights, park/wake, concurrency and private-pool reserved memory.
- [ ] For a kernel hotspot, compare the existing eager/Inductor/Triton baseline with one justified Helion/fusion candidate. Wan norm/modulation/quant fusion must preserve residual dtype, rounding order and downstream scale contracts; Trainium NKI code is conceptual reference, not a CUDA implementation.
- [ ] If compiler tuning is promising, pin GPU/toolkit/kernel revision and compare fixed-configuration baseline with CompileIQ/ACF or a tuning candidate before joint search. Report tuning wall time separately from kernel latency.
- [ ] Verify the selected implementation actually dispatches, including fallback and training/compile modes; lock dependency revisions. Cover forward, dX/dW where needed, full update and custom-op/activation-checkpoint save policy.

## Acceptance and stop rule

Require explicit numerical/gradient/update and quality gates from PT26-00, cross-shape performance and complete-step gains beyond noise. Record allocated and reserved VRAM separately. Stop if tuning/compile amortization is unfavorable, the production bottleneck is elsewhere or the existing compiler is as fast. No global graph/fusion rollout based on a favorable single shape.

## Boundaries

Keep eager fallback, family interfaces, RNG/parking and weight-install contracts. Kernel-only speedups, autotuning speedups and training speedups are different claims. A profile screenshot or a fallback run does not qualify an implementation.

References: [Helion](https://helionlang.com/index.html), [CompileIQ with Helion](https://pytorch.org/blog/one-search-two-layers-compiler-controls-inside-pytorch-helion-using-nvidia-compileiq/).
