# SPRINT: Best rollout system — TeaCache port + selective vLLM-omni adoption

**Branch:** `rollout-vllm-teacache` (all work here; commit locally, do NOT push).
**Owner:** autonomous overnight run (hourly cron). **Started:** 2026-06-20.
**Machine:** single RTX 5090 (Blackwell sm_120, 32GB). cosmos rollout is
compute-bound (SM 100% / MEM ~20%).

This file is the **source of truth** for the autonomous loop. Each cron firing:
read "Next actions", do the next increment, verify (lint + pytest + profile when
it fits), update the journal, commit. Never push. Never delete anything from the
machine. Resolve choices yourself; only stop on a hard blocker (missing
credentials / a real crash you cannot fix) and record it under "Blockers".

---

## Goal (scoped honestly)

Produce the best rollout system on this box. NOT "replace the whole rollout with
vLLM" — that is the wrong target. The right axis is **kernel/algorithm (import &
enhance) vs execution model (only adopt when its value applies)**:

- **Diffusion rollout (cosmos / sd3.5 / wan)** is compute-bound, no KV cache, no
  continuous batching. vLLM's *engine* value does not apply. The wins are
  **importable**: fp8 (done) + **TeaCache** + (later) VAE parallel. → KEEP our
  denoise loop, import the pieces.
- **AR rollout (janus / nextstep)** is autoregressive decode, KV-cache, variable
  length — vLLM's home turf (paged attention + continuous batching). Our
  `vrl/nn/.../vllm_paged.py` + `paged_attention_helpers.py` is reinventing the
  engine; continuous batching is the un-importable part. → AR is the ONLY place
  "replace with vLLM-omni" is justified, and only if a measured throughput gap
  warrants it.

Model coverage in vLLM-omni 0.18.0 (verified): `sd3` ✅, `wan2_2` ✅ (not 2.1),
`nextstep_1_1` (as diffusion-flow, not our AR path), **cosmos ❌** (no dir; only
the generic `DiffusersPipelineLoader`, EDM/conditioning unverified), **janus ❌**.

---

## Phases (ROI order)

### P0 — TeaCache port into the diffusion rollout  ✅ CORE DONE (2026-06-20)
Skip the transformer forward on low-change denoise steps, reuse cached
`noise_pred`. Highest ROI, importable, low risk, default-OFF (baseline exact).

- `vrl/generation/diffusion/teacache.py` — `TeaCacheConfig` + `TeaCacheState`
  (rel-L1 accumulator, warmup/last-step force-run, skip-ratio counters) +
  `teacache_signal`. v1 signal = rel-L1 of input latents.
- Wired: `layout.py` (parse `sampling.teacache`), `executor.py` (gate
  `model.forward_step` in `run_denoise_steps`, emit `teacache_*` counters).
- Tests: `tests/generation/diffusion/test_teacache.py` (9 pass).
- **RL note:** TeaCache adds rollout↔replay drift (skipped steps reuse an
  approximate noise_pred) — same class as fp8; rides the same drift-guard / TIS.

### P0.1 — Profile TeaCache speed + drift  ⏳ NEXT
- Profile cosmos predict2.5 with `--precision bf16` and TeaCache on
  (threshold sweep 0.1/0.15/0.25) vs off: s/step, skip ratio, kernel buckets.
  Extend `vrl/scripts/perf/generation_bottleneck_profile.py` with a `--teacache`
  knob (it drives `model.forward_step` directly, so add the same skip machine, or
  better: profile through the executor `run_denoise_steps`).
- Quantify the rollout↔replay drift TeaCache induces (reuse the fp8 drift probe:
  rollout logprob with TeaCache vs exact replay logprob) → go/no-go per threshold.
- Record the speed/drift Pareto here.

### P0.2 — TeaCache accuracy refinement (per-family modulated signal)  ⬜
- v1 uses raw-latent rel-L1. Add per-family "timestep-modulated input" extractor
  (vLLM-omni's `extractors.py` approach) + optional rescale polynomial as new
  `signal` kinds. Only if P0.1 shows the latent signal skips too aggressively /
  inaccurately.

### P1 — fp8 + TeaCache compose  ⬜
- Confirm fp8 rollout + TeaCache stack (both rollout-only, both drift-corrected):
  profile fp8+TeaCache, confirm no crash + additive speedup, combined drift still
  under advantage signal.

### P2 — vLLM-omni diffusion spike (sd3, the clean native model)  ⬜
- Load sd3 through vLLM-omni `DiffusersPipelineLoader`, run one forward, compare
  `noise_pred` consistency + throughput vs our native sd3.5 path. Decide
  data-driven whether diffusion should EVER move to vLLM-omni (prediction: no for
  cosmos; sd3 maybe for TeaCache+VAE-parallel bundle). Document, do not migrate.

### P3 — AR continuous-batching evaluation (the real "replace" candidate)  ⬜
- Measure our `nextstep`/`janus` paged decode throughput vs vLLM-omni AR
  continuous batching on a batch of rollout prompts. This is where replacing the
  engine pays. If the gap is large, scope an AR-engine adoption sprint; else keep
  ours. Spike + measure only — no migration this run.

---

## Journal (most recent first)

- **2026-06-20 hN** — P0 core landed: TeaCache module + layout/executor wiring +
  9 unit tests (green), lint clean, default-OFF. Branch `rollout-vllm-teacache`
  cut from `fp8-rollout-precision-tis`. Profiler already has `--precision
  {fp32,fp16,bf16,fp8}` + `--fp8-recipe`. Earlier this session: cosmos precision
  ladder profiled (fp32 OOM; fp8 rowwise 1.15 s/step = 1.1x vs bf16 1.29; vLLM
  blockwise 5.97 s/step = a TRAP, reverted as a recommendation).

## Next actions (cron picks the top unchecked)
1. P0.1: add TeaCache to the profiler (or profile via the executor path) and run
   the cosmos bf16 ± TeaCache threshold sweep; dump the table into this file.
2. P0.1: measure TeaCache rollout↔replay drift; go/no-go per threshold.
3. P1: fp8 + TeaCache combined profile.
4. P2: sd3 vLLM-omni `DiffusersPipelineLoader` forward-consistency spike.
5. P3: AR paged-decode vs vLLM-omni continuous-batching throughput spike.

## Blockers
(none)

## Non-goals
- Pushing the branch. Deleting anything. Migrating diffusion wholesale to vLLM.
- Touching the trainer/replay precision (rollout-only changes).
