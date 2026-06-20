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

### P0.1 — Profile TeaCache speed + drift  ✅ DONE (with a correction)
- ✅ **DRIFT measured (gating) — `teacache_drift_probe.py`, real cosmos denoise,
  two-pass rollout-with-teacache vs exact replay logprob:**

  | config | skip% | ratio_dev_mean | ratio_dev_max | mismatch_kl |
  |---|---|---|---|---|
  | baseline (no teacache) | 0% | 0.0000% | 0.0000% | 0.00000 |
  | teacache(thr=0.1) | 5% | 0.0018% | 0.036% | 0.00002 |
  | teacache(thr=0.15) | 5% | 0.0018% | 0.036% | 0.00002 |
  | teacache(thr=0.25) | 5% | 0.022% | 0.43% | 0.00022 |

  → **drift is SAFE at every tested threshold** (max 0.43% « fp8's 1% bar).
  baseline=0 validates the two-pass logprob math.

- 🔴 **CORRECTION — the earlier "2.3x / 50% skip" profiler number was a measurement
  artifact, NOT real.** `generation_bottleneck_profile.py --teacache` reuses ONE
  state and steps `idx % num_steps`, so consecutive profiled "steps" are nearly
  identical → the skip machine fires far too often. On a REAL 20-step denoise
  (fresh latents, large early-step change) the v1 raw-latent signal skips only
  **~5%** at thr 0.1–0.25 → real speedup is **~5%, not 2.3x**. The profiler
  `--teacache` knob is therefore drift-blind AND skip-inflating; the drift probe /
  executor path is authoritative. (Left the knob for quick smoke, doc-flagged.)

- ⬜ Threshold > 0.25 buys more skips but the latent signal makes drift climb fast
  for little skip gain — the real lever is the signal, not the threshold (P0.2).

### P0.2 — Per-family timestep-modulated signal  ❌ RESOLVED — NOT WORTH BUILDING
- `teacache_drift_probe.py --diagnose` (exact cosmos denoise, rel-L1 between
  CONSECUTIVE exact noise_preds = the model-intrinsic skip ceiling):

  | criterion | skippable |
  |---|---|
  | consecutive noise_pred relL1 < 1% | 0/20 |
  | < 2% | 0/20 |
  | < 5% | 0/20 |
  | < 10% | **0/20** |

  Consecutive noise_preds move **34%–138% every step** (1.39 early → 0.34 last).
- **→ STRUCTURAL, not a signal problem.** cosmos's 20-step EDM schedule has ZERO
  step-to-step redundancy; a better signal cannot find redundancy that does not
  exist. The h2 5% "skip" was a coincidence (last step's LATENT converged to 0.03
  while its noise_pred still moved 34%, with low logp impact). **P0.2 dropped** —
  building a modulated extractor cannot beat a 0% ceiling.
- **TeaCache verdict for cosmos: correct + drift-safe but MARGINAL (~0% real win).**
  The port stays (default-off, harmless, tested) as infra for longer-schedule
  families where redundancy exists — see P2 (sd3/wan step counts).

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

- **2026-06-20 h3** — P0.2 RESOLVED as a negative result. `--diagnose` showed
  cosmos's consecutive exact noise_preds move 34–138% EVERY step → **0/20 steps
  skippable at any threshold**. The 5% wall is STRUCTURAL (20-step EDM schedule,
  no redundancy), not a signal problem, so a modulated extractor (P0.2) is futile —
  dropped. TeaCache = correct + drift-safe but ~0% real win on cosmos; port kept as
  infra for longer-schedule families. A 1.5-min diagnostic saved building a useless
  extractor. Pivot to P2 (do sd3/wan even use longer schedules?) and P3 (AR).
- **2026-06-20 h2** — P0.1 DRIFT measured (`teacache_drift_probe.py`, real cosmos,
  two-pass). Drift SAFE at all thresholds (max 0.43% « fp8's 1%). BUT corrected the
  h1 number: profiler's 2.3x/50%-skip was a **measurement artifact** (state reuse +
  idx%num_steps). Real 20-step denoise skips only **~5%** with the v1 raw-latent
  signal → real speedup ~5%, not 2.3x. ⇒ P0.2 (timestep-modulated signal) promoted
  from optional to REQUIRED; it's the lever, not the threshold. Drift dimension is
  done & green either way.
- **2026-06-20 h1 (cont.)** — [SUPERSEDED by h2] profiler `--teacache` showed
  0.56 vs 1.28 s/step "2.3x" — later found to be a state-reuse artifact.
- **2026-06-20 h1** — P0 core landed: TeaCache module + layout/executor wiring +
  9 unit tests (green), lint clean, default-OFF. Branch `rollout-vllm-teacache`
  cut from `fp8-rollout-precision-tis`. Profiler already has `--precision
  {fp32,fp16,bf16,fp8}` + `--fp8-recipe`. Earlier this session: cosmos precision
  ladder profiled (fp32 OOM; fp8 rowwise 1.15 s/step = 1.1x vs bf16 1.29; vLLM
  blockwise 5.97 s/step = a TRAP, reverted as a recommendation).

## Next actions (cron picks the top unchecked)
1. P2-cheap (non-GPU first): grep the sd3.5 / wan_2_1 / wan_2_2 configs for
   `num_steps`. If they also use ~20 steps, TeaCache is marginal across ALL this
   repo's diffusion configs → fully down-rank, keep port as dormant infra. If any
   uses 50+ steps, run `--diagnose` on it to see if redundancy exists there.
2. P3: AR paged-decode vs vLLM-omni continuous-batching throughput spike — the real
   "replace" candidate and likely the biggest win on this box. Start non-GPU:
   inventory what nextstep/janus rollout does today vs what vLLM-omni AR offers.
3. P2-full: sd3 vLLM-omni `DiffusersPipelineLoader` forward-consistency spike (only
   if step 1 shows a longer-schedule family where the engine bundle pays).
4. fp8 stands as the one shipped diffusion rollout win (1.1x, drift-safe). TeaCache
   stays default-off infra. Update the final summary table accordingly.

## Blockers
(none)

## Non-goals
- Pushing the branch. Deleting anything. Migrating diffusion wholesale to vLLM.
- Touching the trainer/replay precision (rollout-only changes).
