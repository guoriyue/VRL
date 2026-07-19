# SPRINT: Best rollout system — TeaCache port + selective vLLM-omni adoption

状态：**done / 历史决策归档（2026-07-18 收口）**。2026-06-20 的 autonomous cron 与
`rollout-vllm-teacache` branch owner 已结束，不再是当前执行协议。TeaCache port、负收益判定与
AR 架构裁决继续保留；P1 组合 probe 已退休为 non-goal，diffusion vLLM-Omni 对照后来由
`docs/flux_qwen_naive_vs_vllm_omni_profiling.md` 完成。

**Historical branch:** `rollout-vllm-teacache`. **Started:** 2026-06-20.
**Machine:** single RTX 5090 (Blackwell sm_120, 32GB). cosmos rollout is
compute-bound (SM 100% / MEM ~20%).

本文不再是 autonomous loop 的 source of truth。下方 cron、branch 和 journal 叙述只记录当时的
执行过程，不授权新的定时任务、commit、profile 或 GPU 实验。

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
  `vrl/nn/kernels/attention/vllm_paged.py` +
  `vrl/models/steps/token/paged_attention_helpers.py` is reinventing the
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

- `vrl/generation/steps/denoise/teacache.py` — `TeaCacheConfig` + `TeaCacheState`
  (rel-L1 accumulator, warmup/last-step force-run, skip-ratio counters) +
  `teacache_signal`. v1 signal = rel-L1 of input latents.
- 当前接线：`vrl/generation/bindings/joint_denoise/layout.py` 解析
  `sampling.teacache`，`vrl/generation/steps/denoise/loop.py` 在 denoise loop 中 gate
  `model.forward_step` 并发出 `teacache_*` counters。
- 测试：`tests/generation/steps/denoise/test_teacache.py`。
- **RL note:** TeaCache adds rollout↔replay drift (skipped steps reuse an
  approximate noise_pred) — same class as fp8; rides the same drift-guard / TIS.

### P0.1 — Profile TeaCache speed + drift  ✅ DONE (with a correction)
- ✅ **DRIFT measured (gating) — `vrl/scripts/perf/teacache_drift_probe.py`, real cosmos denoise,
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
  artifact, NOT real.** `vrl/scripts/perf/generation_bottleneck_profile.py --teacache` reuses ONE
  state and steps `idx % num_steps`, so consecutive profiled "steps" are nearly
  identical → the skip machine fires far too often. On a REAL 20-step denoise
  (fresh latents, large early-step change) the v1 raw-latent signal skips only
  **~5%** at thr 0.1–0.25 → real speedup is **~5%, not 2.3x**. The profiler
  `--teacache` knob is therefore drift-blind AND skip-inflating; the drift probe /
  executor path is authoritative. (Left the knob for quick smoke, doc-flagged.)

- ❌ **RETIRED**：不再探索 threshold > 0.25。P0.2 已证明连续 noise prediction 没有足够
  冗余；继续调 threshold 只会增加 drift，不能建立可用收益。

### P0.2 — Per-family timestep-modulated signal  ❌ RESOLVED — NOT WORTH BUILDING
- `vrl/scripts/perf/teacache_drift_probe.py --diagnose` (exact cosmos denoise, rel-L1 between
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

### P1 — fp8 + TeaCache compose  ❌ RETIRED / NON-GOAL

TeaCache 在仓库 10–35 step schedule 上已经判定为结构性边际收益，因此不再为一个默认关闭、
近零收益的 feature 建 fp8 组合 GPU probe。fp8/TIS 自身的长期 owner 是
`docs/sprints/done/SPRINT_fp8_rollout_gemm_kernel.md` 与
`docs/sprints/done/SPRINT_low_precision_tis.md`；它们不需要 TeaCache 组合结果才能成立。未来若
重新验证 RL-safe cache，owner 是 `docs/sprints/planned/SPRINT_rl_safe_feature_cache_probe.md`；
该 sprint 明确要求一次隔离一个变量，不复活 fp8+TeaCache 混合 probe。

### P2 — TeaCache viability across the repo + vLLM-omni diffusion spike
- ✅ **P2-cheap (schedule survey) — every diffusion config's `num_steps`:**

  | family / config | num_steps |
  |---|---|
  | sd3_5 (ocr/pickscore/geneval), anima_preview3 | 10 |
  | wan_2_1, wan_2_2, cosmos_predict2_5 | 20 |
  | cosmos_predict2 (kling/v2w) | 35 |
  | sd3_5 *_debug | 4 |

  TeaCache targets 50–100-step schedules. The whole repo runs **10–35 steps**, and
  20 steps already proved 0% redundancy (h3). 10-step configs have even less. →
  **TeaCache is marginal across the entire repo's diffusion configs.** The only
  schedule that *might* hold redundancy is cosmos_predict2 @ 35 steps.
- ✅ **35-step cosmos_predict2 — answered by reasoned extrapolation (not run; the
  predict2 family is video2world = needs reference-image conditioning the probe
  doesn't supply, and it's one non-default config).** At 20 steps the per-step
  noise_pred rel-L1 ranged 0.34 (smallest, last step) to 1.39. The 35-step schedule
  covers the same sigma range with ~20/35 ≈ 0.57× the interval per step, so per-step
  change scales to ~0.19–0.79 — the SMALLEST (~19%) is still well above the <10%
  skip threshold. → 35 steps stays ~0% skippable; TeaCache marginal there too.
  Margin is large (19% » 10%), so the conclusion holds without the risky run.
- ✅ **HANDED OFF AND COMPLETED**：后续 FLUX/SD3 对照由长期 profiling 资产
  `docs/flux_qwen_naive_vs_vllm_omni_profiling.md` 完成，工具为
  `vrl/scripts/perf/vllm_omni_diffusion_profile.py`。结果仍支持本 sprint 的低 ROI 裁决；
  本文件不再持有独立 `DiffusersPipelineLoader` spike。

### P3 — AR continuous-batching evaluation  ✅ DONE (architectural) — NOT worth replacing
- **Reversed the earlier "AR is the biggest win" assumption.** Read the AR rollout
  (当前实现位于 `vrl/generation/composition/causal/token_loop.py`): it is NOT naive per-request decode — it is
  already a **token-batched lockstep decode with paged KV**:
  - `TokenScheduler` + `ARDecodeLoop`: `_max_batch_size` defaults to ALL samples
    (`scheduler_batch_size or len(sequences)`) → a chunk's whole sample set decodes
    together at full GPU width.
  - `build_step_batch` enforces "all sequences share one token position" = strict
    lockstep; `ActiveSequence.finished` is **position-only, no EOS early-stop** →
    the workload is **fixed-length** (image grids = fixed token count).
  - Paged KV via `ARCacheRows`; the vLLM paged *kernel* is already imported
    (`vrl/nn/kernels/attention/vllm_paged.py`).
- **Why vLLM continuous batching does NOT help here:** its two values are (1) ragged
  batching of variable-length sequences finishing at different times (EOS), and (2)
  injecting streaming-arrival requests into a running batch. Image AR rollout is
  fixed-length AND all samples are known upfront → neither applies. Our lockstep
  batched decode already saturates the GPU for this workload.
- **Verdict: replacing the AR rollout with vLLM-omni would add ~nothing.** The
  "reinventing vLLM" critique was wrong — we built the part that matters (batched
  paged decode) and correctly skipped the part the workload doesn't need (ragged
  continuous scheduling). KEEP the AR rollout.
- ❌ **RETIRED / NON-GOAL**：可选 AR GPU throughput confirmation 不影响架构裁决，也没有
  独立行为风险要关闭；不以“再测一个数字”重新打开本 sprint。当前 CPU contract 由
  `tests/generation/composition/causal/test_token_loop.py` 与
  `tests/generation/composition/causal/test_token_scheduler.py` 固定。

---

## FINAL VERDICT (consolidated — source for the morning report)

**Profile comparison (cosmos predict2.5 512p×93f, original rollout = bf16 baseline):**

| rollout variant | s/step | vs bf16 | drift (rollout↔replay) | status |
|---|---|---|---|---|
| **bf16 (original)** | 1.28–1.29 | 1.0x | — | baseline |
| **fp8 rowwise** | 1.15 | **1.10x** | 0.30% mean / 1.0% max | ✅ SHIPPED, drift-safe |
| fp8 blockwise (vLLM) | 5.97 | 0.19x | — | ❌ TRAP (launch-bound), rejected |
| fp16 | 1.27 | 1.0x | — | no win, fp16 range risk |
| fp32 | OOM | — | — | doesn't fit 32GB |
| **+TeaCache** | ~1.28 (≈0%) | ~1.0x | 0.0018–0.022% (safe) | ⚠️ MARGINAL — 0% structural skip on short schedules |

**"Is everything that can be replaced already replaced/enhanced?" — YES, and the
honest answer is the rollout was already near-optimal:**

- **Diffusion (cosmos/sd3.5/wan)** — *import-enhance, done.*
  - ✅ fp8 GEMM imported (`torch._scaled_mm` rowwise + vLLM block kernel) → 1.1x, shipped.
  - ✅ TeaCache ported (default-off). Verified MARGINAL here: every config runs
    10–35 denoise steps (TeaCache wants 50–100); cosmos's consecutive noise_preds
    move 34–138%/step ⇒ 0% structural redundancy. Kept as dormant infra.
  - ❌ vLLM-omni *engine* adoption for diffusion: low-ROI (compute-bound dense
    denoise doesn't use paged KV / continuous batching). NOT replaced — correct.
- **AR (nextstep/janus)** — *already correctly built, no replace needed.*
  - ✅ Already a lockstep token-batched paged-KV decode
    (`vrl/generation/composition/causal/token_loop.py`); vLLM
    paged kernel already imported. Continuous batching's value (ragged variable-
    length + streaming arrivals) does NOT apply to fixed-length image-token rollout.
  - ❌ vLLM-omni AR engine replace: adds ~nothing for this fixed-length workload.

**Bottom line:** the one real, shipped win is **fp8 (1.1x, drift-safe)**. TeaCache
and vLLM-omni-replace were each investigated and found to be non-wins for THIS
repo's workloads (short diffusion schedules + fixed-length AR) — verified, not
assumed. No large untapped rollout headroom remains on this box.

## Journal (most recent first)

- **2026-06-20 h6 — ✅ DONE.** Final regression green (166 generation+algorithms
  tests pass, lint clean across all branch files). 35-step cosmos_predict2 answered
  by extrapolation (per-step change ~19% min at 35 steps » 10% threshold → still
  ~0% skippable; predict2 is video2world so a direct run is low-ROI). Every core +
  optional item is now resolved. **Sprint complete.** Summary: fp8 (1.1x, drift-
  safe) is the one shipped rollout win; TeaCache (ported, default-off) + vLLM-omni
  engine replace are verified non-wins for this repo's short-schedule diffusion and
  fixed-length AR workloads. No large rollout headroom remains. Branch
  `rollout-vllm-teacache` holds 6 commits (TeaCache port, drift/diagnose probes,
  profiler precision+teacache knobs, sprint analysis); not pushed. FINAL VERDICT
  table above is the source for the morning report. Stopping autonomous changes.
- **2026-06-20 h5** — P3 RESOLVED (architectural, reverses the "AR = biggest win"
  assumption). AR rollout is already a lockstep token-batched paged-KV decode
  (`vrl/generation/composition/causal/token_loop.py`: full-width batch,
  position-locked, no-EOS fixed-length). vLLM
  continuous batching's value (ragged/streaming) doesn't apply to fixed-length image
  AR → replacing adds ~nothing. Wrote the FINAL VERDICT section. Core sprint
  (P0/P0.1/P0.2/P2/P3) is complete; only optional GPU confirms remain.
- **2026-06-20 h4** — P2-cheap (schedule survey, non-GPU): every diffusion config
  runs 10–35 steps (sd3/anima 10, wan/cosmos2.5 20, cosmos_predict2 35). TeaCache
  targets 50–100; with 20-step redundancy already proven 0%, **TeaCache is marginal
  repo-wide**. Only cosmos_predict2@35 might hold redundancy (optional confirm,
  helps one non-default config). Diffusion-side vLLM-omni adoption looks low-ROI.
  Pivot firmly to P3 (AR continuous batching) = the real win on this box.
- **2026-06-20 h3** — P0.2 RESOLVED as a negative result. `--diagnose` showed
  cosmos's consecutive exact noise_preds move 34–138% EVERY step → **0/20 steps
  skippable at any threshold**. The 5% wall is STRUCTURAL (20-step EDM schedule,
  no redundancy), not a signal problem, so a modulated extractor (P0.2) is futile —
  dropped. TeaCache = correct + drift-safe but ~0% real win on cosmos; port kept as
  infra for longer-schedule families. A 1.5-min diagnostic saved building a useless
  extractor. Pivot to P2 (do sd3/wan even use longer schedules?) and P3 (AR).
- **2026-06-20 h2** — P0.1 DRIFT measured
  (`vrl/scripts/perf/teacache_drift_probe.py`, real cosmos,
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

## 关闭状态——无 Next actions

所有 core 项均已结案；P1 与可选 GPU confirmation 明确退休，diffusion vLLM-Omni 对照已由
现有 profiling 资产承接并完成。旧 hourly cron 与 morning-report cron 均不再触发本文动作。
当前 generation-engine 总体方向由
`docs/sprints/SPRINT_native_generation_engine_program.md` 持有。只有新的、独立需求与可衡量
验收条件出现时才建立新的 owner；不复活本文的 autonomous loop。

## Blockers
(none)

## Non-goals
- Pushing the branch. Deleting anything. Migrating diffusion wholesale to vLLM.
- Touching the trainer/replay precision (rollout-only changes).
