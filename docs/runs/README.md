# docs/runs — curated training-run records

This is **not a dump of every run.** It is a small, curated library of runs worth
remembering, kept for one of two reasons:

- ✅ **Good example** — an exemplary *successful* run that shows what healthy training
  looks like (reward curve, eval-image progression). Use it as a reference when judging
  whether a new run is going well.
- ⚠️ **Issue-exposing** — a run that surfaced a bug, a wall, or an open question that
  must be **remembered or fixed**. Use it so we don't re-hit the same problem or forget
  the lesson.

Each run directory keeps the lightweight, reviewable artifacts (`metrics.csv`,
`eval_metrics.csv`, `resolved_config.yaml`, launch command) and — going forward — a
`README.md` explaining what it shows. Large checkpoints (GBs) stay out of the repo; large
eval-image grids are kept only when they *are* the lesson (visual proof of learning).

**When adding a run:** only add it if it's a good example or it exposes something worth
remembering. Write a `README.md` (config + result + what it teaches). Don't commit
multi-GB checkpoints.

---

## Index

| Run | Type | What it teaches |
|---|---|---|
| `sd3_5_ocr_grpo_fp32_20ep_20260507_194051` | ✅ Good example | Successful SD3.5-medium OCR-GRPO run (fp32, 20→200 epochs). Eval-image grids (`eval_epoch_*`, `comparison_*`) show the model visibly improving across epochs — the reference for **what a working RL training curve + eval progression looks like**. |
| `sd3_5_ocr_grpo_crossnode_continuous_3epochs` | ⚠️ Issue-exposing | Cross-node continuous-orchestration validation (3 epochs). Its old `resolved_config.yaml` (`mixed_precision:'no'` / `bf16:false`) is the **triggering artifact for `SPRINT_precision_naming_unification`** — the precision-knob naming mess. Don't delete (referenced by that sprint). |
| `sd3_5_ocr_grpo_crossnode_continuous_real_wait1800` | reference | Cross-node continuous run with a realistic `wait_timeout=1800` (1 epoch, has `checkpoint-final`). Validates the continuous queue under real wait timing. |
| `sd3_5_ocr_grpo_crossnode_profile` | reference | Cross-node profiling run (`gpu_usage`, 0 training epochs) — perf/placement profile of the cross-node path. |
| `cosmos_predict25_nft_kling_480p33f_rbs16_20260620` | ⚠️ Issue-exposing | Cosmos-Predict2.5 2B DiffusionNFT + Kling-reward, DDP 2×1 on 2× L40S, 6 epochs. **Pipeline works end-to-end but shows no learning signal** (reward oscillates −4.37…−4.56). Exposed: host-RAM OOM at 512p (→ `trajectory_storage.dtype=bfloat16` fix), torch.compile rank-desync at 512p (→ compile off), 512p/93f impractical on L40S (→ 480p_33f). Open question: **why no learning** (too few epochs? weak advantage signal?). See its `README.md` for the full engineering journey. |

---

## Conventions

- **Name:** `<model>_<task/algo>_<key-config>_<date>` so the run is identifiable at a glance.
- **Keep in git:** `README.md`, `metrics.csv`, `eval_metrics.csv`, `resolved_config.yaml`,
  launch command, and a *small* eval-comparison image if it is the lesson.
- **Keep out of git:** multi-GB checkpoints (leave in `outputs/`, reference the path),
  raw rollout videos, full per-step logs unless small.
- A run that is neither a good example nor issue-exposing does **not** belong here — let it
  live in `outputs/` and be cleaned up normally.
