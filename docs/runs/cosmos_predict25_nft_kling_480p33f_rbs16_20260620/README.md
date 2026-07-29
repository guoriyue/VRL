# Run: Cosmos-Predict2.5 2B · DiffusionNFT · Kling video reward · DDP 2×1

**Date:** 2026-06-19 → 2026-06-20 · **Status:** stopped at epoch 6 (manually) · **Verdict:** pipeline fully works end-to-end, but **no clear learning signal in 6 epochs** (reward oscillates in a noise band).

Output (checkpoints + logs) lives at `outputs/cosmos_ddp_paper/` on the cluster (gitignored, NOT copied here — 5.1 GB each). This dir keeps the lightweight, reviewable artifacts: `metrics.csv`, `eval_metrics.csv`, `resolved_config.yaml`, `launch.cmd`.

---

## 1. TL;DR

- **Algorithm/model:** DiffusionNFT on Cosmos-Predict2.5 2B (LoRA), Kling VideoReward (Qwen2-VL-2B) as reward.
- **Batch (paper-faithful):** rbs=16/rank × 2 ranks = **32 conditions/update (paper)**, n=8 samples/prompt, no-CFG, 20 denoise steps.
- **Resolution (speed override, NOT paper):** **480p_33f (832×480, 33 frames)** instead of paper's 512p/93f — the user's deliberate trade to make epochs tractable on L40S.
- **Hardware:** 2× NVIDIA **L40S** (350 W, 46 GB), one GPU per node, cross-node DDP (gradient all-reduce only).
- **Speed:** ~13.5 min/group × 16 groups ≈ **~3.6 h/epoch**; ran 6 epochs in ~22 h.
- **Result:** reward `−4.48 (baseline) → oscillates −4.37…−4.56`, mean ≈ −4.43. **No trend.** grad_norm stable (0.18–3.3, one transient spike). Mechanism healthy; signal weak.

---

## 2. Exact config

Recipe: `experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1` (paper-aligned base).

Launch overrides (see `launch.cmd`):
```
rollout.rollout_batch_size=16        # paper batch: 16/rank × 2 ranks = 32 global conditions
rollout.sample_batch_size=4          # memory/chunk knob (NOT a paper RL param)
distributed.training.ddp.find_unused_parameters=true   # NFT previous/reference adapters get no grad
sampling.width=832 sampling.height=480 sampling.num_frames=33   # 480p_33f (speed override; paper is 512p/93f)
rollout.trajectory_storage.device=cpu
rollout.trajectory_storage.dtype=bfloat16              # host-RAM fix (see §5); harmless at 480p
model.torch_compile.enable=false     # disabled to avoid 512p cold-compile rank desync (see §5)
trainer.save_freq=3                  # checkpoint every 3 epochs
```
Paper RL params kept untouched: **n=8, no-CFG (guidance_scale=1.0), 20 steps, rbs=32 global, timestep_fraction=0.5**. Only resolution/frames (speed), sbs/trajectory dtype/compile (memory+perf) were overridden — none of which change the RL objective.

Full resolved config: `resolved_config.yaml`.

---

## 3. Results

Baseline eval (epoch −1, fixed 70-video eval set): **eval_reward_mean = −4.4847 ± 0.0339**.

Training reward (per-update mean over the 256 rollouts; `metrics.csv`):

| epoch | reward_mean | grad_norm | loss |
|---|---|---|---|
| 0 | −4.4312 | 0.2255 | 8.291 |
| 1 | −4.4163 | **3.2970** | 7.722 |
| 2 | −4.3666 | 0.8092 | 8.592 |
| 3 | −4.4262 | 0.5163 | 8.709 |
| 4 | −4.5616 | 0.1820 | 8.381 |
| 5 | −4.3688 | 0.4642 | 8.190 |

**Reading:** `−4.43 → −4.42 → −4.37 → −4.43 → −4.56 → −4.37`. Oscillates between ~−4.37 (high) and ~−4.56 (low); epoch-4 dipped **below** the baseline. Mean ≈ −4.43, marginally above baseline, **no upward envelope** → no learning visible in 6 epochs. The grad_norm=3.30 at epoch 1 was a one-off spike (back to <1 after); not sustained instability.

**Note** training reward (train prompts) and baseline eval (fixed eval set) are different prompt sets — not directly comparable level-to-level; the *trend within each* is what matters, and the training trend is flat/noisy.

Checkpoints (in `outputs/cosmos_ddp_paper/`): `checkpoint-3` (weights @ epoch 3, reward ≈ −4.43), `checkpoint-6` (weights @ epoch 6, reward ≈ −4.37).

---

## 4. Per-group cost profile (measured)

One group cycle = **~13.5 min** (very consistent across all 96 groups). Breakdown from log timestamps + py-spy:

| phase | time | share |
|---|---|---|
| Generation (8 videos × 20 denoise steps) | ~4.8 min | ~35% |
| Reward scoring (Qwen2-VL on 8 videos) + NFT backward | ~8.5 min | ~62% |
| Model swaps (rollout reload ~8 s + reward load ~9 s) | ~17 s | ~2% |

- **Biggest cost = NFT backward**, because NFT does **3 forwards/sample** (previous-policy / current-grad / reference) at the sampled timesteps.
- **Model on_demand swaps are negligible (~2%)** — NVMe-cached, ~8–9 s each. (Investigated because of GPU-memory dips in monitoring; confirmed not the bottleneck.)
- **Trajectory CPU-offload is NOT a bottleneck** — py-spy showed ~all time in transformer compute (Linear 20%, attention 12%, LoRA, NFT loss); no `.to()`/copy in the hot path.
- **Genuinely compute-bound:** both GPUs draw **307–330 W / 350 W (~90%)** during steps → SMs really busy, not stalled/scheduling. nvidia-smi 100% util alone can't prove this; power draw does.
- **`sample_batch_size` is ineffective at 480p_33f:** peak GPU ~32 GB at both sbs=2 and sbs=4 (footprint dominated by model + 33-frame activations, not the chunk), util already 100%. So "raise sbs to fill 46 GB" gives nothing here.

Rollout videos verified **real** (not color blocks, the no-CFG risk): decoded `sample-7-7.mp4` → per-frame spatial std mean **65.9**, frame-to-frame motion 4.84.

---

## 5. Engineering journey (walls hit → fixes) — important for reproduction

This run only became feasible after fixing a chain of issues at the original 512p/93f paper resolution:

1. **torch.compile cold-compile desync (512p):** ~30 min cold compile per rank caused the two ranks to desync at the NCCL all-reduce (one compiles while the other waits) → near-deadlock. **Fix:** `model.torch_compile.enable=false` (eager; ~1.37× slower steady-state but balanced). *At 480p the compile would be much shorter — re-enabling is the #1 untried speedup (~1.37×, recipe default is True).*
2. **save_freq=32 (recipe default) → ~weeks between saves. Fix:** `save_freq=3`.
3. **HOST-RAM OOM at 512p/93f (the big one):** rollout trajectories (latents/logprobs) accumulate in **host RAM**, hitting 49–58 GB on the 62 GB nodes → Ray killed the worker. **Not rbs-driven** (49 GB at both rbs=32 and rbs=8). **Fix:** `rollout.trajectory_storage.dtype=bfloat16` halved it (peak 47.7 GB, under the 58.8 GB kill threshold). This is the only thing that made 512p run at all. See memory note `cosmos-512p-host-ram-wall-bf16-fix`.
4. **512p/93f is ~7 h/epoch (≈75 days for 256 epochs) → impractical on L40S.** User chose to drop to **480p_33f** (832×480, 33f) keeping paper batch → ~3.6 h/epoch (~4.7× faster). 480p_33f has no host-RAM wall (~13 GB), so bf16/cpu trajectory storage is unnecessary but harmless.

DDP correctness verified: rank0 and rank1 each draw a **disjoint** 16-prompt slice → 32 global conditions; gradient NCCL all-reduced. 2 machines ≈ **genuine 2× speedup** for the paper batch (16 groups/rank in parallel vs 32 sequential on one).

---

## 6. Conclusion & next steps

**What works:** the full DiffusionNFT video-RL pipeline on Cosmos-Predict2.5, DDP 2×1 cross-node, end-to-end, no crashes, real rollouts, healthy gradients, checkpointing. The infra is validated.

**What's unresolved:** **no learning signal in 6 epochs.** Reward sits in a noise band around the baseline.

**ROOT CAUSE FOUND (2026-06-20) — the trained reward target is MIS-WEIGHTED at 480p_33f, not the policy.** Two de-risk probes (run on **both** L40S in parallel), all scores deterministic (rescore gap = 0.0000, so the spreads below are real model behavior):

*Probe 1* (mixed-degradation, **since retired**) — real rollout vs a heavily-degraded copy (noise + shuffle + dropped frames + an fps change): Overall did **not** drop (gap −0.24, degraded scored *higher*). Because it moved four variables at once the result could not be attributed to any one of them, which is exactly what Probe 2 was built to fix; the script was deleted so nobody re-runs the confounded version. To reproduce or extend this line of investigation, use `kling_reward_diagnosis_probe.py`.

*Probe 2* (`kling_reward_diagnosis_probe.py`) — a clean gaussian-noise ladder (σ=0,20,40,80,160) on 32 real rollouts, isolating one axis:

| dim | σ=0 | σ=160 | drop (s0−s160) | reading |
|---|---|---|---|---|
| VQ (visual) | −1.3916 | −1.4894 | **+0.0978** | RESPONDS — noise lowers it (correct) |
| MQ (motion) | −0.3962 | −0.0434 | **−0.3528** | **INVERTED** — noise monotonically *raises* it (flicker read as "motion") |
| Overall | −4.3965 | −4.2039 | **−0.1926** | **INVERTED** — MQ corruption flips the trained target |

The training reward is `score_key: overall_reward` (`configs/reward/kling_video_reward.yaml`). So the policy optimizes **Overall**, whose **MQ sub-score is inverted at 33 frames** — high-frequency noise/flicker is scored as good motion, so a *worse* video can score *higher*. That is why training was flat/gameable, **not** too-few-epochs. (VQ alone is correctly signed and real rollouts have a genuine quality spread — Overall std **0.18** across rollouts — so usable signal exists; it is the Overall/MQ combination that is broken.)

**Implication & cheap fix (test before any longer run):**
1. **Switch `score_key` `overall_reward` → `visual_quality`** — VQ is correctly signed and discriminating at 480p_33f. One-line config change, no resolution change.
2. The MQ inversion is most likely a **too-few-frames artifact (33f)**; restoring frames toward the paper's 93f should let MQ work — but that is expensive, so try (1) first.
3. **Re-run `kling_reward_diagnosis_probe.py` and require the *trained* key to RESPOND** (drop > +0.0339, not INVERTED) before investing GPU-days. A ≥30–50 epoch sweep on the current `overall_reward` would optimize a gameable target.

**Follow-up — is VQ's gradient *policy-improvable*? (2026-06-21, steps-ladder probe).** The diagnosis probe proves VQ responds to *artificial* noise; this asks the sharper question behind "we never improve regardless of reward": does VQ reward the kind of quality the policy could actually climb? Held the policy checkpoint, prompts, and seed fixed for every row, varied only denoise steps (more steps = objectively more-converged generation) via `cosmos_predict25_kling_eval --steps {8,16,32,64} --score-key visual_quality`:

| steps | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| VQ_mean | −1.5008 | −1.4718 | −1.4184 | −1.3100 |

**Monotone, +0.191 over 8→64, well above the ±0.034 eval noise.** So VQ rewards real generation quality (convergence) — it is **not** dead or inverted; there is a climbable gradient. Conclusion: the flat training reward is **not** a reward-deadness problem for VQ — the remaining suspect is **undersampling** (this run did 6 optimizer updates, the VQ fast run 2, vs the paper's 256). Caveat: this proves the reward landscape has a correctly-signed quality direction; it does not by itself prove RL will climb it at a fixed step count — only a long-enough run on `score_key=visual_quality` settles that. This de-risks the long run: VQ is worth GPU-days, `overall_reward` was not.

---

## 7. How to resume

On the cluster (2× L40S, node A master 172.31.36.21 + node B 172.31.32.107):
```
# launcher: /home/ubuntu/ddp_launch.sh  (resume-capable, auto-finds latest checkpoint)
bash /home/ubuntu/ddp_launch.sh 0   # rank0 on node A
bash /home/ubuntu/ddp_launch.sh 1   # rank1 on node B
```
It resumes from `outputs/cosmos_ddp_paper/checkpoint-<latest>`. See `launch.cmd` for the exact overrides. NOTE the hourly self-check cron + 30-min monitor that drove this run were stopped on 2026-06-20; restart them if running unattended again.
