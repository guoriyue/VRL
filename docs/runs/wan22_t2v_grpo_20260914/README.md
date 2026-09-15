# Wan 2.2 T2V-A14B dual-expert GRPO learning experiment (2026-09-14)

Question: does this RL stack, on the real `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
weights with LoRA on both experts and four L40S, not only update parameters
correctly but produce a measurable learning effect on a held-out prompt set?

Short answer: **yes, within the tested scope.** After 8 GRPO updates (32
samples each) the trained objective, Kling VideoReward `visual_quality`,
improved by +0.163 ± 0.056 on the 16 training prompts and +0.170 ± 0.034 on
24 never-trained validation prompts (prompt-level paired deltas against the
untrained base model on a fixed prompt/seed grid; win rates 13/16 and 20/24;
bootstrap 95% CIs exclude zero). On the independent 35-prompt test split
(official VideoPhy eval captions, never used for any decision) checkpoint-8
scored +0.174 ± 0.044 (27/35 prompts). Checkpoints 2 and 4 showed no effect,
6 and 8 a monotone rise. Every update passed the exact replay-parity gate.
Stage 2 (updates 9-16) was stopped at the user's request after update 9 and
is queued to resume (see "What is not verified").

Everything below is reproducible from this worktree (`exp/wan22-t2v-grpo`,
commit `80175a2d`, branched from `feat/reward-reload-handoff@81415f9c`) and the
NVMe run directory `/mnt/nvme/outputs/wan22_t2v_grpo/`.

## 1. Setup decisions and the checks behind them

| Item | Decision | Evidence |
| --- | --- | --- |
| Code / numerics | The accepted four-rank native FP32-LoRA configuration (`wan22_rebased_gpu_checkpoint_four.yaml`: both experts trainable, FSDP `precision_policy=none` + CPU offload, `lora_parameter_dtype=float32`, generation batch == replay microbatch == 2, CPS SDE, Kling `memory_parking_mode=reload`, replay-parity gate 1e-8) | `docs/research/wan22_dual_expert_l40s_preflight_20260912.md`, `docs/research/reward_reload_handoff_20260913.md` (exact replay, exact cold resume, 3.0x over one card). Not re-run: no Wan/reward code changed. |
| Geometry | 320x320, 17 frames, **20** denoise steps (the accepted lifecycle proof used 10) | `configs/quality_probe.md` below: at 10 steps frame 0 is mosaic garbage and frames 1-3 ghost (model output, not codec: lossless PNG == mp4); VAE tiling off, 480p and the official negative prompt do not fix it; 20 steps is clean except residual first-frame speckle on ~half the clips; 40 steps is fully clean but 2x the cost. |
| Reward | Kling VideoReward, `score_key=visual_quality` | Probe on 32 base clips: repeat scoring exact (0.0); VQ lowered by frozen clip / gaussian noise on 78-84% of clips (correct sign); `motion_quality` lowered by noise on only 9-25% (inverted, as found earlier at 480p/33f); `text_alignment` correct (94-97%); `overall_reward` carries the inverted MQ term. Base spread: VQ std 0.24, within-prompt seed diff 0.15. |
| Data | VideoPhy captions: `train_small` 16 / `val` 24 (drawn from the official train split, seed 20260914) / `test` 35 (= official eval split, held out) | `datasets/videophy_wan22_grpo/report.json`: exact-normalized dedup, token-Jaccard near-duplicate screen (max cross-set 0.25 train↔val/test). No decision used the test set. |
| Batch / optimizer | 4 ranks x 1 prompt x 8 samples = 32 samples/update, group-local advantage normalization, one AdamW step per update, lr 1e-4 (repo Wan reference), EMA 0.9, clip 1e-4 (a drift rail only: ratio ≡ 1 with exact replay and one step) | `configs/stage1_grpo_4rank.launched.yaml` |
| Budget (declared before launch) | 8 updates, checkpoint every 2, ~30 min/update, stop on non-finite / parity failure / reward collapse / memory kill; continue only if paired train delta > 0 with win rate > 0.5 and val not negative | `STAGE1_PLAN.md` |

## 2. Mechanical correctness (stage 1, 8 updates, 14:10-18:11 PDT)

- All four ranks exited `status: success`; checkpoints 2/4/6/8/final written.
- `pre_update_logprob_abs_diff_max = 0.000000` on every update (rollout/replay
  log-probs identical through both experts and the 875 boundary; gate 1e-8).
- Gradient norm 0.0074, 0.0080, 0.0052, 0.0124, 0.0291, 0.0315, 0.0223, 0.0431:
  finite, nonzero, growing. `adv_zero_rate = 0`, per-group VQ std 0.08-0.32.
- Checkpoint audit (CPU): both experts hold 640 finite FP32 LoRA tensors; all
  320 `lora_B` per expert nonzero; every tensor changed between checkpoint-2 and
  checkpoint-8 (mean `lora_B` norm 0.034 -> 0.079 high-noise, 0.047 -> 0.113
  low-noise); 1280 Adam first moments nonzero; checkpoint-final == checkpoint-8;
  step / global_step / EMA count consistent.
- Strict resume: stage 2 restarted from checkpoint-8 with `start_epoch=8` and
  produced update 9 with replay diff 0 (numerical resume equivalence itself was
  established earlier by the reload-handoff evidence and was not re-measured).
- Per-update phase times (per rank, 8 samples): generation 635 s, reward 43 s,
  replay forward 470 s, backward 614 s. GPU memory ~14 GB/card; the run is
  PCIe-bound (both 28.6 GB experts streamed from host every step).

## 3. Learning effect (fixed prompt/seed grid, native 20-step sampler, 2 seeds/prompt)

Prompt-level paired delta vs the base arm (same prompts, same seeds; seeds of
one prompt averaged first). `eval_train_small/report.json`, `eval_val/report.json`.

| Updates | train_small VQ Δ (n=16) | win | val VQ Δ (n=24) | win |
| --- | ---: | ---: | ---: | ---: |
| 2 | +0.009 ± 0.030 | 0.56 | +0.043 ± 0.024 | 0.54 |
| 4 | −0.049 ± 0.032 | 0.38 | +0.026 ± 0.033 | 0.50 |
| 6 | +0.097 ± 0.053 | 0.62 | +0.138 ± 0.037 | 0.83 |
| 8 | **+0.163 ± 0.056** (CI +0.06..+0.27) | 0.81 | **+0.170 ± 0.034** (CI +0.11..+0.24) | 0.83 |

Test split (n=35, `eval_test/report.json`, scored 2026-09-15 05:10): VQ
+0.174 ± 0.044 (win 0.77), MQ +0.105 ± 0.033, TA +0.135 ± 0.043, overall
+0.413 ± 0.085. The effect size matches train_small and val, so the 8-update
gain generalizes beyond the trained prompts.

Diagnostic keys at checkpoint-8 (not optimized): val MQ +0.127 ± 0.040,
overall +0.354 ± 0.109, TA +0.057 ± 0.087 (flat on val, positive on test). Absolute VQ moved from −1.09
to −0.92: still below the reward's normalized zero, i.e. far from "good video".

What changed in the videos (`eval_val`, base -> ck8): the fraction of clips with
a corrupted first frame (frame0->1 mean abs diff > 10) fell 0.88 -> 0.58 (train
0.97 -> 0.53); frame-0 std fell to the level of frame 8; mean inter-frame motion
6.9 -> 6.1 (slightly more static); frames 8/16 of four inspected prompts stay
coherent with no blur, collapse or texture loss. The policy mostly learned to
repair the base model's bad first frame, which is exactly the defect the reward
probe showed Kling VQ penalizes most.

Training curve: `stage1_training_curves.svg` (batch reward means are not
comparable across updates; only the parity and gradient panels carry signal).

## 4. What is not verified / limitations

- **Test split**: evaluated for checkpoint-8 only (above); checkpoints 12/16
  will be added by the queued stage-2 evaluation job.
- **Stage 2** (updates 9-16): attempt 1 was killed by Ray's 95% host-memory
  monitor in its first generation phase (354.1/372.7 GB: four rollout workers
  56-62 GB each + four trainer ranks 28 GB each; stage 1 had the same peak and
  survived by ~3 GB). Attempt 2 with `RAY_memory_usage_threshold=0.98` (measured
  peak bounded at 351-354 GB) completed update 9 (min available 24 GB) and was
  then stopped by the user at 22:33. Nothing after checkpoint-8 is evaluated.
- Reward: Kling MQ is inverted at 17 frames; VQ is dominated by first-frame
  quality at this geometry. A rise in VQ is therefore a real, visible fix but a
  narrow notion of quality; TA did not move.
- Sampler mismatch: training explores with the CPS SDE, evaluation uses the
  deterministic native sampler (what inference deploys).
- Not exercised here: the full 480x832/81-frame geometry, compile, batch shapes
  other than 2, four-rank host-memory headroom (a code-level fix such as
  sharing frozen expert weights across the four rollout workers is still open).

## 5. Artifacts

- Data: `datasets/videophy_wan22_grpo/{train_small,val,test}.txt`, `report.json`, `make_splits.py`.
- Code: `vrl/scripts/eval/wan22_kling_checkpoint_eval.py` (generate / score / probe),
  `vrl/scripts/perf/wan22_generation_quality_probe.py`.
- Run directory `/mnt/nvme/outputs/wan22_t2v_grpo/`:
  `configs/*.launched.yaml`, `STAGE1_PLAN.md`, `NOTES_probes.md`, `tools/*.sh`,
  `stage1/` (metrics.csv, checkpoints 2/4/6/8/final, reward_debug, run_evidence),
  `stage2/` (update 9 only), `stage2_attempt1_raykill/`,
  `eval_train_small/`, `eval_val/`, `eval_test/` (videos per arm, scores.jsonl,
  report.json, reward_probe.json), `quality_probe/` (mp4 + lossless PNG frames),
  `eval10step_*/` (the retired 10-step base arms, kept as evidence of the mosaic defect),
  `logs/`.
- Copies of the launched configs, plan and probe notes: this directory.

## 6. Commands

```bash
cd /home/ubuntu/VRL-wan22
export PYTHONPATH=/home/ubuntu/VRL-wan22 HF_HOME=/mnt/nvme/hf/huggingface HF_HUB_OFFLINE=1
V=/mnt/nvme/venvs/vrl-review-all/bin/python
O=/mnt/nvme/outputs/wan22_t2v_grpo

# splits
$V datasets/videophy_wan22_grpo/make_splits.py

# generation-setting probe (GPU) and base arm + reward probe
CUDA_VISIBLE_DEVICES=3 $V -m vrl.scripts.perf.wan22_generation_quality_probe --run-dir $O/protocol --output-dir $O/quality_probe --prompt "..." --device cuda:0
CUDA_VISIBLE_DEVICES=0 $V -m vrl.scripts.eval.wan22_kling_checkpoint_eval generate --run-dir $O/protocol --output-dir $O/eval_val --prompts datasets/videophy_wan22_grpo/val.txt --limit 24 --samples-per-prompt 2 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $V -m vrl.scripts.eval.wan22_kling_checkpoint_eval probe --run-dir $O/protocol --output-dir $O/eval_train_small --limit 32 --device cuda:0

# training (4 ranks, GPUs 0-3); tools/launch_train.sh sets the env and the memory sampler
$O/tools/launch_train.sh $O/configs/stage1_grpo_4rank.yaml stage1
$O/tools/launch_train.sh $O/configs/stage2_resume_16.yaml stage2   # strict resume from checkpoint-8

# checkpoint arms + scoring (paired against the base arm already in the grid)
CUDA_VISIBLE_DEVICES=0 $V -m vrl.scripts.eval.wan22_kling_checkpoint_eval generate --run-dir $O/stage1 --output-dir $O/eval_val --prompts datasets/videophy_wan22_grpo/val.txt --limit 24 --no-base --checkpoint ck8=$O/stage1/checkpoint-8 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $V -m vrl.scripts.eval.wan22_kling_checkpoint_eval score --run-dir $O/stage1 --output-dir $O/eval_val --device cuda:0

# still to run for the test split (~45 min on one GPU)
CUDA_VISIBLE_DEVICES=0 $V -m vrl.scripts.eval.wan22_kling_checkpoint_eval generate --run-dir $O/stage1 --output-dir $O/eval_test --prompts datasets/videophy_wan22_grpo/test.txt --limit 35 --no-base --checkpoint ck8=$O/stage1/checkpoint-8 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $V -m vrl.scripts.eval.wan22_kling_checkpoint_eval score --run-dir $O/stage1 --output-dir $O/eval_test --device cuda:0
```
