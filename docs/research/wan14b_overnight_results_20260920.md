# Wan 14B overnight results — 2026-09-20

The four-L40S Wan2.2 A14B LoRA experiment completed 31,196.98 seconds
(8 h 39 m 57 s) of uninterrupted training in its recovery run, with three
updates. Including the two earlier saved updates, the final checkpoint is step 5.
All five updates passed exact-zero replay logprob parity with finite nonzero
gradients. Both experts' rank-32 FP32 adapters were trained; base weights remained
frozen in native mixed precision. The reward was Kling VideoReward visual_quality.
Training used 16 VideoPhy-derived prompts, 32 generated videos/update,
480×832×33 frames, 20 denoise steps and 19 replay timesteps, CFG 4.5.

## Held-out evaluation

24 held-out prompts × 2 matched seeds × base/final = 96 videos. Scores below
average the two seeds per prompt before computing paired differences; the
independent comparison unit is the prompt. All videos and paired seeds were
verified. Positive deltas favor the final adapters.

| Kling score | Base mean | Final mean | Delta | Prompt wins | Approx. 95% paired t interval |
|---|---:|---:|---:|---:|---:|
| Visual quality (training objective) | -0.49175 | -0.45577 | +0.03598 | 17/24 | [-0.0123, 0.0842] |
| Motion quality | -0.12101 | -0.08277 | +0.03824 | 14/24 | [-0.0167, 0.0932] |
| Text alignment | 0.29190 | 0.32527 | +0.03337 | 13/24 | [-0.0268, 0.0935] |
| Overall reward | -0.32086 | -0.21327 | +0.10759 | 16/24 | [-0.0146, 0.2298] |

This is a modest positive signal, not conclusive evidence of general quality
improvement: all mean-delta intervals include zero, the prompt set is small,
and the evaluator is the same reward-model family used for optimization.
No human preference or physical-correctness improvement is claimed.

## Performance and failures

Batch-1 generation of 32 videos took 1,871.7 seconds versus 1,681.0 seconds for
the earlier batch-2 attempt (11.3% longer). Batch 2 failed during replay from
GPU OOM. Batch 1 and NVMe-backed trainer parking enabled complete updates.
Replay forward-completion intervals were about 55 seconds, including backward
and orchestration; 152 evaluations/rank account for roughly 139 minutes/update.
First cold update launch-to-final-checkpoint was 178.2 minutes. Recovery training
averaged 173.3 minutes/update across three updates, including phase transitions
and periodic checkpoints but excluding initial construction. This is not a
controlled speedup comparison. Physical GPU memory during replay was roughly
43 GiB/GPU. No detailed compute/communication profiling was performed.

The initial continuous attempt saved step 2, then hit host OOM while all ranks
parked simultaneously. Serializing NVMe parking passed four-GPU exact update
 equivalence tests and then two real inter-update transitions in the recovery run.

After the final checkpoint was written, the EBS backup headroom guard failed
and terminated the process during shutdown. Therefore this was not a clean
process exit. All four ranks had already logged the duration-stop boundary,
and training_duration.json and the final checkpoint were complete. An older EBS
backup was retained on NVMe, the final backup was completed, and every file's
SHA-256 was checked against the NVMe original. Original failure records remain.

## Artifacts

- [Evaluation, videos, configs, metrics, and completion audit](../../outputs/wan14b_overnight_final_20260920/)
- [Detailed paired scores and intervals](../../outputs/wan14b_overnight_final_20260920/results_with_intervals.json)
- [Staging comparison baseline](wan14b_staging_baseline_20260919.md)
- Final NVMe checkpoint: `/mnt/nvme/vrl-night-20260918/runs/wan14b_priority_v5/continuous_recovery1/checkpoint-final`
- Verified EBS backup: `/home/ubuntu/VRL-night-20260918/outputs/wan14b_priority_20260919/attempt05/persistent_checkpoints/checkpoint-latest`

The experiment and held-out evaluation are finished. H3 and unrelated jobs
remain deferred. The evidence does not establish that further unbounded training
or full-parameter fine-tuning would improve quality.
