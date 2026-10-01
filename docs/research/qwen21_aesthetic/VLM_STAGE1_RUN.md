# Stage 1 with the Qwen3-VL-32B pairwise judge (`vlm_stage1`)

Started 2026-10-01 01:42. The Seedream 3.0 / RewardDance recipe applied to the shared-noise sampler: the
reward is a large generative VLM judging each rollout against a deterministic base-model image of the same
prompt, not a CLIP-style scalar. Reading note: `READING_seedream_rlhf.md`; judge card:
`docs/rewards/cards/vlm_preference_qwen3vl32b_qwen21_t2i.md` (pairwise mode passes gates 1-4).

## Setup

- Preset `experiment/qwen_image_21/online_grpo_aesthetic_shared_noise_4gpu` + `/reward=vlm_preference_http`;
  3 FSDP ranks on GPUs 0-2 (colocated trainer/rollout), judge service on GPU 3
  (`vrl/config/reward_service/qwen3vl32b_pairwise.yaml`, port 8318, 41-43 GB).
- Sampler: 1024 px / 40 steps, SDE noise 0.7 on 2 steps in [0, 6], shared initial latent per group of 8.
- Optimizer: lr 1e-4, 4 optimizer steps per batch, ppo_epochs 1, 2 prompts per round (16 rollouts),
  LoRA rank 32, `shard_trainable_only`. 60 epochs, checkpoint every 5.
- Data: 400 prompts sampled (seed 20261001) from `manifests/pickscore_sfw/train_safe.txt`, each with a base
  reference image `aes_refs/stage1/{i:03d}.png` (native 1024/40, seed 7000+i) named by
  `metadata.judge_reference` in `/mnt/nvme/aes2/stage1_prompts.jsonl`.
- Reward: `vlm_preference` = weighted mean of P(candidate preferred | both orders) over criteria
  overall (w=2), structure, clean. 0.5 means "as good as the base image for this prompt".
- Cost: ~15 min per round (6 min generation per prompt per rank, ~2 min judge, ~2 min park/update);
  judge adds no wall time because it overlaps generation.

## Progress

| epochs | mean reward (5-round avg) | reward_std range |
|---|---|---|
| 0-4 | 0.49 | 0.18-0.24 |
| 5-9 | 0.56 | 0.15-0.25 |
| 10-14 | 0.64 | 0.15-0.21 |

grad_norm 1e-4 and approx_kl ~0 at the first optimizer step are the same as `main_run1` (which learned);
the reward is moving while its within-batch std holds, which is the RewardDance collapse signal we watch.

## Planned evaluation

Deterministic generation on the 115 safe held-out prompts for base vs checkpoints 10/20/30/..., then
HPSv3 / SigLIP / PickScore as offline diagnostics and blind pairwise judging for appeal, adherence and
drift (same protocol as `SHARED_NOISE_RUN.md`), run after training frees the GPUs.
