# Shared-noise preference GRPO on Qwen-Image-2.1 (after DEBUG_20260927)

Goal (user, 2026-09-27): a correctly set up aesthetic GRPO run.

1. Choose a sampler where the policy controls the group contrast.
2. Qualify the reward blind at that sampler.
3. Show the policy moves on a small overfit set.
4. Train on 4 GPUs and judge base vs checkpoints blind on unseen prompts with deterministic sampling.

## 1. Sampler

Setup: 8 training prompts × 8 samples, one shared initial latent per group, 1024 px, 40 steps. The table gives the mean
within-group SigLIP standard deviation.

| SDE window / noise | within-group SigLIP std |
|---|---|
| old: 2 steps in [4, 36], noise 0.2 | 0.033 |
| **2 steps in [0, 6], noise 0.7** | **0.214** |
| 4 steps in [0, 10], noise 0.7 | 0.221 |
| 2 steps in [0, 6], noise 1.0 | 0.128 |
| 4 steps in [0, 10], noise 1.0 | 0.186 |
| reference: distinct initial latents | 0.366 |

Choice: 2 steps in [0, 6] at noise 0.7. Groups keep their composition and vary light, detail and styling
(`/mnt/nvme/aes2/n07w2_montage.jpg`).

## 2. Reward

See `docs/rewards/cards/hpsv3_siglip_qwen21_t2i.md`.

- **Blind top-of-group vs random** (64 prompts):
  - SigLIP 57.8%: FAIL.
  - PickScore 64.1%: PASS.
  - HPSv3 72.7%: PASS.
  - Composite HPSv3 + 1.0·SigLIP (z): 66.4% [55.5, 76.6], PASS.
- **Why the composite and not HPSv3 alone**: HPSv3 rewards mild pixel noise (+1.0 group std), and SigLIP cancels it.

## 3. Overfit test

Setup: 12 fixed prompts, 3 FSDP ranks, HPSv3 served on GPU 3, lr 3e-4.

### Making the policy move

- **Too few optimizer steps**: 2 optimizer steps per round at lr 1e-4 gave a per-round ratio deviation of ~1e-6 and KL ~1e-9.
  The policy did not move.
- **Fix**: `actor.optimizer_steps_per_batch: 8`, lr 3e-4, one PPO epoch.
- **Sharding the frozen base is too slow**: sharding the frozen 14 GB base under FSDP took more than 35 minutes per update on
  PCIe L40S.
- **Fix**: `fsdp.shard_trainable_only: true` brought it to about 3.5 minutes; a round is now about 20 minutes, mostly
  generation.
- **HPSv3 cannot share a colocated rank's card with the generator.** It now runs as an HTTP service
  (`vrl/config/reward_service/hpsv3.yaml`). Two fixes were needed:
  - the image-tensor bug in `HPSv3Model` (images were rejected as non-video tensors);
  - the request-size limit, raised to cover a batch of float32 1024 px images.

### Checkpoint 5 vs base

Native 1024 px / 40 steps, the 12 overfit prompts × 2 seeds:

| metric | Δ vs base [95% CI] |
|---|---|
| HPSv3 | +2.00 [+1.15, +2.92] |
| SigLIP | +0.45 [+0.17, +0.75] |
| brightness | −0.11 [−0.16, −0.07] |
| saturation | +0.026 |
| edge energy (grain) | −0.004 (no grain increase) |

The policy moves, so step 3 passes.

### What checkpoint 5 looks like

Visual check: `/mnt/nvme/aes2/ck5_vs_base.jpg`.

- **The outputs collapse toward one look**: cinematic, detailed, warm and dark photoreal portraits.
- **Prompt adherence and style suffer**:
  - #0: an anime prompt comes out semi-realistic.
  - #6: a lion T-shirt becomes a superhero.
  - #7: a pencil sketch grows extra eyes.
  - #10: logo text is lost.
  - #9: a child with a glowing object becomes a glamour adult woman with cleavage. This drift is unacceptable and must be
    checked on held-out evaluations.

This is reward overoptimization after 40 aggressive steps on 12 prompts. The main run therefore uses:

- **lr 1e-4**;
- **4 optimizer steps per round**;
- **the full PickScore-SFW prompt set** (15,486 prompts; 6 prompts × 8 samples per round);
- **selection by blind held-out judgments** of appeal, prompt adherence and inappropriate drift, never by reward.

## 4. Main run

`main_run1` (started 2026-09-28 03:05).

- **Launcher**: `/mnt/nvme/work/launch_aes_main.sh` with LR=1e-4, OPT_STEPS=4, prompts_per_batch=2, save every 5 rounds.
- **Output**: `/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/main_run1`.
