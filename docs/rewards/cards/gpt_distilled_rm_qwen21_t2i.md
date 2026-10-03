# Reward card: reward head distilled from GPT 5.6 Luna pairwise ratings, Qwen-Image-2.1 text-to-image

Goal: a local, deterministic scalar reward that reproduces GPT's "genuine / fidelity / structure" judgments
(`gpt56luna_codex_judge_qwen21_t2i.md`), usable as a GRPO group reward where GPT itself cannot be.

## Data (2026-10-02)

Cross-arm pairs of held-out eval images (shards 0-1, 80 prompts; shard 2 kept for evaluation): arms = base, stage-1
ck10-ck60, anchored ck10-ck40. GPT rating protocol (both orders, reasoning high): 1461 pairs with both orders, 2924
calls, 39 failures. Label = mean composite rating difference (genuine 2, fidelity 1, structure 1). GPT's own
order-consistency: 0.71 over all pairs, **0.91 on pairs with |gap| >= 2** (65% of pairs) -- the usable ceiling.

Split by prompt: 61 train prompts (1165 pairs), 15 validation prompts (296 pairs, 189 with |gap| >= 2).
Metric: sign agreement with GPT on validation pairs with |gap| >= 2.

## Backbone / head sweep (frozen features unless noted; MLP = LN-Linear(256)-GELU-Dropout-Linear(1), Bradley-Terry loss)

| features | dim | agreement |
|---|---|---|
| SigLIP so400m, token mean | 1152 | 0.65 |
| SigLIP token mean+std, linear ridge | 2304 | 0.71 |
| SigLIP token mean+std, MLP | 2304 | 0.709 |
| SigLIP learned attention pool over tokens | -- | 0.698 |
| SigLIP attention-pooled head output | 1152 | 0.683 |
| SigLIP global + 2x2 crops (mean+std) | 5760 | 0.656 |
| SigLIP last 4 blocks fine-tuned (lr 1e-5, 15 ep) | -- | 0.677 |
| DINOv2-L cls+mean+std | 3072 | 0.667 |
| SigLIP + DINOv2 | 5376 | 0.667 |
| **CLIP ViT-H/14 (laion2B) pooled+mean+std** | 3840 | **0.725** |
| Qwen3-VL-8B vision tower mean+std (896 px) | 2304 | 0.704 |
| Qwen3-VL-32B-FP8 vision tower mean+std | 2304 | 0.709 |
| CLIP-H + Qwen3-VL-32B | 6144 | 0.709 |
| all four concatenated | 14208 | 0.603 |
| pointwise regression on mean absolute rating (CLIP-H) | 3840 | 0.672 |
| reference: PickScore / SigLIP-aes / HPSv3 vs the same labels | -- | 0.66 / 0.60 / 0.58 |

Reading: every backbone lands at 0.65-0.73 and more capacity does not help; the validation set is 15 prompts
(±0.065 s.e.), so these are indistinguishable. The limiting factor is data: 61 prompts' worth of cross-arm pairs
are highly correlated. Next: a dedicated dataset -- base / stage-1 ck30 / stage-1 ck60 / anchored ck40 on 300
*training* prompts (`/mnt/nvme/eval/rmdata_s*`), all 6 arm pairs per prompt GPT-rated (3600 calls), then
re-train on ~2600 pairs from ~360 prompts.

**Not qualified yet** (gate 2 target ≥ 0.80 on GPT, gates 3/4 not run).
