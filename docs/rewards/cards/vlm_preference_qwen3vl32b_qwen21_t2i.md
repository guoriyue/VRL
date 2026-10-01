# Reward card: vlm_preference (Qwen3-VL-32B generative judge), Qwen-Image-2.1 text-to-image

- **Kind**: learned, generative (next-token probabilities of a VLM judge).
- **Judge**: Qwen/Qwen3-VL-32B-Instruct-FP8 @ 4bf2c2f3, 1024 px input, one batched forward per image.
- **Code**: `vrl/rewards/models/vlm_preference.py`; service `vrl/config/reward_service/qwen3vl32b_preference.yaml`.
- **Sampler this card applies to**: 1024 px / 40 steps, Flow-GRPO SDE noise 0.7, 2 stochastic steps in [0, 6],
  shared initial latent per group (same as `hpsv3_siglip_qwen21_t2i.md`). Data: the 64 frozen qualification prompts × 8
  (`/mnt/nvme/aes2/qual`), blind A/B appeal labels already collected for 134 pairs of that set.

## Pointwise mode (`DEFAULT_CRITERIA`: aesthetic rating, structure yes/no, clean rating, alignment yes/no)

**Ready for a training key: no (gate 2).**

| gate | result | verdict |
|---|---|---|
| 1 repeatability | max score range 0.0 over 32 images × 2 | PASS |
| 2 agreement | directional accuracy vs blind appeal labels (134 pairs): composite **0.567**, aesthetic 0.515, structure 0.507, clean 0.575, alignment 0.612. Same pairs: HPSv3 0.694, PickScore 0.627, SigLIP 0.552 | FAIL |
| 3 shortcuts | share of transforms scoring ≥ the genuine image: blur r3 3%, noise σ25 3%, black 0%, checkerboard 0%, noise σ10 31% | PASS (noise σ10 logged) |
| 4 spread | within-group std of the key 0.040 (median 0.026); alignment saturates (median 0.000) | weak |

Reading: the judge is deterministic and nearly impossible to fool with degraded images, but it does not resolve the
small differences between eight shared-noise siblings of one prompt -- absolute ratings saturate per prompt (aesthetic
≈ 0.99 for most groups). Adding it to HPSv3 lowered directional accuracy (HPSv3 + 0.25·VLM: 0.679; + 1.0·VLM: 0.642;
guard-gated HPSv3: 0.627). The 134 labeled pairs were built from the earlier rewards' top-vs-random picks, which
inflates those rewards' accuracy somewhat, but the VLM's own criteria sit at chance, so the verdict stands.

Blind spots to carry forward: fine within-prompt ranking; alignment question saturated (useless as a ranking signal at
this sampler, still a guard against off-prompt collapse).

## Pairwise mode (`DEFAULT_PAIRWISE_CRITERIA`, candidate vs a deterministic base image of the same prompt, both orders)

Pending: gate 2/4 running on the same 512 images against `/mnt/nvme/aes2/qual_ref` (seed 7000+i, native 1024/40).
Smoke test on 3 groups: reference vs itself = 0.500 exactly (position bias cancels); within-group std of the
"overall" criterion 0.04–0.14 (3–5× the pointwise spread); 7.3 s per image; 44 GB peak.
