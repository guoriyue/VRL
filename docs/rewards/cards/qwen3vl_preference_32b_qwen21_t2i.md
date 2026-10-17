# Reward card: qwen3vl_preference (Qwen3-VL-32B generative judge), Qwen-Image-2.1 text-to-image

- **Kind**: learned, generative (next-token probabilities of a Qwen3-VL judge).
- **Judge**: Qwen/Qwen3-VL-32B-Instruct-FP8 @ 4bf2c2f3, 1024 px input, one batched forward per image.
- **Code**: `vrl/rewards/models/qwen3vl_preference.py`; service `vrl/config/reward_service/qwen3vl32b_preference.yaml`.
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

As a guard inside an HPSv3 composite (per-group z-scores), share of degraded images scoring ≥ the genuine one:

| transform | HPSv3 alone | + 0.25·VLM | + 0.5·VLM | + 1.0·VLM | VLM alone |
|---|---|---|---|---|---|
| noise σ10 | 56% | 55% | 52% | 42% | 31% |
| noise σ25 | 30% | 19% | 13% | 5% | 3% |
| blur r3 | 0% | 0% | 0% | 0% | 3% |

The weight that actually blocks grain (1.0) is the one that costs ranking accuracy (0.694 → 0.642); SigLIP
(`hpsv3_siglip_qwen21_t2i.md`) is the cheaper anti-grain term.

Blind spots to carry forward: fine within-prompt ranking; alignment question saturated (useless as a ranking signal at
this sampler, still a guard against off-prompt collapse).

## Pairwise mode (`DEFAULT_PAIRWISE_CRITERIA`, candidate vs a deterministic base image of the same prompt, both orders)

Service config: `vrl/config/reward_service/qwen3vl32b_pairwise.yaml`; references
`/mnt/nvme/data/external/aes_refs/<set>/<i>.png` named by `metadata.judge_reference` in the prompt jsonl.

**Ready for a training key: yes.**

| gate | result | verdict |
|---|---|---|
| 1 repeatability | deterministic (reference vs itself = 0.500 exactly; pointwise range 0.0) | PASS |
| 2 agreement | directional accuracy vs the same 134 blind pairs: composite **0.649** [0.575, 0.739], overall 0.627, structure 0.597, clean 0.619. HPSv3 0.694 [0.627, 0.776] on pairs that were picked by HPSv3/PickScore top-vs-random, which favors HPSv3 | PASS (above chance; not distinguishable from HPSv3) |
| 3 shortcuts | share of transforms of s0 scoring ≥ the genuine image against the base reference (64 groups, scored through the live service): noise σ10 **19%**, noise σ25 **0%**, blur r3 **0%** (HPSv3: 56% / 30% / 0%) | PASS |
| 4 spread | within-group std of the key **0.120** (overall 0.175, structure 0.094, clean 0.069); raw key mean 0.48, std 0.18, 46% of rollouts beat the base reference | PASS |

Composites on the same pairs (per-group z-scores): HPSv3 + 0.5·pairwise 0.687, + 1.0 0.679, + 2.0 0.664;
HPSv3 + 2.75·SigLIP 0.590 (its own card's blind set gave 0.664); pairwise + pointwise 0.642.
corr(z HPSv3, z pairwise) = 0.37, so the judge ranks on something the CLIP-style rewards do not.

Decision 2026-10-01: stage-1 trains on the pairwise key alone (Seedream 3.0 / RewardDance style, the user's
chosen direction); HPSv3/SigLIP/PickScore are scored offline on checkpoints as diagnostics. Cost: 7.3 s/image on
one L40S, i.e. ~2 min of judge time per 16-rollout round against ~10 min of generation + update.

## Drift shortcuts and the "natural" criteria (2026-10-01 evening)

Stage-1 policies trained on the default pairwise criteria drifted brighter, more saturated and smoother
(`archive/main-20261016:docs/research/qwen21_aesthetic/VLM_STAGE1_RUN.md`), and the user judged later checkpoints "more fake".
New shortcut transforms of s0, scored pairwise against the base reference (64 groups; rate = share of
transforms scoring >= the genuine image):

| criteria set | key agreement (134 pairs) | spread | noise25 | blur3 | bright +18% | sat +35% | smooth | bloom |
|---|---|---|---|---|---|---|---|---|
| default pairwise (overall 2, structure, clean) | 0.649 | 0.120 | 0.0 | 0.0 | -- | -- | -- | -- |
| natural (`qwen3vl32b_pairwise_natural.yaml`: natural 2, fidelity, structure) | 0.604 | 0.100 | 0.047 | 0.0 | **0.375** | **0.453** | 0.016 | **0.469** |

The judge is indifferent to global exposure, saturation and glow changes (rates near 50%) even when the
question names them, so no wording stops that drift. The 134 appeal labels come from AI judges and favor
polish, which explains the lower agreement of the natural set; human labels are being collected
(`/mnt/nvme/aes2/label_page_hidden.json` maps the labelling page's pairs).

Decision at the time: train the natural judge (weight 1) plus a model-free tonal anchor (weight 2): 0 inside a
0.02 band around the base reference's mean luma / saturation, negative outside. Run `vlm_natural_tonal1` started
20:45 from base. The anchor was an experiment-only heuristic and is not part of the codebase; train with
`/reward=qwen3vl_preference_http`.

## Detail criteria (`qwen3vl32b_pairwise_detail.yaml`, 2026-10-02 evening): structure 2, natural 2, detail 1 on 2x2 crops, fidelity 0.5

Quick gates through the live service (6 shared-noise groups vs their base reference, `/mnt/nvme/aes2/qualify_detail_http.py`):

| key | spread | noise25 | blur3 | smooth | bloom | bright | sat |
|---|---|---|---|---|---|---|---|
| composite | 0.064 | 0.00 | 0.00 | 0.00 | 0.33 | 0.67 | 0.67 |
| structure | 0.044 | 0.00 | 0.00 | 0.00 | 1.00 | 0.67 | 0.50 |
| natural | 0.097 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.33 |
| **detail (crops)** | 0.085 | **0.00** | **0.00** | **0.00** | 0.33 | 0.67 | 0.33 |

The crop-judged detail criterion is not fooled by noise, blur or smoothing (the failure modes a "detail" reward
usually has); exposure/saturation/glow remain blind spots of every criterion.
n = 6 groups, so rates are coarse; cost is 10 judge forwards per image (~23 s). In use for `vlm_gptref_detail1`
(GPT references, SDE window [8, 24]).
