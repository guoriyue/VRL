# Qwen-Image-2.1 RL session, 2026-10-08 to 2026-10-10

Three lines of work on Qwen-Image-2.1 with Qwen3-VL-32B judges: the GPT-reference aesthetic runs (v13-v15),
nudity-removal pilots, and close multi-person limb-integrity RL. All runs used 4x L40S (three FSDP ranks,
judge service on the fourth GPU), Flow-GRPO with shared initial noise per group of 8, SDE window [0, 6],
LoRA rank 32, pipelined rollout and `lookahead_next_group`. Run outputs live under
`/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/` on the training host and are not in the repository.

## 1. Aesthetic runs against GPT references (v13-v15)

Judge: `docs/research/qwen21_session_20261008/judge_configs/qwen3vl32b_pairwise_three.yaml` (commits 787c1c857 .. 02f649020).

- `correct` (weight 2): absolute yes/no anatomy-defect question, inverted.
- `stray_text` (1): unrequested text, inverted. Added back after a probe showed `like_ref` ignores an added
  signature line.
- `single_scene` (2): panels, sticker sheets, grids, or the main subject repeated small around one large copy.
  Added after v13 ck40 and v14 ck20 drifted into exactly that.
- `accurate` (1): per-item yes/no over an objects-and-actions checklist (style words removed; they diluted
  the score, e.g. a child figure outscored the actual Capitol building).
- `like_ref` (1): group criterion -- siblings of one prompt compared pairwise (ring offsets 1 and 3, both
  orders) with the GPT image shown as Image 3; score = win probability. An absolute 1-5 version gave the GPT
  image itself 3/5 and spread siblings by 0.02.

| run | prompts | best checkpoint | failure after it |
|---|---|---|---|
| v13 | 846 GPT-reference prompts, objects checklist | ck30 | ck40: over-cute symmetric chibi, giant-hair girl instead of the Capitol, mandala cat |
| v14 | same prompts rewritten by GPT (camera, pose, layering, idioms resolved) | ck10-ck20 | ck20: turtle sticker sheet, 3x3 expression grid |
| v15 | v14 + `single_scene` | ck20 | ck30: a 3x3 grid on the Capitol prompt (whole group was grids, no contrast) |

Conclusion: RL with this judge removes defects that appear in some of the base's 8 samples (extra limbs, stray
text, collages) and then exploits the judge after 20-40 rounds. It does not teach what the base never samples:
named styles (Beatrix Potter), material construction (a golem built from recognisable candy), world knowledge
(Capitol Hill), or unstated objects (the Suzuki). GPT prompt rewriting at inference closes most of that gap.

## 2. Nudity removal (ForceForget-style) pilots

Prompts: I2P `sexual` subset filtered to nudity >= 20% and no youth keywords; a strict second Codex audit
then removed 48 of 414 prompts that named real people or referred to characters who are minors in canon
(the first audit had missed many). Only the cleaned sets were used afterwards (326 train / 46 held-out).
Real-person and minor-coded prompts must stay out of rollouts even for suppression training.

| pilot | reward | held-out result (46 clean prompts) |
|---|---|---|
| 1 | Falconsai NSFW classifier x2 + PickScore(`safe_prompt`) | failed: the classifier scored 0 on most Qwen nudity, penalty fired once in 10 rounds |
| 2 | NudeNet exposed-part score x2 + PickScore(`safe_prompt`) x0.5 (commits 2d8a20b4c, 28f01d833) | exposed images 7 -> 3 (NudeNet), 10 -> 4 (independent Qwen3-VL); benign prompts went grey and lost subjects by ck10 |
| 3 | Qwen3-VL judge: no_nudity x2, faithful-to-`safe_prompt`, correct (`qwen3vl32b_nudity_erase.yaml`) on the uncensored GGUF | stopped at round 5: the I2P prompts trigger nudity only ~15-22% of the time on Qwen, too weak a signal |

- `0xSojalSec/Qwen-Image-2.1-Uncensored-HF`: text encoder and VAE identical to upstream; only the 128 attention
  projections differ, by 1e-4 to 4e-4 relative. On the 46 clean prompts it was not more nude than upstream.
  `scripts/data/convert_uc_gguf_to_diffusers.py` reproduces the local diffusers directory.
- The Danbooru adult-only safety manifests (`manifests/danbooru/safety/*_c_adherence.jsonl`) were screened
  before any training: 93% of 600 base renders were nude, but Qwen3-VL judged 24% to depict someone who
  looks under 18 (anime styling; youth correlates with twintails, ribbons, bows, simple backgrounds). They
  were not used for training. Every NSFW image set must pass an apparent-age screen (`scripts/judge_eval/vlm_screen.py`).

## 3. Close multi-person limb-integrity RL

Corpus (`data/limb_corpus/`): 573 Codex-written prompts (2/3/4/6/8 adults, eight contact interaction types,
eight scene families, photo and anime), balanced by count x interaction, no human-shaped props or limb-like
objects; 64 held-out prompts whose actions never appear in training. Judge: one absolute inverted question
(`docs/research/qwen21_session_20261008/judge_configs/qwen3vl32b_limb.yaml`, commit f953078d5). Run `limb_rl_v2`, lr 1e-4, 54 rounds.

Held-out, 64 prompts x 1 sample, all scores from Qwen3-VL-32B:

| version | limb defect | exact people count | action matches prompt | all three ok |
|---|---|---|---|---|
| base | 33% | 73% | 95% | 50% |
| ck10 | 20% | 81% | 89% | 59% |
| ck20 | 11% | 78% | 86% | **62%** |
| ck30 | 3% | 58% | 55% | 33% |
| ck50 | 3% | 47% | 41% | 28% |

The limb-defect drop is mostly bought by dropping the requested contact: after ck20 the model lines people
up side by side, removes the lift or embrace, draws fewer people for 6-8 person prompts and gives everyone
the same outfit. Two-person prompts lose the action too (ck50: 50% action match). The only real gain is
ck10-ck20 (50% -> 62% all-three-ok, inside the noise of 64 images for single groups). The evaluation judge is
the reward model asking a near-identical question, so the limb numbers are optimistic.

Next run: reward = limbs x exact count x action fidelity (multiplicative), evaluate all three with 4 samples
per prompt and a question phrased differently from the reward.

## Files

- `scripts/judge_eval/`: limb, action and count evaluation (`vlm_limb_eval.py`, `vlm_action_eval.py`),
  NudeNet/VLM nudity evaluation, the apparent-age screen, limb scoring and the live-judge rescore helper.
- `scripts/chains/`: launch and pause-evaluate-resume chains for v13-v15, the nudity pilots and the limb run;
  `build_series_page.py` builds the checkpoint comparison strips; `prune_parking.sh` removes stale trainer
  parking directories (46k of them, 820 GB, filled the NVMe on 2026-10-09).
- `scripts/data/`: GPT reference generation and audit through Codex, prompt rewriting, checklist expansion,
  the Danbooru `safe_prompt` rewrite and the GGUF conversion.
- `data/limb_corpus/`: train and held-out prompts, the cell tables and the Codex task templates.
