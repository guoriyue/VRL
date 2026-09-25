---
name: reward
description: The offline reward toolkit (`python -m reward`, `vrl.scripts.rewards.rescore_media`) -- score existing media independently of training, qualify a reward for RL gate by gate and write its reward card, calibrate a combination of axes on blind preferences and bind it to training with a receipt. Use when adding or choosing a reward, before launching a GRPO run on a reward without a card, when a curve rises but held-out judgments do not, or when you need scores for generated outputs without a trainer.
---

# Reward toolkit

Everything here works on an `Evaluation`: one scoring run of existing media,
written by `rescore_media`, content-hashed and resumable. Pick the sub-skill for
the job and read it before running commands:

| Need | Read | Commands |
| --- | --- | --- |
| Scores for existing images/videos, locally or through the service training uses | `scoring.md` | `rescore_media`, `health`, `compare`, `paired`, `repeat` |
| Decide whether a reward may enter a training key; write its card | `qualification.md` | `repeat`, `agreement`, `stress-manifest` / `shortcut-manifest` + `stress`, `spread`, `card` |
| Fit a combination of axes on blind human preferences; bind training to validated scoring | `calibration.md` | `review-export` / `review-import`, `fit`, `evaluate`, `apply`, `qualify` |

Ground rules that hold across all three:

- Evidence about a reward comes from **the policy's own outputs** (generated under
  the training sampler) judged **blind** by people or independent judges -- never
  from dataset reference images, and never from the reward under test.
- A reward is qualified **per task and per axis**. A composite gets its own card.
- Report a judge's blind spots next to its strengths (per-contrast AUC), keep
  failing rewards as logged observations, and put nothing in a training key that
  has not passed gates 1-4.
- After training, "reward up, blind verdicts flat" means the reward is being gamed:
  stop, add the exploit as a shortcut transform, re-run the card.
- Do not hand-write task formulas over model outputs as rewards (the object-move
  lesson); combine released judges, and qualify the combination.

Design and rationale: `docs/sprints/planned/SPRINT_reward_qualification.md`;
full command reference: `docs/rewards_offline_evaluation.md`.
