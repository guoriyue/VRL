---
name: reward
description: The offline reward toolkit (`python -m reward`, `vrl.scripts.rewards.rescore_media`) -- score existing media independently of training, and qualify a reward for RL gate by gate. Use when adding or choosing a reward, before launching a GRPO run on a reward that has not passed the gates, when a curve rises but held-out judgments do not, or when you need scores for generated outputs without a trainer.
---

# Reward toolkit

Everything here works on an `Evaluation`: one scoring run of existing media,
written by `rescore_media`, content-hashed and resumable. Pick the sub-skill for
the job and read it before running commands:

| Need | Read | Commands |
| --- | --- | --- |
| Scores for existing images/videos, locally or through the service training uses | `scoring.md` | `rescore_media`, `health`, `compare`, `paired`, `repeat` |
| Decide whether a reward may enter a training key | `qualification.md` | `repeat`, `agreement`, `stress-manifest` / `shortcut-manifest` + `stress`, `spread` |

Ground rules that hold across both:

- Evidence about a reward comes from **the policy's own outputs** (generated under
  the training sampler) judged **blind** by people or independent judges -- never
  from dataset reference images, and never from the reward under test.
- A reward is qualified **per task and per axis**. A composite is qualified on its own.
- Report a judge's blind spots next to its strengths (per-contrast AUC), keep
  failing rewards as logged observations, and put nothing in a training key that
  has not passed gates 1-4.
- After training, "reward up, blind verdicts flat" means the reward is being gamed:
  stop, add the exploit as a shortcut transform, re-run the gates.
- Do not hand-write task formulas over model outputs as rewards (the object-move
  lesson); combine released judges, and qualify the combination.

Design and rationale: `docs/sprints/planned/SPRINT_reward_qualification.md`;
full command reference: `docs/rewards_offline_evaluation.md`.
