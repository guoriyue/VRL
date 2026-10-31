# Layering reward review snapshot — 2026-10-31

This branch captures the code used for the layering experiments, including earlier uncommitted diagnostics and the current additive-excess ablation. It is an experimental review snapshot, not a demonstrated quality improvement.

## Review order

1. `vrl/rewards/models/layered_grouping.py` and `tests/rewards/test_layered_*.py`: optimal assignment, annotation-only fragment accounting, partition scoring, complete-object integrity, and independent excess-opacity penalty.
2. `vrl/rewards/models/layered_removal.py`: object-deletion diagnostics, including their limitations.
3. `vrl/config/schema.py`, `vrl/trainers/distributed.py`: configurable collective timeout for unequal rollout durations.
4. `tools/layering_*` and layering presets: accumulated experiment preparation, validation, reporting and launch tools. Some are historical recipes, not the active run.
5. This directory: exact active ablation command/config, rollout checks, and completed control results.

## Current ablation

Pure GRPO from checkpoint 28 with reset Adam; seven updates, four prompts across four GPUs, four candidates per prompt, 30 sampling steps, learning rate 5e-5, Adam epsilon 1e-12 and KL coefficient 0.04. No SFT; empty layers are allowed.

The only recipe change relative to the reviewed-annotation control is `excess_weight=1.0`:

```
reward = partition + fidelity + reference_integrity - excess_foreground_opacity
```

Completion weight is zero. Excess opacity is the positive per-pixel difference between summed predicted foreground alpha and summed exhaustive truth alpha, normalized by truth foreground mass. It requires `complete_layer_truth: true`. Visible-only photographs receive no excess-opacity or integrity supervision. This term cannot guarantee semantic completeness or detect every duplicate, and it can penalize legitimate content absent from incomplete references. Never mark a visible mask as complete truth.

The existing reference-integrity term already includes a multiplicative excess-opacity penalty. This experiment tests an additional independent penalty; it does not replace that term. Default `excess_weight=0` preserves the pre-ablation reward output. Earlier assignment and fragment-accounting fixes do change the original main-branch reward behavior.

## Evidence and status

At snapshot preparation, six of seven updates had completed; final held-out evaluation was still pending. Do not interpret implementation validation as model improvement.

- 41 relevant reward and distributed-training tests passed.
- Ruff passed for changed/untracked Python files; `git diff --check` passed.
- All four production preflights had zero measured rollout/replay log-probability drift and exact artifact/direct reward agreement.
- The first 16 saved candidate stacks exactly match the control before any update; original score terms are unchanged. Twelve complete-reference candidates receive the additional penalty. Saved PNG equality is weaker than latent equality.
- The prior annotation pair improved a small controlled cohort but regressed on real COCO photographs. Neither prior arm justified long training.
- Layer-budget reports show that more layers reduce some background omissions while increasing fragmentation. Their threshold counts and pixel edit metrics are not semantic ground truth.

## Review on another machine

The repository source and tests are portable; experiment artifacts retain the exact original `/mnt/nvme/...` and `/home/ubuntu/VRL-layering/...` paths for provenance. `run_ablation.py` and historical tools are archived machine-specific launchers: inspect and remap their paths before executing. This branch does not include model weights, checkpoints, training images, annotations, rollout PNGs, external evaluation scripts, or the large frozen-input archive. Their locations are recorded in the protocol/runner. Do not launch the archived runner unchanged on another machine.

The run evaluates the exported live adapter, verifies checkpoint equality and optimizer steps, then uses unchanged scoring for 34 main cases and 15 controlled layouts from five source assets. Longer training remains conditional on held-out structural and edit-quality improvement.

The repository-requested `.claude/skills/review/` skill was unavailable on this host. Source inspection, tests, lint, and runtime checks were used for self-review; no independent reviewer is claimed.
