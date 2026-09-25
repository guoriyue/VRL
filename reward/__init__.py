"""Offline reward development: analyze, calibrate and qualify rewards without training.

This package sits beside ``vrl`` and depends on it; ``vrl`` never imports it.
It consumes scoring runs written by ``vrl.scripts.rewards.rescore_media``
(``vrl.rewards.evaluation.Evaluation``) and produces reports, fitted
combinations and deployment receipts. Training only reads the results through
``vrl.rewards.deployment.RewardDeployment``.

* ``analysis.Analysis``: health, stress, sequence, ranking, paired and repeatability reports.
* ``calibration.Calibration``: preference fitting, application, holdout evaluation, review packets.
* ``stress.build_stress_manifest``: deterministic perturbations for stress audits.
* ``shortcuts.build_shortcut_manifest``: task-avoiding candidates (unchanged source,
  shifted or re-cropped frame, another scene) for edit rewards, paired like stress.
* ``labels.agreement``: per-contrast AUC of one axis against blind categorical labels.
* ``Analysis.spread``: success band and within-prompt spread under the training sampler.
* ``card.build_card``: the reward card -- every qualification gate's verdict in one place.
* ``python -m reward``: one command-line entry point for all of the above.
* ``skills/reward/``: the toolkit as a Claude Code skill set -- ``SKILL.md`` routes to
  ``scoring.md``, ``qualification.md`` and ``calibration.md`` (``.claude/skills/reward`` links here).
"""
