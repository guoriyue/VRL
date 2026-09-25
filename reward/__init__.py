"""Offline reward development: analyze, calibrate and qualify rewards without training.

This package sits beside ``vrl`` and depends on it; ``vrl`` never imports it.
It consumes scoring runs written by ``vrl.scripts.rewards.rescore_media``
(``vrl.rewards.evaluation.Evaluation``) and produces reports, fitted
combinations and deployment receipts. Training only reads the results through
``vrl.rewards.deployment.RewardDeployment``.

* ``analysis.Analysis``: health, stress, sequence, ranking, paired and repeatability reports.
* ``calibration.Calibration``: preference fitting, application, holdout evaluation, review packets.
* ``stress.build_stress_manifest``: deterministic perturbations for stress audits.
* ``python -m reward``: one command-line entry point for all of the above.
"""
