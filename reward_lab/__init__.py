"""Offline reward development: analyze, calibrate and qualify rewards without training.

This package sits beside ``vrl`` and depends on it; ``vrl`` never imports it.
It consumes scoring runs written by ``vrl.scripts.rewards.rescore_media`` and
produces reports, fitted combinations and deployment receipts. Training only
reads the results through ``vrl.rewards.calibration.FrozenRewardCombination``
and ``vrl.rewards.deployment.load_reward_deployment``.

* ``diagnostics``: health, stress, ranking, paired and repeatability reports.
* ``calibration``: preference pairs, logistic fitting, application, holdout evaluation.
* ``qualification``: measure offline-versus-runtime parity and write the receipt.
* ``annotation``: blinded preference review packets and answer import.
* ``scripts``: ``analyze_scores``, ``calibrate_scores``, ``stress_media``.
"""
