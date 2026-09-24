"""Agentic visual control: a bounded editing environment and a trainable controller.

This package sits beside the ``vrl`` framework and depends on it; ``vrl`` never
imports it. It owns episode state, the finite action vocabulary, transitions
through a frozen editor, termination, credit assignment, the Qwen3-VL controller
policy and its on-policy trainer. Rewards stay an independent role (``Judge``)
supplied by ``vrl.rewards``; one-shot collection and editor training stay in
``vrl``.

* ``episode``: ``Task``/``Observation``/``Decision`` records, the three role
  protocols, and ``Episode``, which runs one episode and rebuilds training steps.
* ``roles``: ``LocalEditor`` (frozen family model) and ``RewardJudge``.
* ``controller``: the categorical Qwen3-VL policy and its replay records.
* ``trainer``: on-policy controller updates over episode groups.
* ``export``: an episode's images as a media manifest for offline rescoring.
* ``scripts``: collect, train, compare, probe, scripted sequences, media export.

Consumers import the concrete modules directly; this facade re-exports nothing.
"""
