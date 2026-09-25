"""Multi-step image editing above the ``vrl`` framework: edit chains.

This package sits beside ``vrl`` and depends on it; ``vrl`` never imports it.
A chain is one source image and an ordered list of editing instructions. The
same loop runs it for evaluation (a frozen editor, a reward judge) and for
training (sample groups per step through ``vrl``'s collector, trained by
``vrl``'s one-shot trainer).

* ``chains``: ``EditChain`` (``load_manifest``, ``run``), ``ChainRun`` (the run
  record; ``export`` for offline rescoring, ``evaluation``), the ``Editor`` /
  ``Judge`` protocols, and ``GroupEditor`` for training.
* ``roles``: ``LocalEditor`` and ``RewardJudge`` for evaluation.
* ``scripts``: ``train_edit_chains``, ``evaluate_chains``, ``export_chain_media``.

Consumers import the concrete modules directly; this facade re-exports nothing.
"""
