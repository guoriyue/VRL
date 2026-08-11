"""Concrete RewardModel implementations for reward runtimes.

The models here are the scoring networks behind the thin, config-facing
classes in ``vrl.rewards.functions``. They are never constructed from YAML
directly: the reward scorer resolves each ``model_factory`` dotted path
(``vrl.rewards.models.<module>:<Class>``) and builds the model on the
resolved device inside its owning memory frame, so a reward FUNCTION (name,
weight, score-key selection, transport) and a reward MODEL (the network)
keep separate owners and lifecycles.
"""
