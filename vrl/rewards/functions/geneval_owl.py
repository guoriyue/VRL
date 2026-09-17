"""GenEval-style compositional reward with an OWL detector."""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class GenEvalOwlReward(ModelRewardFunction):
    """GenEval-style compositional reward with an OWL detector."""

    model_factory = "vrl.rewards.models.geneval_owl:GenEvalOwlRewardModel"
    name = "geneval_owl"
    default_score_key = "geneval_owl_dense"
    score_keys = ("geneval_owl_dense", "geneval_owl_partial", "geneval_owl_strict")
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["GenEvalOwlReward"]
