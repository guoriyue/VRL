"""GenEval-style compositional reward with an OWL detector."""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class GenEvalOwlReward(DiskArtifactRewardFunction):
    """GenEval-style compositional reward with an OWL detector.

    Model paths and dtype come from YAML; this binding pins the factory and
    the transport: in-process the runtime builds the model on the resolved
    device (CuMem-pooled under a shared GPU), ``inference.kind=service`` hands
    the same worker_config to a driver-launched service.
    """

    model_factory = "vrl.rewards.models.geneval_owl:GenEvalOwlRewardModel"
    request_prefix = "geneval_owl"
    debug_basename = "geneval_owl"
    default_reward_name = "geneval_owl"
    default_score_key = "geneval_owl_dense"
    score_keys = ("geneval_owl_dense", "geneval_owl_partial", "geneval_owl_strict")
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"


__all__ = ["GenEvalOwlReward"]
