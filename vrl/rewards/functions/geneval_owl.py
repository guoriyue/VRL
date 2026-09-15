"""GenEval-style compositional reward with an OWL detector."""

from __future__ import annotations

from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import MediaType
from vrl.rewards.base import DiskArtifactRewardFunction
from vrl.rewards.protocols import RewardScorer

GENEVAL_SCORE_KEYS = ("geneval_owl_dense", "geneval_owl_partial", "geneval_owl_strict")


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
    default_artifact_format = "tensor"
    default_media_type = "image"
    in_process_media = "memory"

    def __init__(
        self,
        device: str = "cuda",
        score_key: str = "geneval_owl_dense",
        scorer: RewardScorer | None = None,
        inference: RewardInferenceConfig | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        retain_artifacts: bool = False,
        **kwargs: Any,
    ) -> None:
        if score_key not in GENEVAL_SCORE_KEYS:
            raise ValueError(
                f"geneval_owl score_key must be one of {list(GENEVAL_SCORE_KEYS)}, got {score_key!r}"
            )
        worker_config = {
            **kwargs,
        }
        super().__init__(
            reward_name="geneval_owl",
            score_key=score_key,
            worker_config=worker_config,
            device=device,
            scorer=scorer,
            inference=inference,
            artifact_format=artifact_format,
            media_type=media_type,
            artifact_dir=artifact_dir,
            retain_artifacts=retain_artifacts,
        )


__all__ = ["GENEVAL_SCORE_KEYS", "GenEvalOwlReward"]
