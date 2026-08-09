"""GenEval reward (model-backed over the in-process transport).

Scores samples against structured GenEval prompt metadata. The actual
image/object scoring is delegated to an import-path callable (or an injected
``scorer``), keeping the training stack independent from the GenEval repo
layout while preserving the exact prompt metadata the evaluator needs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.geneval import GenEvalRewardModel, _OutputBox
from vrl.rewards.runtime import InProcessRewardInferenceRuntime
from vrl.rewards.types import RewardSample


class GenEvalReward(RewardFunction):
    """Score samples against GenEval prompt metadata.

    The built-in mode delegates actual image/object scoring to an import-path
    callable. This keeps the training stack independent from the GenEval repo
    layout while preserving the exact prompt metadata needed by the evaluator.
    """

    def __init__(
        self,
        device: str = "cuda",
        import_path: str = "",
        debug_dir: str = "",
        artifact_dir: str = "",
        scorer: Callable[..., Any] | None = None,
    ) -> None:
        # No backend knob: the model uses the injected ``scorer`` if given,
        # otherwise resolves the ``import_path`` callable. An unknown reward.kwargs
        # key is a typo and fails loud here (the __init__ is the per-reward
        # validation boundary), same as ocr.py's explicit signature.
        # Build eagerly so an injected scorer and the import_path are wired
        # before the first score call.
        model = GenEvalRewardModel(
            {
                "device": device,
                "import_path": import_path,
                "debug_dir": debug_dir,
                "artifact_dir": artifact_dir,
                "scorer": scorer,
            },
        )
        super().__init__(
            reward_name="geneval",
            score_key="geneval",
            runtime=InProcessRewardInferenceRuntime(model=model),
            artifact_builder=self._build_artifacts,
            debug_dir=debug_dir,
            request_prefix="geneval",
            debug_basename="geneval",
        )

    def _build_artifacts(
        self,
        samples: list[RewardSample],
    ) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        for sample in samples:
            sample_metadata = dict(sample.metadata or {})
            geneval = self._extract_geneval_metadata(sample)
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=(f"{sample.source_request_id}:{sample.sample_id}:geneval"),
                    path="",
                    media_type="image",
                    media=_OutputBox(sample.output),
                    prompt=str(sample.prompt),
                    source_request_id=sample.source_request_id,
                    sample_id=sample.sample_id,
                    group_id=sample.group_id,
                    trajectory_id=sample.trajectory_id,
                    policy_version=sample.policy_version,
                    metadata={
                        "geneval": geneval,
                        "rollout_metadata": sample_metadata,
                    },
                ),
            )
        return artifacts

    @staticmethod
    def _extract_geneval_metadata(sample: RewardSample) -> dict[str, Any]:
        metadata = dict(sample.metadata)
        geneval = metadata.get("geneval")
        if isinstance(geneval, dict):
            return geneval

        manifest_row = metadata.get("manifest_row")
        if isinstance(manifest_row, dict):
            row_metadata = manifest_row.get("metadata")
            if isinstance(row_metadata, dict) and isinstance(row_metadata.get("geneval"), dict):
                return dict(row_metadata["geneval"])
            if isinstance(manifest_row.get("geneval"), dict):
                return dict(manifest_row["geneval"])
            if "tag" in manifest_row and "include" in manifest_row:
                return {
                    key: manifest_row[key]
                    for key in ("tag", "include", "exclude")
                    if key in manifest_row
                }

        raise ValueError("GenEvalReward requires metadata.geneval on each sample")


__all__ = ["GenEvalReward"]
