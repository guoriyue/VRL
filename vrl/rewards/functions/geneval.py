"""GenEval reward (model-backed over the local transport).

Scores rollouts against structured GenEval prompt metadata. The actual
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
from vrl.rewards.runtime import LocalRewardRuntime
from vrl.rewards.types import RewardRollout


class GenEvalReward(RewardFunction):
    """Score rollouts against GenEval prompt metadata.

    The built-in mode delegates actual image/object scoring to an import-path
    callable. This keeps the training stack independent from the GenEval repo
    layout while preserving the exact prompt metadata needed by the evaluator.
    """

    def __init__(
        self,
        device: str = "cuda",
        import_path: str = "",
        timeout_s: float = 120.0,
        debug_dir: str = "",
        artifact_dir: str = "",
        scorer: Callable[..., Any] | None = None,
        **_unused: Any,
    ) -> None:
        # ``_unused`` absorbs stray reward.kwargs keys (e.g. a legacy
        # ``evaluator: import_path``); the backend is decided by whether a
        # scorer is injected, otherwise the import_path — there is no knob.
        self.device = device
        self.import_path = import_path
        self.timeout_s = float(timeout_s)
        self.artifact_dir = artifact_dir
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
        self._model = model
        super().__init__(
            reward_name="geneval",
            score_key="geneval",
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=self._build_artifacts,
            debug_dir=debug_dir,
            request_prefix="geneval",
            debug_basename="geneval",
        )

    def _build_artifacts(
        self,
        rollouts: list[RewardRollout],
    ) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        for index, rollout in enumerate(rollouts):
            rollout_metadata = dict(rollout.metadata or {})
            policy_version = rollout_metadata.get("policy_version")
            geneval = self._extract_geneval_metadata(rollout)
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=f"geneval-{index}",
                    path="",
                    media_type="image",
                    media=_OutputBox(rollout.trajectory.output),
                    prompt=str(rollout.trajectory.prompt),
                    sample_id=f"sample-{index}",
                    policy_version=None if policy_version is None else int(policy_version),
                    metadata={
                        "geneval": geneval,
                        "rollout_metadata": rollout_metadata,
                    },
                ),
            )
        return artifacts

    @staticmethod
    def _extract_geneval_metadata(rollout: RewardRollout) -> dict[str, Any]:
        metadata = dict(rollout.metadata)
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

        raise ValueError("GenEvalReward requires metadata.geneval on each rollout")


__all__ = ["GenEvalReward"]
