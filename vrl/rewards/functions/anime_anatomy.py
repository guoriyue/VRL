"""OpenPose-style anime anatomy structure reward (model-backed over the local transport)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.anime_anatomy_model import (
    AnimeAnatomyStructureRewardModel,
    _onnx_providers,
)
from vrl.rewards.runtime import LocalRewardRuntime

# ``_onnx_providers`` is re-exported for callers/tests that resolved ONNX
# provider selection through this module before the model was extracted.
__all__ = ["AnimeAnatomyStructureReward", "_onnx_providers"]


class AnimeAnatomyStructureReward(RewardFunction):
    """Reward anime images using whole-body OpenPose-style keypoint geometry."""

    def __init__(
        self,
        device: str = "cuda",
        model_repo: str = "yzd-v/DWPose",
        detector_file: str = "yolox_l.onnx",
        pose_file: str = "dw-ll_ucoco_384.onnx",
        cache_dir: str = "",
        local_files_only: bool = False,
        detector: Callable[[list[Any]], list[Any]] | None = None,
        detect_resolution: int = 512,
        min_keypoint_confidence: float = 0.25,
        min_body_keypoints: int = 8,
        require_hands: str | bool = "prompt",
        require_feet: str | bool = "prompt",
        debug_dir: str | None = None,
        inference_runtime: str = "local",
    ) -> None:
        if str(inference_runtime) != "local":
            raise ValueError(
                "anime_anatomy_structure reward currently supports "
                "inference_runtime='local' only",
            )
        # Build eagerly so config validation fires now and the injected detector
        # is wired into the in-process model.
        model = AnimeAnatomyStructureRewardModel(
            {
                "device": device,
                "model_repo": model_repo,
                "detector_file": detector_file,
                "pose_file": pose_file,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "detector": detector,
                "detect_resolution": detect_resolution,
                "min_keypoint_confidence": min_keypoint_confidence,
                "min_body_keypoints": min_body_keypoints,
                "require_hands": require_hands,
                "require_feet": require_feet,
                "debug_dir": debug_dir,
            },
        )
        self._model = model
        super().__init__(
            reward_name="anime_anatomy_structure",
            score_key="anime_anatomy_structure",
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )

    @property
    def last_diagnostics(self) -> list[dict[str, Any]]:
        """Per-image pose diagnostics from the most recent batch."""

        return self._model.last_diagnostics
