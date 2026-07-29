"""Robotics video reward blending semantics, target fidelity, and motion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel
from vrl.rewards.models.motion_dynamics import MotionDynamicsModel
from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel


@dataclass(frozen=True, slots=True)
class RoboticsRewardWeights:
    """Weights for the three independently audited robotics reward signals."""

    target_dino_similarity: float = 1.0
    motion_dynamics: float = 0.2
    kling_text_alignment: float = 0.3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RoboticsRewardWeights:
        payload = dict(value or {})
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unsupported robotics reward weights: {unknown}")
        weights = cls(**{key: float(raw) for key, raw in payload.items()})
        observed = {item.name: float(getattr(weights, item.name)) for item in fields(cls)}
        if any(not math.isfinite(weight) or weight < 0.0 for weight in observed.values()):
            raise ValueError(
                f"robotics reward weights must be finite and non-negative: {observed}"
            )
        if not any(weight > 0.0 for weight in observed.values()):
            raise ValueError("at least one robotics reward weight must be positive")
        return weights


class RoboticsVideoRewardModel:
    """One service-owned model combining three complementary video signals.

    Kling contributes only prompt alignment: its visual, motion, and aggregate
    axes ranked DROID color blobs and static clips above real targets. DINOv2
    supplies the target-conditioned visual/temporal anchor, while RAFT supplies
    the anti-static motion floor. Keeping all three behind one HTTP model lets
    generation continue on disjoint GPUs while this service scores artifacts.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        device = str(cfg.get("device", "cuda:0"))
        data_root = str(cfg.get("data_root", ""))
        self.weights = RoboticsRewardWeights.from_mapping(cfg.get("weights"))

        kling_config = _child_config(cfg, "kling")
        kling_config.setdefault("device", device)
        dino_config = _child_config(cfg, "dino")
        dino_config.setdefault("device", device)
        if data_root:
            dino_config.setdefault("data_root", data_root)
        motion_config = _child_config(cfg, "motion")
        motion_config.setdefault("device", device)

        self.kling = KlingVideoRewardModel(kling_config)
        self.dino = TargetDinoSimilarityModel(dino_config)
        self.motion = MotionDynamicsModel(motion_config)

    def prepare_for_inference(self) -> None:
        """Materialize every lazy child while the composite CuMem pool owns CUDA."""

        for child in (self.kling, self.dino, self.motion):
            prepare = getattr(child, "prepare_for_inference", None)
            if callable(prepare):
                prepare()

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        kling = dict(self.kling(artifact=artifact, request=request))
        dino = dict(self.dino(artifact=artifact, request=request))
        motion = dict(self.motion(artifact=artifact, request=request))
        selected = {
            "target_dino_similarity": float(dino["target_dino_similarity"]),
            "motion_dynamics": float(motion["motion_dynamics"]),
            "kling_text_alignment": float(kling["text_alignment"]),
        }
        if any(not math.isfinite(value) for value in selected.values()):
            raise ValueError(f"robotics reward produced non-finite component scores: {selected}")
        blend = (
            self.weights.target_dino_similarity * selected["target_dino_similarity"]
            + self.weights.motion_dynamics * selected["motion_dynamics"]
            + self.weights.kling_text_alignment * selected["kling_text_alignment"]
        )
        return {
            "robotics_blend": float(blend),
            **selected,
            "kling_visual_quality": float(kling["visual_quality"]),
            "kling_motion_quality": float(kling["motion_quality"]),
            "kling_overall_reward": float(kling["overall_reward"]),
        }


def _child_config(worker_config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = worker_config.get(name) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"robotics reward worker_config.{name} must be a mapping")
    return dict(value)


__all__ = ["RoboticsRewardWeights", "RoboticsVideoRewardModel"]
