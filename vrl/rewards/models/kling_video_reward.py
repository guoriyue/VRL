"""Kling VideoReward model loading."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrl.ray.dependencies import import_from_path
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.videoalign_compat import prepare_videoalign_runtime
from vrl.rewards.ray.model import RewardModel

_DEFAULT_REWARD_MODEL = "KlingTeam/VideoReward"
_DEFAULT_REVISION = "main"
_DEFAULT_MODEL_SUBDIR = "checkpoint-11352"
_VIDEOALIGN_INFERENCE_CLASS = "inference:VideoVLMRewardInference"
_SCORE_KEY_MAP = {
    "overall_reward": "Overall",
    "visual_quality": "VQ",
    "motion_quality": "MQ",
    "text_alignment": "TA",
}
logger = logging.getLogger(__name__)


class KlingVideoRewardModel(RewardModel):
    """One RewardModel: load KlingTeam/VideoReward weights and score a video artifact."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.reward_model_name = str(
            self.worker_config.get("reward_model_name")
            or self.worker_config.get("model_name")
            or "",
        ).strip()
        self.model_root = _resolve_model_root(self.worker_config)
        self.dtype = _torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.use_norm = bool(self.worker_config.get("use_norm", True))
        logger.info(
            "Loading Kling VideoReward model root=%s device=%s dtype=%s use_norm=%s",
            self.model_root,
            self.device,
            self.dtype,
            self.use_norm,
        )
        self._inferencer = self._build_inferencer()

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        prompt = artifact.prompt or str(artifact.metadata.get("prompt", ""))
        if not prompt:
            raise ValueError(
                f"Kling VideoReward requires a prompt for artifact {artifact.artifact_id!r}; "
                "found none on artifact.prompt or artifact.metadata['prompt']",
            )
        artifact_path = str(Path(artifact.path).expanduser().resolve())
        rewards = self._inferencer.reward(
            [artifact_path],
            [prompt],
            use_norm=self.use_norm,
        )
        if not rewards:
            raise RuntimeError("Kling VideoReward returned no scores")
        raw_scores = rewards[0]
        if not isinstance(raw_scores, Mapping):
            raise TypeError("Kling VideoReward must return a mapping of scores")
        return _normalize_scores(raw_scores)

    def _build_inferencer(self) -> Any:
        try:
            cls = import_from_path(_VIDEOALIGN_INFERENCE_CLASS)
        except Exception as exc:
            raise RuntimeError(
                "KlingTeam/VideoReward requires the VideoAlign inference code. "
                "Install https://github.com/KlingAIResearch/VideoAlign on PYTHONPATH "
                "before launching the production reward worker.",
            ) from exc
        disable_flash_attn2 = self.worker_config.get("disable_flash_attn2", None)
        prepare_videoalign_runtime(
            cls,
            disable_flash_attn2=None
            if disable_flash_attn2 is None
            else bool(disable_flash_attn2),
        )
        return cls(
            load_from_pretrained=str(self.model_root),
            device=self.device,
            dtype=self.dtype,
        )


def preflight_kling_video_reward_backend() -> None:
    """Validate model backend code availability without downloading weights."""

    import_from_path(_VIDEOALIGN_INFERENCE_CLASS)


def _resolve_model_root(worker_config: Mapping[str, Any]) -> Path:
    model_path = str(worker_config.get("model_path", "")).strip()
    if model_path:
        root = Path(model_path).expanduser().resolve()
    else:
        reward_model_name = str(
            worker_config.get("reward_model_name")
            or worker_config.get("model_name")
            or "",
        ).strip()
        if not reward_model_name:
            raise ValueError("Kling VideoReward requires worker_config.reward_model_name")
        repo_id, revision = _split_reward_model_name(reward_model_name)
        if repo_id != _DEFAULT_REWARD_MODEL:
            raise ValueError(
                "Kling VideoReward loader currently supports only "
                f"{_DEFAULT_REWARD_MODEL!r}, got {repo_id!r}",
            )
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo_id=repo_id, revision=revision)).resolve()

    if not (root / "model_config.json").exists():
        parent = root.parent
        if root.name.startswith("checkpoint-") and (parent / "model_config.json").exists():
            root = parent
    if not root.exists():
        raise FileNotFoundError(f"Kling VideoReward model path does not exist: {root}")
    if not (root / "model_config.json").exists():
        raise FileNotFoundError(
            "Kling VideoReward model root must contain model_config.json: "
            f"{root}",
        )
    if not (root / _DEFAULT_MODEL_SUBDIR).exists():
        raise FileNotFoundError(
            "Kling VideoReward model root must contain "
            f"{_DEFAULT_MODEL_SUBDIR}: {root}",
        )
    return root


def _split_reward_model_name(value: str) -> tuple[str, str]:
    if "@" not in value:
        return value, _DEFAULT_REVISION
    repo_id, revision = value.rsplit("@", 1)
    return repo_id, revision or _DEFAULT_REVISION


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported Kling VideoReward dtype: {name!r}")


def _normalize_scores(raw_scores: Mapping[str, Any]) -> dict[str, float]:
    raw = {str(key): float(value) for key, value in raw_scores.items()}
    scores = dict(raw)
    for public_key, model_key in _SCORE_KEY_MAP.items():
        if model_key in raw:
            scores[public_key] = float(raw[model_key])
    return scores


__all__ = [
    "KlingVideoRewardModel",
    "preflight_kling_video_reward_backend",
]
