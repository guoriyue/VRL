"""Reward model contract and shared base for in-process torch reward models.

``RewardModel`` is the scoring contract: given one already-materialized
artifact plus the request, return named scores. ``TorchRewardModel``
implements it and absorbs the device/dtype/lazy-load boilerplate that every
torch-nn reward used to hand-roll. Subclasses implement ``_load`` (build the
model once) and ``score_media`` (score one media payload + prompt). Media is
pulled via ``artifact.as_media()``, so the same model scores an in-memory
tensor or a materialized disk artifact (``.pt`` / ``.mp4`` path).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest


def require_prompt_and_video_path(
    artifact: RewardInferenceArtifact,
    *,
    family: str,
) -> tuple[str, str]:
    """Shared ``__call__`` prelude for caption-conditioned video judges.

    Fails fast when the artifact carries no prompt — the consumers' rubrics
    are caption-conditioned, so an empty prompt would silently score garbage.
    Returns ``(prompt, resolved_media_path)``.
    """

    prompt = artifact.prompt or str(artifact.metadata.get("prompt", ""))
    if not prompt:
        raise ValueError(
            f"{family} requires a prompt for artifact {artifact.artifact_id!r}; "
            "found none on artifact.prompt or artifact.metadata['prompt']",
        )
    video_path = str(Path(artifact.as_path()).expanduser().resolve())
    return prompt, video_path


class RewardModel(Protocol):
    """A reward model the runtime loads and runs.

    The runtime is model-agnostic: any model that implements this protocol and
    is named by ``worker_config.model_factory`` can be loaded. Concrete models
    live under ``vrl.rewards.models``.
    """

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> Mapping[str, float]: ...


class TorchRewardModel:
    """Base class for torch reward models loaded in-process."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self.worker_config = cfg
        self.device = str(cfg.get("device", "cuda"))
        self.dtype_str = str(cfg.get("dtype", "float32"))
        self._loaded = False

    @property
    def dtype(self) -> Any:
        return resolve_torch_dtype(self.dtype_str)

    def _load(self) -> None:
        """Build the underlying model. Called once, lazily."""

        raise NotImplementedError

    def score_media(self, *, media: Any, prompt: str, request: Any) -> Mapping[str, float]:
        """Return named scores for one artifact's media payload."""

        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def __call__(self, *, artifact: Any, request: Any) -> Mapping[str, float]:
        self._ensure_loaded()
        return self.score_media(
            media=artifact.as_media(),
            prompt=artifact.prompt,
            request=request,
        )


__all__ = ["RewardModel", "TorchRewardModel", "require_prompt_and_video_path"]
