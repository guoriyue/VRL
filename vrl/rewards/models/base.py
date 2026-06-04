"""Shared base for in-process torch reward models.

``TorchRewardModel`` implements the ``RewardModel`` protocol
(``__call__(artifact, request) -> Mapping[str, float]``) and absorbs the
device/dtype/lazy-load boilerplate that every torch-nn reward used to hand-roll.
Subclasses implement ``_load`` (build the model once) and ``score_media``
(score one media payload + prompt). Media is pulled via ``artifact.as_media()``,
so the same model runs under the local transport (in-memory tensor) or the Ray
transport (tensor loaded from the materialized ``.pt`` path).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.models.dtypes import resolve_torch_dtype


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


__all__ = ["TorchRewardModel"]
