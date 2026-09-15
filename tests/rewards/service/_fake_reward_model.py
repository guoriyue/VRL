"""Importable-by-subprocess fake reward model for managed-service tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class FakeRewardModel:
    """Scores a tensor artifact by its mean; echoes worker_config knobs."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.scale = float(worker_config.get("scale", 1.0))
        self.device = str(worker_config.get("device", "cpu"))

    def __call__(self, artifact: Any) -> dict[str, float]:
        tensor = artifact.as_media()
        return {"overall": float(tensor.float().mean()) * self.scale}
