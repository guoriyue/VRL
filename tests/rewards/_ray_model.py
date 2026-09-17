"""Small real actor model fixture; no model downloads or CUDA work."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


class TinyRewardModel:
    def __init__(self, config: dict[str, Any]) -> None:
        if config["device"] != "cpu":
            raise AssertionError("this fixture must never launch GPU work")
        self.scale = config.get("scale", 1.0)

    def __call__(self, artifact: Any) -> dict[str, float]:
        time.sleep(float(artifact.metadata.get("delay", 0)))
        if artifact.metadata.get("fail"):
            raise RuntimeError("injected reward model failure")
        return {
            "score": float(artifact.as_media().mean()) * self.scale,
            "pid": float(os.getpid()),
        }


class TinyFileRewardModel(TinyRewardModel):
    input_artifact_format = "tensor"

    def __call__(self, artifact: Any) -> dict[str, float]:
        path = Path(artifact.as_path())
        if not path.is_file():
            raise AssertionError("file reward did not receive a materialized file")
        # This actor-root marker lets the CPU integration test cancel only
        # after the real file has reached the blocking model call.
        (path.parent.parent / "model-started").touch()
        return super().__call__(artifact)
