"""GenEval reward over structured prompt metadata."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardSample


class GenEvalReward(RewardFunction):
    """Delegate one sample to an injected or import-path GenEval score_fn."""

    def __init__(
        self,
        device: str = "cuda",
        import_path: str = "",
        debug_dir: str = "",
        artifact_dir: str = "",
        score_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.device = str(device)
        self.import_path = str(import_path)
        self.debug_dir = str(debug_dir)
        self.artifact_dir = str(artifact_dir)
        self._scorer = score_fn

    async def score(self, sample: RewardSample) -> float:
        """Score one generated sample without an artificial artifact transport."""

        metadata = dict(sample.metadata)
        result = (self._scorer or self._load_import_path())(
            prompt=sample.prompt,
            output=sample.output,
            geneval=self._extract_geneval_metadata(sample),
            metadata=metadata,
            device=self.device,
            artifact_dir=self.artifact_dir,
            debug_dir=self.debug_dir,
        )
        if inspect.isawaitable(result):
            result = await result
        return _normalize_result(result)

    @staticmethod
    def _extract_geneval_metadata(sample: RewardSample) -> dict[str, Any]:
        geneval = sample.metadata.get("geneval")
        if isinstance(geneval, dict):
            return dict(geneval)
        raise ValueError("GenEvalReward requires metadata.geneval on each sample")

    def _load_import_path(self) -> Callable[..., Any]:
        if not self.import_path:
            raise RuntimeError(
                "GenEvalReward requires an injected score_fn or reward.kwargs.geneval.import_path",
            )
        module_name, separator, attribute_name = self.import_path.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError(
                "GenEval import_path must have the form 'module.submodule:function'",
            )
        score_fn = getattr(importlib.import_module(module_name), attribute_name)
        if not callable(score_fn):
            raise TypeError(f"GenEval import_path target is not callable: {self.import_path}")
        self._scorer = score_fn
        return score_fn


def _normalize_result(result: Any) -> float:
    if isinstance(result, dict):
        if "score" not in result:
            raise ValueError("GenEval score_fn dict result must contain a 'score' key")
        result = result["score"]
    return float(result)


__all__ = ["GenEvalReward"]
