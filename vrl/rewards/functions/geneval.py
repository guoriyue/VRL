"""GenEval reward wrapper for prompt metadata driven scoring."""

from __future__ import annotations

import importlib
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vrl.rewards.base import RewardFunction
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
        evaluator: str = "import_path",
        import_path: str = "",
        timeout_s: float = 120.0,
        debug_dir: str = "",
        artifact_dir: str = "",
        scorer: Callable[..., Any] | None = None,
    ) -> None:
        self.device = device
        self.evaluator = evaluator
        self.import_path = import_path
        self.timeout_s = float(timeout_s)
        self.debug_dir = debug_dir
        self.artifact_dir = artifact_dir
        self._scorer = scorer

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        scores: list[float] = []
        for rollout in rollouts:
            start = time.monotonic()
            score = await self._score_one(rollout)
            elapsed = time.monotonic() - start
            if elapsed > self.timeout_s:
                raise TimeoutError(
                    f"GenEval scorer exceeded timeout_s={self.timeout_s}: {elapsed:.2f}s",
                )
            scores.append(float(score))
            self._write_debug_record(rollout, float(score), elapsed)
        return scores

    async def _score_one(self, rollout: RewardRollout) -> float:
        if self.evaluator != "import_path":
            raise ValueError(f"unsupported GenEval evaluator: {self.evaluator!r}")

        scorer = self._scorer or self._load_import_path()
        result = scorer(
            prompt=rollout.trajectory.prompt,
            output=rollout.trajectory.output,
            geneval=self._extract_geneval_metadata(rollout),
            metadata=dict(rollout.metadata),
            device=self.device,
            artifact_dir=self.artifact_dir,
            debug_dir=self.debug_dir,
        )
        if inspect.isawaitable(result):
            result = await result
        return self._normalize_result(result)

    def _load_import_path(self) -> Callable[..., Any]:
        if not self.import_path:
            raise RuntimeError(
                "GenEvalReward evaluator='import_path' requires reward.kwargs.geneval.import_path",
            )
        module_name, sep, attr_name = self.import_path.partition(":")
        if not sep or not module_name or not attr_name:
            raise ValueError(
                "GenEval import_path must have the form 'module.submodule:function'",
            )
        module = importlib.import_module(module_name)
        scorer = getattr(module, attr_name)
        if not callable(scorer):
            raise TypeError(f"GenEval import_path target is not callable: {self.import_path}")
        self._scorer = scorer
        return scorer

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

    @staticmethod
    def _normalize_result(result: Any) -> float:
        if isinstance(result, dict):
            if "score" not in result:
                raise ValueError("GenEval scorer dict result must contain a 'score' key")
            result = result["score"]
        return float(result)

    def _write_debug_record(self, rollout: RewardRollout, score: float, elapsed_s: float) -> None:
        if not self.debug_dir:
            return
        debug_path = Path(self.debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        record = {
            "prompt": rollout.trajectory.prompt,
            "score": score,
            "elapsed_s": elapsed_s,
            "geneval": self._extract_geneval_metadata(rollout),
        }
        with (debug_path / "geneval_scores.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["GenEvalReward"]
