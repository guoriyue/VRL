"""GenEval scoring as a model-backed RewardModel.

GenEval is metadata-driven: it scores a generated image against structured
prompt metadata (the ``geneval`` object: tag / include / exclude) by delegating
to a user-supplied callable. The callable is resolved from an injected
``scorer`` (preferred, e.g. for tests) or an ``import_path`` so the training
stack stays independent of the GenEval repo layout.

Ported verbatim from the in-process ``GenEvalReward._score_one`` logic: the
scorer is invoked with the bespoke keyword signature ``(prompt, output,
geneval, metadata, device, artifact_dir, debug_dir)`` and its result is
normalised to a float and returned as ``{"geneval": value}``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest


class _OutputBox:
    """Wrap a rollout output so it survives the artifact's media-presence check.

    GenEval rollouts often carry ``output=None`` (scoring is metadata-driven),
    but ``RewardInferenceArtifact`` requires a non-None media or a path. The box
    lets ``as_media()`` round-trip an arbitrary (possibly ``None``) payload.
    """

    __slots__ = ("output",)

    def __init__(self, output: Any) -> None:
        self.output = output


def _unbox(media: Any) -> Any:
    return media.output if isinstance(media, _OutputBox) else media


class GenEvalRewardModel:
    """RewardModel returning ``{"geneval": score}`` for one image artifact."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self.import_path = str(cfg.get("import_path", ""))
        self.debug_dir = str(cfg.get("debug_dir", ""))
        self.artifact_dir = str(cfg.get("artifact_dir", ""))
        self._scorer: Callable[..., Any] | None = cfg.get("scorer")

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        # Backend is decided by what is wired, not a knob: use the injected
        # scorer if present, otherwise resolve the import_path callable.
        scorer = self._scorer or self._load_import_path()
        result = scorer(
            prompt=artifact.prompt,
            output=_unbox(artifact.as_media()),
            geneval=dict(artifact.metadata.get("geneval", {})),
            metadata=dict(artifact.metadata.get("rollout_metadata", {})),
            device=self.device,
            artifact_dir=self.artifact_dir,
            debug_dir=self.debug_dir,
        )
        if inspect.isawaitable(result):
            result = _run_awaitable(result)
        return {"geneval": _normalize_result(result)}

    def _load_import_path(self) -> Callable[..., Any]:
        if not self.import_path:
            raise RuntimeError(
                "GenEvalReward requires an injected scorer or "
                "reward.kwargs.geneval.import_path",
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


def geneval_reward_model(worker_config: Mapping[str, Any]) -> GenEvalRewardModel:
    """RewardModel factory for the reward registry / InProcessRewardRuntime."""

    return GenEvalRewardModel(worker_config)


def _run_awaitable(awaitable: Any) -> Any:
    """Run an awaitable scorer result to completion from a sync context.

    The model's ``__call__`` is driven synchronously by ``score_artifacts_with_model``;
    if a scorer returns a coroutine we drive it on a fresh event loop so async
    GenEval evaluators keep working under the model-backed transport.
    """

    import asyncio

    return asyncio.new_event_loop().run_until_complete(awaitable)


def _normalize_result(result: Any) -> float:
    if isinstance(result, dict):
        if "score" not in result:
            raise ValueError("GenEval scorer dict result must contain a 'score' key")
        result = result["score"]
    return float(result)


__all__ = ["GenEvalRewardModel", "geneval_reward_model"]
