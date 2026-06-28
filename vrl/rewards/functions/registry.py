"""Multi-reward registry — weighted combination of named reward functions.

Ported from the multi_score() pattern in flow_grpo/rewards.py.
"""

from __future__ import annotations

import inspect
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout

# Registry of reward function factories.
# Each factory takes (device,) and returns a RewardFunction instance.
_REWARD_REGISTRY: dict[str, type[RewardFunction]] = {}


def get_reward(name: str) -> type[RewardFunction]:
    """Look up a registered reward function class by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward function: {name!r}. Available: {list(_REWARD_REGISTRY)}")
    return _REWARD_REGISTRY[name]


def _register_builtins() -> None:
    from vrl.rewards.functions.aesthetic import AestheticReward
    from vrl.rewards.functions.geneval import GenEvalReward
    from vrl.rewards.functions.kling_video_reward import KlingVideoReward
    from vrl.rewards.functions.nsfw_safety import NSFWSafetyReward
    from vrl.rewards.functions.ocr import OCRReward
    from vrl.rewards.functions.phymotion import PhyMotionReward
    from vrl.rewards.functions.pickscore import PickScoreReward
    from vrl.rewards.functions.target_video_similarity import TargetVideoSimilarityReward
    from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
    from vrl.rewards.functions.videocon_physics import VideoConPhysicsReward
    from vrl.rewards.functions.videoscore2 import VideoScore2Reward

    _REWARD_REGISTRY.update({
        "aesthetic": AestheticReward,
        "geneval": GenEvalReward,
        "nsfw_safety": NSFWSafetyReward,
        "ocr": OCRReward,
        "pickscore": PickScoreReward,
        "target_video_similarity": TargetVideoSimilarityReward,
        "kling_video_reward": KlingVideoReward,
        "videocon_physics": VideoConPhysicsReward,
        "videoscore2": VideoScore2Reward,
        "unified_reward_video": UnifiedRewardVideoReward,
        "phymotion": PhyMotionReward,
        "target_video_similarity": TargetVideoSimilarityReward,
    })


class MultiReward(RewardFunction):
    """Weighted combination of named reward functions.

    Tracks per-component raw scores since the last reset so the training loop
    can log epoch-wide component means instead of only the final reward call.
    This is essential for spotting reward hacking early (e.g. aesthetic
    collapses while OCR climbs).

    Usage::

        reward_fn = MultiReward.from_dict(
            {"ocr": 1.0, "aesthetic": 0.3},
            device="cuda",
        )
        total = await reward_fn.score(rollout)
        # reward_fn.last_components -> {"ocr": [0.87], "aesthetic": [5.2]}
    """

    def __init__(
        self,
        rewards: list[tuple[str, float, RewardFunction]],
    ) -> None:
        self.rewards = rewards
        self.last_components: dict[str, list[float]] = {}
        self.last_results: list[Any] = []
        self.last_timing_ms: dict[str, float] = {}

    def reset_components(self) -> None:
        """Clear component score history before a new trainer step."""
        self.last_components = {}

    async def shutdown(self) -> None:
        for _, _, fn in self.rewards:
            shutdown = getattr(fn, "shutdown", None)
            if shutdown is None:
                continue
            result = shutdown()
            if inspect.isawaitable(result):
                await result

    @classmethod
    def from_dict(
        cls,
        score_dict: dict[str, float],
        device: str = "cuda",
        reward_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> MultiReward:
        """Build from ``{"name": weight}`` dict, like flow_grpo config.reward_fn.

        ``reward_kwargs`` allows passing per-reward init kwargs, keyed by name,
        e.g. ``{"ocr": {"debug_dir": "out/ocr_debug"}}``.
        """
        _register_builtins()
        reward_kwargs = reward_kwargs or {}
        triples: list[tuple[str, float, RewardFunction]] = []
        for name, weight in score_dict.items():
            reward_cls = get_reward(name)
            # `or {}`: a bare YAML key (kwargs: <name>:) parses as None.
            extra = reward_kwargs.get(name) or {}
            triples.append((name, weight, reward_cls(device=device, **extra)))
        return cls(triples)

    async def score(self, rollout: RewardRollout) -> float:
        self._reset_last_inference_observations()
        total = 0.0
        components: dict[str, list[float]] = {}
        for name, weight, fn in self.rewards:
            s = await fn.score(rollout)
            self._append_inference_observations(fn)
            components[name] = [s]
            total += weight * s
        self._append_components(components)
        return total

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        self._reset_last_inference_observations()
        totals = [0.0] * len(rollouts)
        components: dict[str, list[float]] = {}
        for name, weight, fn in self.rewards:
            sub_scores = await fn.score_batch(rollouts)
            self._append_inference_observations(fn)
            components[name] = list(sub_scores)
            for i, s in enumerate(sub_scores):
                totals[i] += weight * s
        self._append_components(components)
        return totals

    def _append_components(self, components: dict[str, list[float]]) -> None:
        for name, values in components.items():
            self.last_components.setdefault(name, []).extend(float(v) for v in values)

    def _reset_last_inference_observations(self) -> None:
        self.last_results = []
        self.last_timing_ms = {}

    def _append_inference_observations(self, fn: RewardFunction) -> None:
        self.last_results.extend(list(getattr(fn, "last_results", []) or []))
        for key, value in (getattr(fn, "last_timing_ms", {}) or {}).items():
            self.last_timing_ms[str(key)] = self.last_timing_ms.get(str(key), 0.0) + float(value)
