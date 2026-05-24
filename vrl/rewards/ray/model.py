"""Contract for reward models executed inside a reward Ray actor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest


class RewardModel(Protocol):
    """A reward model the Ray actor loads and runs.

    Given one already-materialized artifact (a file path) plus the request, it
    returns named scores. The actor is model-agnostic: any model that implements
    this protocol and is named by worker_config.model_factory can be hosted.
    Concrete models live under vrl.rewards.models and inherit this protocol. It
    is the worker-side counterpart of an in-process RewardFunction.
    """

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> Mapping[str, float]: ...


__all__ = ["RewardModel"]
