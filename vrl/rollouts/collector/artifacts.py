"""Reward artifact lifecycle helpers for rollout collection."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from vrl.generation import GenerationOutput
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.storage import trajectory_tensor_bytes


@dataclass(frozen=True, slots=True)
class RewardArtifactPolicy:
    """Runtime-only policy for decoded reward artifacts after scoring."""

    keep_after_reward: bool = True


def reward_artifact_policy_from_cfg(value: object) -> RewardArtifactPolicy:
    """Parse rollout.reward_artifact config into a typed lifecycle policy."""

    if value is None:
        return RewardArtifactPolicy()
    value = _to_builtin(value)
    if isinstance(value, RewardArtifactPolicy):
        return value
    if isinstance(value, Mapping):
        return RewardArtifactPolicy(
            keep_after_reward=bool(value.get("keep_after_reward", True)),
        )
    return RewardArtifactPolicy(
        keep_after_reward=bool(_cfg_get(value, "keep_after_reward", True)),
    )


def extract_reward_artifact(output: GenerationOutput) -> object:
    """Return the decoded artifact carried by a generation output."""

    return output.output


def reward_artifact_bytes(artifact: object) -> int:
    """Return an estimated byte count for decoded reward artifacts."""

    return trajectory_tensor_bytes(artifact)


def release_reward_artifact_if_needed(
    batch: object,
    policy: RewardArtifactPolicy,
) -> None:
    """Drop reward-only artifact references when policy asks for release."""

    if policy.keep_after_reward:
        return

    if isinstance(batch, GenerationOutput):
        batch.output = None
        _release_reward_view_tensors(batch.trajectory)
        return

    if isinstance(batch, RolloutBatch):
        batch.videos = None
        _release_reward_view_tensors(batch.trajectory)
        return

    if hasattr(batch, "videos"):
        with suppress(AttributeError):
            batch.videos = None
    _release_reward_view_tensors(getattr(batch, "trajectory", None))


def _release_reward_view_tensors(trajectory: Any) -> None:
    if trajectory is None:
        return
    segments = getattr(trajectory, "segments", None)
    reward_views = getattr(trajectory, "reward_views", None)
    if not isinstance(segments, Mapping) or not isinstance(reward_views, Mapping):
        return

    refs: set[str] = set()
    for view in reward_views.values():
        refs.update(getattr(view, "tensor_refs", ()) or ())

    for ref in refs:
        if "." not in ref:
            continue
        segment_name, tensor_name = ref.split(".", 1)
        segment = segments.get(segment_name)
        if segment is None or getattr(segment, "trainable", False):
            continue
        tensor = getattr(segment, "tensors", {}).get(tensor_name)
        if tensor is None:
            continue
        tensor.value = None
        tensor.metadata = {
            **dict(getattr(tensor, "metadata", {})),
            "released_after_reward": True,
        }


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


def _to_builtin(value: object) -> object:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


__all__ = [
    "RewardArtifactPolicy",
    "extract_reward_artifact",
    "release_reward_artifact_if_needed",
    "reward_artifact_bytes",
    "reward_artifact_policy_from_cfg",
]
