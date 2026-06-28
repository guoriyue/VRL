"""VLA/Env rollout contract (P1 draft) — ownership boundaries pinned.

These tests fix the typed seam from SPRINT_physical_ai_model_support.md §P1:
the environment owns reward, the policy owns action/logprob, and an eval-only
trajectory (no replayable logprob) is representable and clearly distinguishable
from a trainable one — so a trainer can refuse to build an RL update from it
instead of treating absent logprobs as zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrl.models.vla import ActionPolicy
from vrl.rollouts.envs import (
    ActionChunk,
    ActionTrajectoryBatch,
    Env,
    EnvObservation,
    EnvResetSpec,
    EnvRewardSignal,
    EpisodeArtifact,
)


def test_reset_spec_requires_task() -> None:
    with pytest.raises(ValueError, match="task must be a non-empty string"):
        EnvResetSpec(task="")
    spec = EnvResetSpec(task="libero_10/put_bowl", embodiment="franka")
    assert spec.embodiment == "franka"
    assert spec.options == {}


def test_observation_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="step must be non-negative"):
        EnvObservation(step=-1)
    obs = EnvObservation(step=0, images={"wrist": np.zeros((2, 2, 3))})
    assert "wrist" in obs.images


def test_action_chunk_trainable_flag() -> None:
    """A chunk is trainable iff it carries a replayable log-prob."""
    eval_only = ActionChunk(actions=np.zeros((4, 7)))
    assert eval_only.log_prob is None
    assert eval_only.is_trainable is False

    trainable = ActionChunk(
        actions=np.zeros((4, 7)),
        log_prob=np.zeros((4,)),
        distribution="flow_matching",
    )
    assert trainable.is_trainable is True


def test_reward_signal_is_env_owned() -> None:
    """Reward/success live on the env signal, with per-term provenance."""
    sig = EnvRewardSignal(
        reward=1.0, success=True, terminal=True, components={"task": 1.0, "shaping": 0.0},
    )
    assert sig.success is True
    assert sig.components["task"] == 1.0


def test_action_trajectory_batch_eval_only_vs_trainable() -> None:
    """An eval-only batch (log_probs=None) is distinguishable from a trainable one."""
    eval_batch = ActionTrajectoryBatch(
        actions=np.zeros((2, 8, 16, 7)),
        log_probs=None,
        rewards=np.zeros((2,)),
        success=np.zeros((2,), dtype=bool),
        dones=np.zeros((2, 8), dtype=bool),
        group_ids=np.zeros((2,), dtype=np.int64),
        episode_lengths=np.full((2,), 8, dtype=np.int64),
    )
    assert eval_batch.is_trainable is False

    trainable_batch = ActionTrajectoryBatch(
        actions=np.zeros((2, 8, 16, 7)),
        log_probs=np.zeros((2, 8, 16)),
        rewards=np.zeros((2,)),
        success=np.zeros((2,), dtype=bool),
        dones=np.zeros((2, 8), dtype=bool),
        group_ids=np.zeros((2,), dtype=np.int64),
        episode_lengths=np.full((2,), 8, dtype=np.int64),
    )
    assert trainable_batch.is_trainable is True


def test_episode_artifact_is_optional() -> None:
    art = EpisodeArtifact()
    assert art.video is None and art.action_trace is None


def test_protocols_are_runtime_checkable() -> None:
    """A minimal duck-typed env + policy satisfy the runtime_checkable protocols."""

    class _Env:
        action_dim = 7

        def reset(self, spec: EnvResetSpec) -> EnvObservation:
            return EnvObservation(step=0)

        def step(self, action: ActionChunk) -> tuple[EnvObservation, EnvRewardSignal]:
            return EnvObservation(step=1, done=True), EnvRewardSignal(reward=0.0, terminal=True)

        def render_episode(self) -> EpisodeArtifact:
            return EpisodeArtifact()

    class _Policy:
        action_horizon = 16
        can_replay_logprob = False

        def reset(self, spec: EnvResetSpec) -> None:
            return None

        def act(self, obs: EnvObservation) -> ActionChunk:
            return ActionChunk(actions=np.zeros((16, 7)))

    assert isinstance(_Env(), Env)
    assert isinstance(_Policy(), ActionPolicy)


def test_closed_loop_smoke() -> None:
    """One reset/step loop drives the contract end-to-end without a real model."""

    class _Env:
        action_dim = 7

        def __init__(self) -> None:
            self._t = 0

        def reset(self, spec: EnvResetSpec) -> EnvObservation:
            self._t = 0
            return EnvObservation(step=0, instruction=spec.instruction)

        def step(self, action: ActionChunk) -> tuple[EnvObservation, EnvRewardSignal]:
            self._t += 1
            done = self._t >= 3
            obs = EnvObservation(step=self._t, done=done)
            sig = EnvRewardSignal(reward=1.0 if done else 0.0, success=done, terminal=done)
            return obs, sig

    env = _Env()
    obs = env.reset(EnvResetSpec(task="t", instruction="go"))
    total = 0.0
    while not obs.done:
        chunk = ActionChunk(actions=np.zeros((16, 7)))
        obs, sig = env.step(chunk)
        total += sig.reward
    assert obs.step == 3
    assert total == 1.0
