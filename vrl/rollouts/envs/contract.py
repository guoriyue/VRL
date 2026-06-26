"""Closed-loop robot/world environment rollout contract (P1 draft).

This is the typed seam for robotics / VLA rollouts, deliberately kept separate
from the diffusion ``RolloutBatch`` (``vrl/rollouts/batch/core.py``). A robot
episode is not ``prompt -> video -> reward``; it is::

    env.reset(spec) -> obs_0
    obs_t -> policy -> action_chunk
    env.step(action_chunk) -> obs_{t+1}, reward_signal
    trajectory -> ActionTrajectoryBatch

Ownership boundaries (the reason this is its own seam, not a diffusion family):

    environment owns success/reward   -> EnvRewardSignal
    policy owns action/logprob        -> ActionChunk
    collector owns episode grouping   -> ActionTrajectoryBatch.group_ids
    trainer only consumes typed       -> ActionTrajectoryBatch + rewards

These structs do NOT reuse the image/video-latent fields of ``RolloutBatch``
(``observations=x_t`` / ``actions=x_{t-1}``); robot actions, env state, success
signal and episode video have different ownership. See
``docs/sprints/planned/SPRINT_physical_ai_model_support.md`` (§3.2, §P1).

No environment or policy is registered here yet: per the sprint training gate,
real env adapters (LIBERO) and trainable policies (PI0.5, OpenVLA-OFT) land in
later MRs only after a simulator smoke + logprob-contract pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EnvResetSpec:
    """Immutable request to (re)initialize one episode in a robot/world env.

    ``task`` selects the scene/goal (e.g. ``"libero_10/put_bowl_in_drawer"``);
    ``embodiment`` pins the robot body so a DROID policy is never silently run
    against a LIBERO-Franka action space (the embodiment gap the sprint calls
    out in Tier 1).
    """

    task: str
    seed: int | None = None
    embodiment: str | None = None
    instruction: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("EnvResetSpec.task must be a non-empty string")


@dataclass(slots=True)
class EnvObservation:
    """One environment observation at a single control step. Policy input.

    ``images`` is a camera-name -> array map (``[H, W, C]``) so multi-view rigs
    (DROID has wrist + exterior cameras) stay explicit instead of being packed
    into an anonymous tensor.
    """

    step: int
    images: Mapping[str, Any] = field(default_factory=dict)
    state: Any | None = None
    instruction: str | None = None
    done: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("EnvObservation.step must be non-negative")


@dataclass(slots=True)
class ActionChunk:
    """A policy-emitted action chunk for one control step. Policy output.

    Robotics policies (PI0.5 flow-matching, OpenVLA-OFT token-action) emit a
    *chunk* of future actions per inference, not a single action; ``actions`` is
    ``[horizon, action_dim]``. ``log_prob`` is the policy's own replayable
    log-prob over the chunk and is ``None`` for an eval-only policy that cannot
    expose a trainable distribution (e.g. a server-level JSON action) — the
    sprint forbids fabricating a logprob in that case.
    """

    actions: Any
    log_prob: Any | None = None
    distribution: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        """True when the chunk carries a replayable action log-prob."""
        return self.log_prob is not None


@dataclass(slots=True)
class EnvRewardSignal:
    """Reward / success owned by the environment, never by the policy.

    ``components`` keeps per-term reward provenance (e.g. task-success vs shaping)
    so the collector can fold it into trajectory stats without the policy ever
    inventing a reward.
    """

    reward: float
    success: bool = False
    terminal: bool = False
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeArtifact:
    """Storage-bounded renderable artifacts for one episode.

    Kept separate from the reward signal so episode video / action traces can be
    dropped under a storage policy without touching the trainable trajectory
    (the sprint's "episode artifact storage bounded" training gate).
    """

    video: Any | None = None
    action_trace: Any | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ActionTrajectoryBatch:
    """Typed robot-action trajectory consumed by the trainer.

    Parallel to ``RolloutBatch`` but for closed-loop env rollouts. The collector
    fills this after stepping an environment to completion across a group of
    episodes. ``log_probs`` is ``None`` for an eval-only collection (no trainable
    policy distribution); a trainer must refuse to build an RL update from such a
    batch rather than treat absent logprobs as zero.
    """

    observations: Any            # stacked obs features [B, T, ...]
    actions: Any                 # [B, T, horizon, action_dim]
    log_probs: Any | None        # [B, T, horizon] or None (eval-only)
    rewards: Any                 # [B] terminal/return reward per episode
    success: Any                 # [B] bool success per episode
    dones: Any                   # [B, T] per-step done flags
    group_ids: Any               # [B] task/prompt group assignment
    episode_lengths: Any         # [B] control steps actually taken per episode
    distribution: str | None = None
    videos: Any | None = None    # [B, C, T, H, W] episode frames (for video reward)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        """True when every episode carries replayable action log-probs."""
        return self.log_probs is not None


@runtime_checkable
class Env(Protocol):
    """Closed-loop robot/world environment seam.

    The environment is the sole owner of reward and success. A policy calls
    ``reset`` once per episode, then alternates ``step`` until the returned
    ``EnvRewardSignal.terminal`` (or ``EnvObservation.done``) is set.
    """

    @property
    def action_dim(self) -> int:
        """Width of a single action vector this env accepts."""
        ...

    def reset(self, spec: EnvResetSpec) -> EnvObservation:
        """Initialize an episode and return the first observation."""
        ...

    def step(self, action: ActionChunk) -> tuple[EnvObservation, EnvRewardSignal]:
        """Apply one action chunk and return the next observation + reward."""
        ...

    def render_episode(self) -> EpisodeArtifact:
        """Return the renderable artifacts for the episode so far."""
        ...


__all__ = [
    "ActionChunk",
    "ActionTrajectoryBatch",
    "Env",
    "EnvObservation",
    "EnvResetSpec",
    "EnvRewardSignal",
    "EpisodeArtifact",
]
