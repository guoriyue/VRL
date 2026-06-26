"""Closed-loop env rollout contract for robotics / VLA (P1 draft).

Public typed seam for ``env.reset -> policy -> env.step -> trajectory``. See
``vrl/rollouts/envs/contract.py`` for the ownership boundaries.
"""

from vrl.rollouts.envs.contract import (
    ActionChunk,
    ActionTrajectoryBatch,
    Env,
    EnvObservation,
    EnvResetSpec,
    EnvRewardSignal,
    EpisodeArtifact,
)

__all__ = [
    "ActionChunk",
    "ActionTrajectoryBatch",
    "Env",
    "EnvObservation",
    "EnvResetSpec",
    "EnvRewardSignal",
    "EpisodeArtifact",
]
