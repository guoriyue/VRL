"""Vision-language-action policy contract (P1 draft).

The action-policy seam, kept separate from the diffusion ``ReplayModel``
(``vrl/models/interfaces/replay.py``) because a robot policy's payload is an
action chunk, not an image/video latent. ``ActionPolicy`` mirrors the
replay/runtime split: an eval-only policy answers ``can_replay_logprob ==
False`` and returns ``ActionChunk.log_prob is None``; a trainable policy
(PI0.5 flow-matching, OpenVLA-OFT token-action) exposes a replayable log-prob
so the RL trainer's old-vs-new ratio is exact.

No concrete policy is implemented here. Per the sprint training gate, PI0.5 /
OpenVLA-OFT land in later MRs only after an env smoke + logprob-contract pass.
See ``docs/sprints/planned/SPRINT_physical_ai_model_support.md`` (§3.2, §P3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vrl.rollouts.envs.contract import ActionChunk, EnvObservation, EnvResetSpec


@runtime_checkable
class ActionPolicy(Protocol):
    """Maps an env observation to an action chunk for closed-loop rollout."""

    @property
    def action_horizon(self) -> int:
        """Number of future action steps emitted per inference."""
        ...

    @property
    def can_replay_logprob(self) -> bool:
        """True when ``act`` returns a chunk with a replayable action log-prob.

        False for an eval-only policy (e.g. a server-level JSON action); the
        collector must then build an eval-only ``ActionTrajectoryBatch`` and the
        trainer must refuse to compute an RL update from it.
        """
        ...

    def reset(self, spec: EnvResetSpec) -> None:
        """Reset any per-episode policy state (action history, KV cache)."""
        ...

    def act(self, obs: EnvObservation) -> ActionChunk:
        """Emit the next action chunk for the given observation."""
        ...


__all__ = ["ActionPolicy"]
