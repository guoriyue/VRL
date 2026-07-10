"""Shared pieces of the Cosmos model families."""

from __future__ import annotations


class CosmosReplayForward:
    """Cosmos replay uses the real ``timestep_idx`` instead of a rebuilt index 0.

    Unlike sd3/wan which pack timesteps as ``[1, B]`` and call
    ``forward_step(state, 0)``, Cosmos's ``forward_step`` indexes
    ``state.scheduler.sigmas[step_idx]`` so the eval path must pass
    through the actual ``timestep_idx`` to keep sigma scaling consistent with
    rollout. The base replay methods share this hook for stored and caller
    latents, preventing the SFT regularizer from drifting to ``sigma[0]``.
    """

    def _replay_forward_step_index(self, timestep_idx: int) -> int:
        return int(timestep_idx)


__all__ = ["CosmosReplayForward"]
