"""Shared pieces of the Cosmos model families."""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult


class CosmosReplayForward:
    """Cosmos replay: forward with the real ``timestep_idx`` (NOT 0).

    Unlike sd3/wan which pack timesteps as ``[1, B]`` and call
    ``forward_step(state, 0)``, Cosmos's ``forward_step`` indexes
    ``state.scheduler.sigmas[step_idx]`` so the eval path must pass
    through the actual ``timestep_idx`` to keep sigma scaling consistent
    with the rollout-time scheduler state.
    """

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del request
        replay_tensors, batch_context, latents = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(
            replay_tensors,
            batch_context,
            latents,
            timestep_idx,
        )
        values = self.forward_step(state, timestep_idx)
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=dict(values),
                ),
            },
        )


__all__ = ["CosmosReplayForward"]
