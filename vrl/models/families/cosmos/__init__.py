"""Shared pieces of the Cosmos model families."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any


class NoOpCosmosSafetyChecker:
    """Avoid loading guardrail weights in internal optimization diagnostics.

    Both predict2 and predict2.5 pipelines instantiate a CosmosSafetyChecker
    during from_pretrained; this stub satisfies the pipeline's surface without
    downloading guardrail weights (identical hand-copies previously lived in
    each family module).
    """

    def to(self, *_args: Any, **_kwargs: Any) -> NoOpCosmosSafetyChecker:
        return self

    def check_text_safety(self, _prompt: str) -> bool:
        return True

    def check_video_safety(self, video: Any) -> Any:
        return video


@contextlib.contextmanager
def no_safety_checker(pipeline_module: Any) -> Iterator[None]:
    """Swap the module's CosmosSafetyChecker for the stub during from_pretrained."""

    original = pipeline_module.CosmosSafetyChecker
    pipeline_module.CosmosSafetyChecker = NoOpCosmosSafetyChecker
    try:
        yield
    finally:
        pipeline_module.CosmosSafetyChecker = original


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


__all__ = ["CosmosReplayForward", "NoOpCosmosSafetyChecker", "no_safety_checker"]
