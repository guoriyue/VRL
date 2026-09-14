"""Pieces shared by the previous-policy-adapter objectives (DiffusionNFT, V-GRPO).

Both objectives train a LoRA policy against a frozen "previous" copy of the
adapter, refresh that copy after every optimizer step, and evaluate the
denoiser at an interpolated ``x_t`` on flow time ``t`` in ``[0, 1]``. Their
lr=0 first-step invariants differ (NFT's loss is symmetric under flipping the
advantages, V-GRPO's antisymmetric) but both need the same seeded double
evaluation.
"""

from __future__ import annotations

from typing import Any


def flow_time(t_raw: Any, x0: Any, *, owner: str) -> Any:
    """Normalize a rollout timestep grid into flow time ``t`` in ``[0, 1]``.

    A ``[0, 1000]`` grid (SD3, FLUX, Wan) divides down. EDM-style grids (e.g.
    Cosmos Predict2 FlowMatch, timesteps up to 80000) would land far outside
    ``[0, 1]`` and silently push the ``xt = (1-t)*x0 + t*noise`` interpolation
    off the data manifold — the same failure shape as the predict2 sigma-domain
    incident — so they fail loud instead.
    """

    import torch

    t = t_raw.to(device=x0.device, dtype=torch.float32)
    if bool((t > 1.0).any()):
        t = t / 1000.0
    if bool((t > 1.0).any()) or bool((t < 0.0).any()):
        raise RuntimeError(
            f"{owner} timestep grid must normalize into [0, 1]; got "
            f"min={float(t.min()):.4g}, max={float(t.max()):.4g} after the /1000 "
            "heuristic. EDM-scale timestep grids are not supported by this normalization.",
        )
    return t


def flipped_advantage_losses(
    algorithm: Any,
    *,
    model: Any,
    batch: Any,
    advantages: Any,
    timestep_index: int,
) -> tuple[float, float]:
    """``(loss(A), loss(-A))`` from ``compute_batch_timestep_loss`` under one seeded RNG.

    The RNG is forked and seeded so both evaluations draw the same noise when
    the trajectory carries none.
    """

    import torch

    def _loss(adv: Any) -> float:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            loss, _ = algorithm.compute_batch_timestep_loss(model, batch, timestep_index, adv)
        return float(loss.detach().float().item())

    return _loss(advantages), _loss(-advantages)


def sync_previous_policy_adapter(model: Any, *, decay: float, owner: str) -> None:
    """Refresh the model's previous-policy adapter copy (EMA with ``decay``)."""

    sync = getattr(model, "sync_previous_policy_adapter", None)
    if not callable(sync):
        raise RuntimeError(
            f"{owner} model must expose sync_previous_policy_adapter(decay=...) "
            "for previous-policy refresh",
        )
    sync(decay=float(decay))


__all__ = ["flipped_advantage_losses", "flow_time", "sync_previous_policy_adapter"]
