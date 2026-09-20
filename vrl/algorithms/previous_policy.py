"""Shared parent of the previous-policy objectives (DiffusionNFT, V-GRPO).

Both objectives train against a frozen copy of the previous step's policy,
refresh that copy after every optimizer step, and evaluate the
denoiser at an interpolated ``x_t`` on flow time ``t`` in ``[0, 1]``. Their
lr=0 first-step invariants differ (NFT's loss is symmetric under flipping the
advantages, V-GRPO's antisymmetric) but both need the same seeded double
evaluation.

The model surface they consume is the shared denoise replay contract, nothing
objective-specific: ``replay_forward_with_latents`` evaluates the family's own
conditional forward at a trajectory step on caller-supplied latents, and the
previous policy is switched in through ``previous_policy``. Whether that policy
is a LoRA adapter or the whole transformer is the model's business. Which families
qualify is a registry fact (a full-sequence replay recipe), not a per-family
flag.
"""

from __future__ import annotations

from typing import Any


class PreviousPolicyObjective:
    """Replay-branch objective whose behaviour policy is the previous step's weights."""

    # These objectives train the forward process from the rollout's clean
    # latents (TrajectoryReader.forward_process_replay) — no reverse-SDE
    # trajectory, no log-probs, no evaluator.
    uses_evaluator = False
    requires_active_trust_region = False
    # The behaviour policy is the previous policy refreshed every optimizer
    # step, not the policy that generated a stale rollout: training on rollouts
    # from a superseded policy would score them against the wrong theta_old
    # (and NFT, being likelihood-free, has no importance ratio to absorb the
    # lag at all). So the continuous-rollout staleness window must be 0;
    # build_rollout_schedule fails fast on an unsound max_stale>0 config.
    tolerates_off_policy_staleness = False

    config: Any  # carries ``weight_copy_decay``

    def compute_batch_timestep_loss(
        self,
        model: Any,
        batch: Any,
        timestep_index: int,
        advantages: Any,
    ) -> tuple[Any, Any]:
        raise NotImplementedError

    # -- x0 regression --------------------------------------------------

    @staticmethod
    def normalized_mse(prediction: Any, target: Any) -> Any:
        """Per-sample MSE normalized by the detached mean absolute error (NFT / V-GRPO Eq. 14)."""

        import torch

        reduce_dims = tuple(range(1, target.ndim))
        with torch.no_grad():
            weight = (
                torch.abs(prediction.double() - target.double())
                .mean(dim=reduce_dims, keepdim=True)
                .clip(min=1e-5)
            )
        return ((prediction - target) ** 2 / weight).mean(dim=reduce_dims)

    # -- flow time -----------------------------------------------------

    def flow_time(self, t_raw: Any, x0: Any) -> Any:
        """Normalize a rollout timestep grid into flow time ``t`` in ``[0, 1]``.

        A ``[0, 1000]`` grid (SD3, FLUX, Wan) divides down. EDM-style grids
        (e.g. Cosmos Predict2 FlowMatch, timesteps up to 80000) would land far
        outside ``[0, 1]`` and silently push the ``xt = (1-t)*x0 + t*noise``
        interpolation off the data manifold — the same failure shape as the
        predict2 sigma-domain incident — so they fail loud instead.
        """

        import torch

        t = t_raw.to(device=x0.device, dtype=torch.float32)
        if bool((t > 1.0).any()):
            t = t / 1000.0
        if bool((t > 1.0).any()) or bool((t < 0.0).any()):
            raise RuntimeError(
                f"{type(self).__name__} timestep grid must normalize into [0, 1]; got "
                f"min={float(t.min()):.4g}, max={float(t.max()):.4g} after the /1000 "
                "heuristic. EDM-scale timestep grids are not supported by this normalization.",
            )
        return t

    # -- lr=0 first-step invariant -------------------------------------

    def _flipped_advantage_losses(
        self,
        model: Any,
        batch: Any,
        advantages: Any,
        timestep_index: int,
    ) -> tuple[float, float]:
        """``(loss(A), loss(-A))`` under one seeded RNG.

        With the previous policy freshly synced the objective is exactly
        invariant (NFT) or antisymmetric (V-GRPO) under flipping the
        advantages; ratio-style parity cannot see this, so each objective's
        ``first_step_invariant_check`` scores this pair. The RNG is forked and
        seeded so both evaluations draw the same noise when the trajectory
        carries none.
        """

        import torch

        def _loss(adv: Any) -> float:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                loss, _ = self.compute_batch_timestep_loss(model, batch, timestep_index, adv)
            return float(loss.detach().float().item())

        return _loss(advantages), _loss(-advantages)

    # -- previous-policy refresh ---------------------------------------

    def after_optimizer_step(self, model: Any, global_step: int) -> None:
        """Refresh the previous policy (EMA with ``weight_copy_decay``)."""

        del global_step
        model.sync_previous_policy(decay=float(self.config.weight_copy_decay))


__all__ = ["PreviousPolicyObjective"]
