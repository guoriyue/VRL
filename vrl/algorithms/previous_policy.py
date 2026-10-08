"""Shared parent of the previous-policy objectives (DiffusionNFT, V-GRPO).

Both objectives score the trainable policy against the behaviour policy
``theta_old`` and evaluate the denoiser at an interpolated ``x_t`` on flow time
``t`` in ``[0, 1]``. ``theta_old`` is the policy as of the current optimizer
step, which each objective reads as the detached current prediction: one
trainable forward, no weight snapshot. With one optimizer step per rollout
(``ppo_epochs: 1``, every shipped preset) that is exactly the policy that
generated the rollout; with more steps per rollout nothing holds the update
to a trust region across them.

The model surface they consume is the shared denoise replay contract, nothing
objective-specific: ``replay_forward_with_latents`` evaluates the family's own
conditional forward at a trajectory step on caller-supplied latents. Which
families qualify is a registry fact (a full-sequence replay recipe), not a
per-family flag.
"""

from __future__ import annotations

from typing import Any


class PreviousPolicyObjective:
    """Replay-branch objective whose behaviour policy is the current step's weights."""

    # These objectives train the forward process from the rollout's clean
    # latents (TrajectoryReader.forward_process_replay) — no reverse-SDE
    # trajectory, no log-probs, no evaluator.
    # The behaviour policy is the current policy, so their config contracts
    # leave tolerates_off_policy_staleness False: a superseded policy's
    # rollout would be scored against the wrong theta_old (and NFT, being
    # likelihood-free, has no importance ratio to absorb the lag).
    uses_evaluator = False
    # Trained on the forward process: no clean-target SFT term, and no KL term
    # unless the objective defines one.
    kl_coef = 0.0
    sft_weight = 0.0

    # -- lifecycle entry points the trainer calls on every objective ---------

    def prepare_update(self, update_timesteps: Any) -> None:
        """Called once per optimizer update; nothing is normalized over it here."""

        del update_timesteps

    def after_optimizer_step(self, global_step: int) -> None:
        """Called after every applied optimizer step."""

        del global_step

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


__all__ = ["PreviousPolicyObjective"]
