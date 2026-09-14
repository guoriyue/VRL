"""Shared parent of the previous-policy-adapter objectives (DiffusionNFT, V-GRPO).

Both objectives train a LoRA policy against a frozen "previous" copy of the
adapter, refresh that copy after every optimizer step, and evaluate the
denoiser at an interpolated ``x_t`` on flow time ``t`` in ``[0, 1]``. Their
lr=0 first-step invariants differ (NFT's loss is symmetric under flipping the
advantages, V-GRPO's antisymmetric) but both need the same seeded double
evaluation.

The model surface they consume is the shared denoise replay contract, nothing
objective-specific: ``replay_forward_with_latents`` evaluates the family's own
conditional forward at a trajectory step on caller-supplied latents, and the
``previous`` adapter is switched through ``activate_adapter``. Which families
qualify is a registry fact (a full-sequence replay recipe), not a per-family
flag.
"""

from __future__ import annotations

from typing import Any, ClassVar


class PreviousAdapterObjective:
    """Replay-branch objective whose behaviour policy is the previous adapter."""

    name: ClassVar[str]
    invariant_event: ClassVar[str]
    invariant_name: ClassVar[str]

    uses_evaluator = False
    # Replay-branch contract (AlgorithmAdapter.validate_inputs): these
    # objectives train the forward process from rollout tensors only — no
    # reverse-SDE trajectory, no log-probs. Declaring the keys lets the adapter
    # fail fast with available-vs-missing diagnostics.
    required_data_keys = ("latents_clean", "prompt_embeds", "timesteps")
    required_signal_keys: tuple[str, ...] = ()
    needs_kl_intermediates = False
    requires_active_trust_region = False
    # The behaviour policy is the previous adapter refreshed every optimizer
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
                f"{self.name} timestep grid must normalize into [0, 1]; got "
                f"min={float(t.min()):.4g}, max={float(t.max()):.4g} after the /1000 "
                "heuristic. EDM-scale timestep grids are not supported by this normalization.",
            )
        return t

    # -- lr=0 first-step invariant -------------------------------------

    def _invariant_residual(self, loss: float, flipped_loss: float) -> float:
        """How far ``(loss(A), loss(-A))`` is from the objective's flip invariant."""

        raise NotImplementedError

    def first_step_invariant_check(
        self,
        *,
        model: Any,
        batch: Any,
        advantages: Any,
        timestep_index: int = 0,
        threshold: float = 1.0e-6,
    ) -> dict[str, Any]:
        """Evaluate the loss under ``A`` and ``-A`` with one seeded RNG and score the residual.

        With the previous adapter freshly synced the objective is exactly
        invariant (NFT) or antisymmetric (V-GRPO) under flipping the
        advantages; ratio-style parity cannot see this, so the trainer's
        debug.first_step branch calls this optional protocol method instead.
        The RNG is forked and seeded so both evaluations draw the same noise
        when the trajectory carries none.
        """

        import torch

        def _loss(adv: Any) -> float:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                loss, _ = self.compute_batch_timestep_loss(model, batch, timestep_index, adv)
            return float(loss.detach().float().item())

        loss = _loss(advantages)
        flipped_loss = _loss(-advantages)
        residual = self._invariant_residual(loss, flipped_loss)
        return {
            "event": self.invariant_event,
            "invariant": self.invariant_name,
            "loss": loss,
            "flipped_loss": flipped_loss,
            "abs_diff": residual,
            "threshold": threshold,
            "passed": residual <= threshold,
        }

    # -- previous-adapter refresh --------------------------------------

    def after_optimizer_step(self, model: Any, global_step: int) -> None:
        """Refresh the previous-policy adapter (EMA with ``weight_copy_decay``)."""

        sync = getattr(model, "sync_previous_policy_adapter", None)
        if not callable(sync):
            raise RuntimeError(
                f"{self.name} model must expose sync_previous_policy_adapter(decay=...) "
                "for previous-policy refresh",
            )
        sync(decay=float(self.config.weight_copy_decay))
        self._after_previous_adapter_sync(global_step)

    def _after_previous_adapter_sync(self, global_step: int) -> None:
        """Per-step state the objective advances once the refresh landed; default none."""

        del global_step


__all__ = ["PreviousAdapterObjective"]
