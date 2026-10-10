"""The forward-process replay branch: DiffusionNFT and V-GRPO.

Both algorithms re-noise the rollout's clean latent on flow time ``t`` in
``[0, 1]`` and regress the denoiser's clean prediction, scoring the trainable
policy against ``theta_old`` (the detached current prediction: one trainable
forward, no weight snapshot). With one optimizer step per rollout
(``ppo_epochs: 1``, every shipped preset) that is exactly the policy that
generated the rollout.

The model surface they consume is the shared denoise replay contract, nothing
algorithm-specific: ``replay_forward_with_latents`` evaluates the family's own
conditional forward at a trajectory step on caller-supplied latents. The
re-noising itself reads the trajectory's recorded timestep, not a scheduler
object: these algorithms run without the SDE evaluator, and the trajectory is
the only state the trainer process is guaranteed to share with generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.advantages import GroupAdvantageConfig
from vrl.algorithms.base import OnlineAlgorithm


@dataclass(slots=True)
class ForwardProcessConfig(GroupAdvantageConfig):
    """Knobs shared by the forward-process algorithms: the papers' tighter std floor."""

    eps: float = 1e-8


class ForwardProcessAlgorithm(OnlineAlgorithm):
    """An algorithm that trains the forward process from the rollout's clean latents.

    No reverse-SDE trajectory, no log-probs, no evaluator; ``theta_old`` is the
    detached current prediction, so a superseded policy's rollout would be
    scored against the wrong weights (``requires_previous_policy``).
    """

    uses_evaluator = False
    sft_source = "unsupported"
    requires_previous_policy = True
    config: ForwardProcessConfig

    # -- re-noising -------------------------------------------------------

    def renoised(self, batch: Any, timestep_index: int) -> tuple[Any, Any, Any]:
        """``(x0, t, xt)`` at this replay step: the clean latent, flow time
        broadcast to its shape, and the fp32 interpolation ``(1-t)*x0 + t*noise``
        with the noise this algorithm draws (``self.noise``).
        """

        import torch

        from vrl.trajectory.reader import TrajectoryReader

        replay = TrajectoryReader.from_batch(batch).forward_process_replay(
            "denoise", timestep_index
        )
        x0 = replay.latents_clean
        t = self.flow_time(replay.timestep, x0).view(-1, *([1] * (x0.ndim - 1)))
        noise = self.noise(
            x0,
            group_ids=getattr(batch, "group_ids", None),
            timestep_index=int(timestep_index),
            recorded=replay.noise,
        )
        return x0, t, (1 - t) * x0.float() + t * noise.to(device=x0.device, dtype=torch.float32)

    def noise(self, x0: Any, *, group_ids: Any, timestep_index: int, recorded: Any | None) -> Any:
        """The forward-process noise for ``x0``: the rollout's recorded draw when
        the trajectory kept one, else a fresh standard-normal draw."""

        import torch

        del group_ids, timestep_index
        return torch.randn_like(x0.float()) if recorded is None else recorded

    def flow_time(self, t_raw: Any, x0: Any) -> Any:
        """Normalize a rollout timestep grid into flow time ``t`` in ``[0, 1]``.

        A ``[0, 1000]`` grid (SD3, FLUX, Wan, Cosmos Predict2.5's flow-sigma
        UniPC) divides down. EDM-style grids (Cosmos Predict2 FlowMatch,
        timesteps up to 80000) would land far outside ``[0, 1]`` and silently
        push the interpolation off the data manifold — the same failure shape
        as the predict2 sigma-domain incident — so they fail loud instead.
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


__all__ = ["ForwardProcessAlgorithm", "ForwardProcessConfig"]
