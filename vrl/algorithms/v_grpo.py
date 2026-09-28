"""V-GRPO: GRPO on an ELBO-based likelihood surrogate (arXiv:2604.23380).

Variational GRPO replaces the per-step SDE log-prob of the induced MDP with the
flow-matching training loss of the *final* sample as a likelihood surrogate:

    log pi(o | c)  <-  -L(theta | o, c) = -E_{t, eps}[ w_t ||NN_theta(z_t) - r_t||^2 ]

so the importance ratio between the current and the behaviour policy is
``rho = exp(-L(theta) + L(theta_old))`` and the GRPO objective is the usual
clipped ``min(rho A, clip(rho) A)``. Rollout is plain generation (no per-step
log-probs, any ODE sampler); training needs the clean sample, the prompt
conditioning and a few ``(t, eps)`` pairs.

What this module implements, and how it maps onto the trainer:

- **One ``(t, eps)`` pair per trainer replay index.** The trainer's per-step
  loss loop hands the objective one ``timestep_index`` at a time; that index
  picks ``t`` off the rollout grid and the objective draws ``eps`` for it. With
  ``actor.timestep_selection: stratified`` and
  ``timestep_fraction = N_MC / num_steps`` the indices are the paper's
  stratified draw (one per equal-length interval of the schedule), resampled
  every update.
- **Group-shared noise.** ``eps`` is generated from a seed derived from the
  prompt group id, the replay index and the update counter, so every sample of
  a group is scored on the same pairs (the paper's within-group variance fix)
  while pairs still change from update to update.
- **Adaptive loss weighting.** Both losses are the x-prediction MSE normalized
  by its own detached mean absolute error (``normalized_mse``, Eq. 14).
- **Behaviour policy = the detached current prediction.** ``theta_old`` is
  the policy as of the current optimizer step; with ``ppo_epochs: 1`` that is
  exactly the paper's ``theta_old``. So ``rho = exp(sg(L) - L)`` is 1 in value
  and its gradient is ``-grad L``: the objective is REINFORCE with a group
  baseline on the surrogate, the paper's fully on-policy Stage-1 recipe. The
  paper's multi-epoch controls (ratio clipping, Eq. 16's KL to ``theta_old``)
  are identically inert at ``rho == 1`` and are not implemented.
- **Advantage soft clipping** ``eta * tanh(A / eta)`` (``adv_soft_clip``,
  Eq. 17) is the one gradient-step control.

The model surface is the shared denoise replay contract, the same one
DiffusionNFT consumes: ``replay_forward_with_latents`` (the family's conditional
forward at a trajectory step on a caller-noised clean latent). Any family with a
full-sequence replay recipe runs either objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.config_contract import AlgorithmConfigContract
from vrl.algorithms.previous_policy import PreviousPolicyObjective
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import TrainStepMetrics
from vrl.models.precision import model_autocast

_SEED_UPDATE = 1_000_003
_SEED_GROUP = 7_919
_SEED_INDEX = 104_729


@dataclass(slots=True)
class VGRPOConfig:
    """Hyper-parameters for the V-GRPO objective.

    ``adv_soft_clip`` is off when ``None``. The paper's SD 3.5 M Stage-1 recipe
    (fully on-policy) uses ``adv_soft_clip=3``.
    """

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=False,
        sft_source="unsupported",
        requires_previous_policy=True,
    )

    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False
    adv_soft_clip: float | None = 3.0

    def __post_init__(self) -> None:
        if self.adv_soft_clip is not None and float(self.adv_soft_clip) <= 0.0:
            raise ValueError(
                f"VGRPOConfig.adv_soft_clip must be > 0 or null, got {self.adv_soft_clip}"
            )


class VGRPO(PreviousPolicyObjective):
    """Variational GRPO objective on the forward-process replay branch.

    ``theta_old`` is the detached current prediction, so the ratio is
    identically 1 and the objective is REINFORCE with a group baseline, which
    is the paper's Stage-1 recipe.
    """

    def __init__(self, config: VGRPOConfig | None = None) -> None:
        self.config = config or VGRPOConfig()
        # Advances with the optimizer so the group-shared noise changes across
        # updates while staying fixed within one.
        self._update_counter = 0

    def compute_advantages_from_tensors(self, rewards: Any, group_ids: Any) -> Any:
        cfg = self.config
        return group_relative_advantages(
            rewards,
            group_ids,
            eps=cfg.eps,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
        )

    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        return self.compute_batch_timestep_loss(
            inputs.model,
            inputs.rollout_batch,
            inputs.timestep_index,
            inputs.advantages,
        )

    # -- the objective ------------------------------------------------------

    def compute_batch_timestep_loss(
        self,
        model: Any,
        batch: Any,
        timestep_index: int,
        advantages: Any,
    ) -> tuple[Any, TrainStepMetrics]:
        """One ``(t, eps)`` pair of the surrogate for a rollout batch.

        ``t`` is the rollout grid's ``timesteps[:, timestep_index]``; ``eps`` is
        the group-shared draw for this index and update.
        """

        import torch

        from vrl.trajectory.reader import TrajectoryReader

        cfg = self.config
        replay = TrajectoryReader.from_batch(batch).forward_process_replay(
            "denoise", timestep_index
        )
        x0 = replay.latents_clean
        t = self.flow_time(replay.timestep, x0)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = self._group_shared_noise(
            x0,
            group_ids=getattr(batch, "group_ids", None),
            timestep_index=int(timestep_index),
        )
        xt = (1 - t_expanded) * x0.float() + t_expanded * noise
        xt_input = xt.to(x0.dtype)
        with model_autocast(model, x0.device):
            prediction = model.replay_forward_with_latents(
                batch, timestep_index, xt_input, classifier_free_guidance=False
            )["noise_pred"]

        # x-prediction reparameterization of the rectified-flow velocity.
        x_pred = xt - t_expanded * prediction.float()
        surrogate = self.normalized_mse(x_pred, x0.float())  # [B], adaptive weighting
        # theta_old is the detached current policy: the ratio is 1 in value and
        # carries the surrogate's gradient.
        ratio = torch.exp(surrogate.detach() - surrogate)

        adv = advantages.to(device=x0.device, dtype=ratio.dtype)
        if cfg.adv_soft_clip is not None:
            eta = float(cfg.adv_soft_clip)
            adv = eta * torch.tanh(adv / eta)
        loss = -(ratio * adv).mean()
        loss_value = float(loss.detach().item())
        return loss, TrainStepMetrics(loss=loss_value, policy_loss=loss_value)

    def _group_shared_noise(
        self,
        x0: Any,
        *,
        group_ids: Any,
        timestep_index: int,
    ) -> Any:
        """One ``eps`` per prompt group, identical across the group's samples.

        Seeded from ``(update counter, group id, replay index)`` so the pairs a
        group is scored on are shared within an update and fresh across them.
        The draw runs on the CPU generator so it is device-independent.
        """

        import torch

        batch = int(x0.shape[0])
        if group_ids is None:
            ids = [0] * batch
        else:
            ids = [int(value) for value in torch.as_tensor(group_ids).reshape(-1).tolist()]
        shape = tuple(x0.shape[1:])
        draws: dict[int, Any] = {}
        rows = []
        for group in ids:
            if group not in draws:
                seed = (
                    self._update_counter * _SEED_UPDATE
                    + group * _SEED_GROUP
                    + timestep_index * _SEED_INDEX
                ) & 0x7FFFFFFF
                generator = torch.Generator(device="cpu").manual_seed(seed)
                draws[group] = torch.randn(shape, generator=generator, dtype=torch.float32)
            rows.append(draws[group])
        return torch.stack(rows, dim=0).to(device=x0.device)

    # -- lifecycle ------------------------------------------------------------

    def after_optimizer_step(self, model: Any, global_step: int) -> None:
        # Advance the group-noise counter so the shared noise changes across
        # updates while staying fixed within one.
        del model
        self._update_counter = int(global_step) + 1


__all__ = ["VGRPO", "VGRPOConfig"]
