"""SDE log-probability signal extraction for diffusion model training."""

from __future__ import annotations

import contextlib

import vrl.math.diffusion.flow_matching as flow_matching_math
from vrl.models.interfaces import ReplayModel, require_replay_model
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
from vrl.trajectory.device import move_value_to_device


class DiffusionSDELogProbEvaluator(Evaluator):
    """Signal extraction for flow-matching diffusion models.

    Uses ``sde_step_with_logprob`` to compute log-probabilities and
    optionally reference model signals for latent-space KL.
    """

    supports_deferred_replay_tensor_move = True

    def __init__(
        self,
        scheduler: object,
        noise_level: float = 1.0,
        sde_type: str = "sde",
    ) -> None:
        self.scheduler = scheduler
        self.noise_level = noise_level
        self.sde_type = sde_type

    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int,
        ref_model: ReplayModel | None = None,
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        """Replay one diffusion step into trajectory-native signals.

        Replay forward ownership lives on the family model. ``model`` must
        satisfy the trainer-facing ReplayModel contract.

        When ref_model is the same object as model (LoRA scenario),
        uses ``disable_adapter()`` to get base-model predictions —
        matching flow_grpo train_wan2_1.py:940.
        """
        import torch

        model = require_replay_model(model, owner="DiffusionSDELogProbEvaluator.model")
        if ref_model is not None:
            ref_model = require_replay_model(
                ref_model,
                owner="DiffusionSDELogProbEvaluator.ref_model",
            )
        if signal_request is None:
            signal_request = SignalRequest()

        from vrl.trajectory import TrajectoryResolver

        timesteps = TrajectoryResolver.from_batch(batch).tensor_value(
            "denoise",
            "timesteps",
        )
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        observations = batch.observations[:, timestep_idx]  # x_t
        actions = batch.actions[:, timestep_idx]             # x_{t-1}

        fwd = model.replay_forward(batch, timestep_idx).require_segment("denoise")
        noise_pred = fwd.require_value("noise_pred")
        device = getattr(noise_pred, "device", None)
        t = move_value_to_device(t, device)
        observations = move_value_to_device(observations, device)
        actions = move_value_to_device(actions, device)

        # SDE step with log-prob
        result = flow_matching_math.sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            t,
            observations,
            prev_sample=actions,
            return_dt=signal_request.need_kl_intermediates,
            noise_level=self.noise_level,
            sde_type=self.sde_type,
        )

        ref_log_prob = None
        ref_prev_sample_mean = None
        ref_sqrt_neg_dt = None

        # Reference model forward for KL
        if signal_request.need_ref and ref_model is not None:
            with torch.no_grad():
                # ReplayModel.disable_adapter() may be a no-op for non-adapter
                # models. A distinct frozen reference still comes through
                # the explicit ref_model path.
                use_adapter_disable = ref_model is model
                ctx = model.disable_adapter() if use_adapter_disable else contextlib.nullcontext()

                with ctx:
                    ref_fwd = ref_model.replay_forward(batch, timestep_idx).require_segment(
                        "denoise",
                    )
                    ref_noise_pred = ref_fwd.require_value("noise_pred")

                    ref_result = flow_matching_math.sde_step_with_logprob(
                        self.scheduler,
                        ref_noise_pred,
                        t,
                        observations,
                        prev_sample=actions,
                        return_dt=signal_request.need_kl_intermediates,
                        noise_level=self.noise_level,
                        sde_type=self.sde_type,
                    )
                    ref_log_prob = ref_result.log_prob
                    ref_prev_sample_mean = ref_result.prev_sample_mean
                    ref_sqrt_neg_dt = ref_result.sqrt_neg_dt

        return TrajectorySignalBuilder(batch).single_segment(
            segment_name="denoise",
            log_prob=result.log_prob,
            ref_log_prob=ref_log_prob,
            prev_sample_mean=result.prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            std_dev_t=result.std_dev_t,
            dt=result.sqrt_neg_dt if result.sqrt_neg_dt is not None else ref_sqrt_neg_dt,
            distribution="flow_matching",
            timestep_idx=timestep_idx,
            mask_key="mask",
        )
