"""SDE log-probability signal extraction for diffusion model training."""

from __future__ import annotations

import contextlib

import vrl.math.denoise.flow_matching as flow_matching_math
from vrl.models.interfaces import (
    ReplayModel,
    require_replay_model,
)
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
        sde_type: str = "flow_grpo",
        math_dtype: object = None,
    ) -> None:
        self.scheduler = scheduler
        self.noise_level = noise_level
        self.sde_type = sde_type
        # dtype for the replay log-prob math (``math`` precision axis);
        # None -> fp32 (the protected default).
        self.math_dtype = math_dtype

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

        resolver = TrajectoryResolver.from_batch(batch)

        fwd = model.replay_forward(batch, timestep_idx).require_segment("denoise")
        noise_pred = fwd.require_value("noise_pred")
        device = getattr(noise_pred, "device", None)
        replay = resolver.replay_tensor_dict(
            "denoise",
            axis="denoise",
            axis_index=timestep_idx,
        )
        t = move_value_to_device(replay["timesteps"], device)
        observations = move_value_to_device(
            replay[resolver.role_tensor("denoise", "observation").name],
            device,
        )
        actions = move_value_to_device(
            replay[resolver.role_tensor("denoise", "action").name],
            device,
        )

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
            math_dtype=self.math_dtype,
        )

        ref_log_prob = None
        ref_prev_sample_mean = None
        ref_sqrt_neg_dt = None

        # Reference model signal for KL. The ref forward is frozen, so generation
        # can cache its noise_pred (sampling.cache_ref_noise_pred). When present we
        # run the same sde_step_with_logprob on the cached tensor instead of
        # rerunning the ref forward every ppo_epoch (Lever D); the math is
        # identical, only the transformer forward is skipped.
        if signal_request.need_ref:
            cached_ref_noise_pred = self._cached_ref_noise_pred(
                batch,
                timestep_idx,
                device,
            )
            with torch.no_grad():
                if cached_ref_noise_pred is not None:
                    ref_result = flow_matching_math.sde_step_with_logprob(
                        self.scheduler,
                        cached_ref_noise_pred,
                        t,
                        observations,
                        prev_sample=actions,
                        return_dt=signal_request.need_kl_intermediates,
                        noise_level=self.noise_level,
                        sde_type=self.sde_type,
                        math_dtype=self.math_dtype,
                    )
                    ref_log_prob = ref_result.log_prob
                    ref_prev_sample_mean = ref_result.prev_sample_mean
                    ref_sqrt_neg_dt = ref_result.sqrt_neg_dt
                elif ref_model is not None:
                    # ReplayModel.disable_adapter() may be a no-op for non-adapter
                    # models. A distinct frozen reference still comes through
                    # the explicit ref_model path.
                    use_adapter_disable = ref_model is model
                    ctx = (
                        model.disable_adapter()
                        if use_adapter_disable
                        else contextlib.nullcontext()
                    )

                    with ctx:
                        ref_fwd = ref_model.replay_forward(
                            batch,
                            timestep_idx,
                        ).require_segment("denoise")
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
                        math_dtype=self.math_dtype,
                    )
                    ref_log_prob = ref_result.log_prob
                    ref_prev_sample_mean = ref_result.prev_sample_mean
                    ref_sqrt_neg_dt = ref_result.sqrt_neg_dt

        # Rollout-time proposal mean for this step, captured at generation
        # (return_prev_sample_mean) and replayed back unchanged. Trust-region
        # losses (Flow-DPPO / GRPO-Guard) read it; None for recipes that did not
        # opt in. Sliced to this step to match result.prev_sample_mean's shape.
        old_prev_sample_mean = self._old_prev_sample_mean(batch, timestep_idx, device)

        return TrajectorySignalBuilder(batch).single_segment(
            segment_name="denoise",
            log_prob=result.log_prob,
            ref_log_prob=ref_log_prob,
            prev_sample_mean=result.prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            old_prev_sample_mean=old_prev_sample_mean,
            std_dev_t=result.std_dev_t,
            dt=result.sqrt_neg_dt if result.sqrt_neg_dt is not None else ref_sqrt_neg_dt,
            distribution="flow_matching",
            timestep_idx=timestep_idx,
            mask_key="mask",
        )

    @staticmethod
    def _old_prev_sample_mean(batch: RolloutBatch, timestep_idx: int, device: object) -> object:
        """Rollout proposal mean for ``timestep_idx`` from the trajectory, or None.

        Stored as a denoise replay tensor at generation; absent unless the recipe
        set return_prev_sample_mean. Shaped ``[B, num_steps, *latent]`` -> sliced
        to ``[B, *latent]`` so it lines up with the replayed prev_sample_mean.
        """

        from vrl.trajectory import TrajectoryResolver

        replay = TrajectoryResolver.from_batch(batch).replay_tensor_dict("denoise")
        stored = replay.get("old_prev_sample_mean")
        if stored is None:
            return None
        step = stored[:, timestep_idx] if getattr(stored, "ndim", 0) > 1 else stored
        return move_value_to_device(step, device)

    @staticmethod
    def _cached_ref_noise_pred(batch: RolloutBatch, timestep_idx: int, device: object) -> object:
        """Frozen reference noise_pred for ``timestep_idx`` from the trajectory, or None.

        Stored as a denoise replay tensor at generation when the recipe set
        sampling.cache_ref_noise_pred; absent otherwise. Shaped
        ``[B, num_steps, *latent]`` -> sliced to ``[B, *latent]`` so it lines up
        with the replayed observations/actions for sde_step_with_logprob.
        """

        from vrl.trajectory import TrajectoryResolver

        replay = TrajectoryResolver.from_batch(batch).replay_tensor_dict("denoise")
        stored = replay.get("ref_noise_pred")
        if stored is None:
            return None
        step = stored[:, timestep_idx] if getattr(stored, "ndim", 0) > 1 else stored
        return move_value_to_device(step, device)
