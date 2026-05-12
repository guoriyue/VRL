"""Algorithm adapters for trajectory-native training inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.algorithms.types import TrainStepMetrics
from vrl.engine.trajectory import TrainingView, TrajectoryBatch
from vrl.rollouts.evaluators.trajectory import (
    old_log_probs_from_trajectory_signals,
    to_trajectory_signals,
    trajectory_signals_to_signal_batch,
)
from vrl.rollouts.evaluators.types import SignalBatch, TrajectorySignalBatch


@dataclass(slots=True)
class AlgorithmInput:
    """Unified algorithm-facing input derived from trajectory contracts."""

    trajectory: TrajectoryBatch | None = None
    training_view: TrainingView | None = None
    signals: TrajectorySignalBatch | SignalBatch | None = None
    rewards: Any | None = None
    group_ids: Any | None = None
    advantages: Any | None = None
    old_log_probs: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AlgorithmAdapter:
    """Project AlgorithmInput into existing objective-specific APIs."""

    def compute_advantages(self, algorithm: Any, inputs: AlgorithmInput) -> Any:
        if inputs.advantages is not None:
            return inputs.advantages
        if inputs.rewards is None:
            raise RuntimeError("AlgorithmInput.rewards is required to compute advantages")
        group_ids = inputs.group_ids
        if group_ids is None and inputs.signals is not None:
            signals = _ensure_trajectory_signals(inputs)
            group_ids = signals.group_ids
        if group_ids is None and inputs.trajectory is not None:
            group_ids = inputs.trajectory.group_ids
        if group_ids is None:
            raise RuntimeError("AlgorithmInput.group_ids is required to compute advantages")
        return algorithm.compute_advantages_from_tensors(inputs.rewards, group_ids)

    def compute_loss(
        self,
        algorithm: Any,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        dpo_loss = _try_compute_dpo_loss(algorithm, inputs)
        if dpo_loss is not None:
            return dpo_loss

        compute_batch_timestep_loss = getattr(algorithm, "compute_batch_timestep_loss", None)
        uses_evaluator = bool(getattr(algorithm, "uses_evaluator", True))
        if not uses_evaluator and callable(compute_batch_timestep_loss):
            return _compute_diffusion_nft_loss(
                algorithm,
                inputs,
                advantages=self.compute_advantages(algorithm, inputs),
            )

        signals = _ensure_trajectory_signals(inputs)
        legacy_signals = trajectory_signals_to_signal_batch(
            signals,
            mask_key=_mask_key(algorithm),
        )
        old_log_probs = old_log_probs_from_trajectory_signals(signals)
        advantages = self.compute_advantages(algorithm, inputs)
        return algorithm.compute_signal_loss(legacy_signals, advantages, old_log_probs)


def _ensure_trajectory_signals(inputs: AlgorithmInput) -> TrajectorySignalBatch:
    if inputs.signals is None:
        raise RuntimeError("AlgorithmInput.signals is required for signal-based algorithms")
    return to_trajectory_signals(
        inputs.signals,
        trajectory=inputs.trajectory,
        training_view=inputs.training_view,
        old_log_probs=inputs.old_log_probs,
        group_ids=inputs.group_ids,
        context=inputs.metadata.get("signal_context"),
        mask_key=inputs.metadata.get("mask_key", "token_mask"),
    )


def _mask_key(algorithm: Any) -> str:
    config = getattr(algorithm, "config", None)
    return str(getattr(config, "mask_key", "token_mask"))


def _compute_diffusion_nft_loss(
    algorithm: Any,
    inputs: AlgorithmInput,
    *,
    advantages: Any,
) -> tuple[Any, TrainStepMetrics]:
    model = inputs.metadata.get("model")
    batch = inputs.metadata.get("rollout_batch")
    timestep_index = int(inputs.metadata.get("timestep_index", 0))
    if model is None:
        raise RuntimeError("DiffusionNFT AlgorithmInput.metadata['model'] is required")
    if batch is None:
        raise RuntimeError("DiffusionNFT AlgorithmInput.metadata['rollout_batch'] is required")
    return algorithm.compute_batch_timestep_loss(model, batch, timestep_index, advantages)


def _try_compute_dpo_loss(
    algorithm: Any,
    inputs: AlgorithmInput,
) -> tuple[Any, TrainStepMetrics] | None:
    from vrl.algorithms.dpo import DiffusionDPOConfig, diffusion_dpo_loss, diffusion_sft_loss

    if not isinstance(algorithm, DiffusionDPOConfig):
        return None
    model_pred = inputs.metadata.get("model_pred")
    ref_pred = inputs.metadata.get("ref_pred")
    target = inputs.metadata.get("target")
    if model_pred is None or ref_pred is None or target is None:
        raise RuntimeError(
            "DiffusionDPO AlgorithmInput.metadata requires model_pred, ref_pred, and target",
        )
    beta = float(inputs.metadata.get("beta", algorithm.beta))
    out = diffusion_dpo_loss(model_pred, ref_pred, target, beta=beta)
    loss = out["loss"]
    if algorithm.sft_weight > 0:
        half = model_pred.shape[0] // 2
        loss = loss + float(algorithm.sft_weight) * diffusion_sft_loss(
            model_pred[:half],
            target[:half],
        )
    return loss, TrainStepMetrics(
        loss=float(loss.detach().item()),
        policy_loss=float(out["loss"].detach().item()),
        approx_kl=float(out["raw_model_loss"].detach().item()),
        clip_fraction=float(out["implicit_acc"].detach().item()),
    )


__all__ = [
    "AlgorithmAdapter",
    "AlgorithmInput",
]
