"""Multi-segment token GRPO for Janus-Pro-R1 style rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import TrajectorySignalBatch


@dataclass(slots=True)
class MultiSegmentTokenGRPOConfig(TokenGRPOConfig):
    """TokenGRPO config with weighted segment losses."""

    segment_weights: dict[str, float] = field(
        default_factory=lambda: {
            "initial_image": 1.0,
            "selfcheck_text": 0.0,
            "final_image": 1.0,
        },
    )
    train_segments: dict[str, bool] = field(
        default_factory=lambda: {
            "initial_image": True,
            "selfcheck_text": False,
            "final_image": True,
        },
    )


class MultiSegmentTokenGRPO(TokenGRPO):
    """Apply TokenGRPO independently per segment, then average by weight."""

    def __init__(self, config: MultiSegmentTokenGRPOConfig | None = None) -> None:
        cfg = config or MultiSegmentTokenGRPOConfig()
        super().__init__(cfg)
        self.config: MultiSegmentTokenGRPOConfig = cfg
        self.last_segment_metrics: dict[str, TrainStepMetrics] = {}

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        if inputs.signals is None:
            raise RuntimeError("AlgorithmInput.signals is required for MultiSegmentTokenGRPO")
        signals = inputs.signals

        total_loss: torch.Tensor | None = None
        metric_values: dict[str, list[float]] = {
            "loss": [],
            "policy_loss": [],
            "kl_penalty": [],
            "clip_fraction": [],
            "approx_kl": [],
        }
        total_weight = 0.0
        self.last_segment_metrics = {}
        train_segments = dict(self.config.train_segments or {})
        weights = dict(self.config.segment_weights or {})
        for name in _ordered_segment_names(inputs, weights):
            if not bool(train_segments.get(name, True)):
                continue
            weight = float(weights.get(name, 1.0))
            if weight <= 0:
                continue
            segment_signal = signals.segments.get(name)
            if segment_signal is None:
                raise RuntimeError(f"missing multi-segment GRPO segment: {name}")
            segment_advantages = _segment_advantages(inputs.advantages, name)
            loss, metrics = super().compute_loss(
                AlgorithmInput(
                    trajectory=inputs.trajectory,
                    training_view=inputs.training_view,
                    signals=TrajectorySignalBatch(
                        segments={name: segment_signal},
                        group_ids=signals.group_ids,
                        context=signals.context,
                        primary_segment=name,
                    ),
                    rewards=inputs.rewards,
                    group_ids=inputs.group_ids,
                    advantages=segment_advantages,
                    old_log_probs=inputs.old_log_probs,
                    metadata=inputs.metadata,
                ),
            )
            self.last_segment_metrics[name] = metrics
            weighted = loss * weight
            total_loss = weighted if total_loss is None else total_loss + weighted
            total_weight += weight
            metric_values["loss"].append(metrics.loss * weight)
            metric_values["policy_loss"].append(metrics.policy_loss * weight)
            metric_values["kl_penalty"].append(metrics.kl_penalty * weight)
            metric_values["clip_fraction"].append(metrics.clip_fraction * weight)
            metric_values["approx_kl"].append(metrics.approx_kl * weight)

        if total_loss is None or total_weight <= 0:
            zero = signals.primary.log_prob.sum() * 0.0
            return zero, TrainStepMetrics()

        total_loss = total_loss / total_weight

        def _weighted_avg(key: str) -> float:
            values = metric_values[key]
            if not values:
                return 0.0
            return sum(values) / total_weight

        return total_loss, TrainStepMetrics(
            loss=float(total_loss.item()),
            policy_loss=_weighted_avg("policy_loss"),
            kl_penalty=_weighted_avg("kl_penalty"),
            clip_fraction=_weighted_avg("clip_fraction"),
            approx_kl=_weighted_avg("approx_kl"),
        )


def _ordered_segment_names(inputs: AlgorithmInput, weights: dict[str, float]) -> list[str]:
    if inputs.training_view is not None and inputs.training_view.loss_units:
        out: list[str] = []
        seen: set[str] = set()
        for unit in inputs.training_view.loss_units:
            if unit.segment not in seen:
                out.append(unit.segment)
                seen.add(unit.segment)
        return out

    if weights:
        return list(weights)
    if inputs.signals is None:
        return []
    return list(inputs.signals.segments)


def _segment_advantages(advantages: Any, name: str) -> Any:
    if isinstance(advantages, dict):
        if name in advantages:
            return advantages[name]
        if "__default__" in advantages:
            return advantages["__default__"]
        raise RuntimeError(f"missing multi-segment advantages for segment: {name}")
    return advantages
