"""Multi-segment token GRPO for Janus-Pro-R1 style rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import SignalBatch


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

    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,
        old_log_probs: Any,
    ) -> tuple[Any, TrainStepMetrics]:
        segments = (signals.aux or {}).get("segments")
        if not isinstance(segments, dict):
            return super().compute_signal_loss(signals, advantages, old_log_probs)
        old_by_segment = self._old_log_probs_by_segment(signals, old_log_probs)

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
        for name, weight_raw in self.config.segment_weights.items():
            if not bool(train_segments.get(name, True)):
                continue
            weight = float(weight_raw)
            if weight <= 0:
                continue
            segment_signal = segments.get(name)
            segment_old = old_by_segment.get(name)
            if segment_signal is None or segment_old is None:
                raise RuntimeError(f"missing multi-segment GRPO segment: {name}")
            loss, metrics = super().compute_signal_loss(
                segment_signal,
                advantages,
                segment_old,
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
            zero = signals.log_prob.sum() * 0.0
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

    @staticmethod
    def _old_log_probs_by_segment(
        signals: SignalBatch,
        old_log_probs: Any,
    ) -> dict[str, torch.Tensor]:
        aux_old = (signals.aux or {}).get("old_log_probs")
        if isinstance(aux_old, dict):
            return aux_old
        if isinstance(old_log_probs, dict):
            return old_log_probs
        return {}
