"""On-policy controller updates from completed bounded episodes.

The editor is frozen. Each controller decision is credited with the tool costs
and terminal score that follow it; other episodes of the same task supply a
leave-one-out baseline. The loss is GRPO's clipped surrogate over categorical
signals, without a reference KL.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from agentic.controller import CategoricalController
from agentic.episode import Episode, PolicyStamp
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.models.parking import ModelParking
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch
from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state
from vrl.utils.artifacts import atomic_file
from vrl.utils.json_files import canonical_json_sha256

CHECKPOINT_SCHEMA = "vrl.visual-controller-checkpoint.v2"


class ControllerTrainer:
    """One optimizer owner; collect a fresh task group before each update."""

    def __init__(
        self,
        controller: CategoricalController,
        *,
        learning_rate: float = 1e-4,
        clip_ratio: float = 0.2,
        max_grad_norm: float = 1.0,
        replay_tolerance: float = 1e-5,
    ) -> None:
        self.controller = controller
        self.parameters = [p for p in controller.model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("controller has no trainable parameters")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate)
        self.algorithm = GRPO(GRPOConfig(clip_ratio=clip_ratio, kl_coef=0.0))
        self.max_grad_norm = max_grad_norm
        self.replay_tolerance = replay_tolerance
        self._optimizer_parking = ModelParking()
        self._pending_optimizer_state: dict[str, Any] | None = None
        self.config = {
            "learning_rate": learning_rate,
            "clip_ratio": clip_ratio,
            "max_grad_norm": max_grad_norm,
            "replay_tolerance": replay_tolerance,
        }

    def save_checkpoint(
        self, path: Path, *, contract: dict[str, Any], progress: dict[str, Any]
    ) -> None:
        """Policy weights, optimizer state, process RNG and the run's contract, together."""

        if not self.controller.base_identity:
            raise ValueError("controller checkpoint requires a base identity")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "base_identity": self.controller.base_identity,
            "policy": asdict(self.controller.policy_stamp),
            "contract": contract,
            "progress": progress,
            "config": self.config,
            "temperature": self.controller.temperature,
            "max_image_pixels": self.controller.max_image_pixels,
            "observation_mode": self.controller.observation_mode,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.controller.model.named_parameters()
                if value.requires_grad
            },
            "optimizer": self._pending_optimizer_state or self.optimizer.state_dict(),
            "rng": capture_rng_state(),
        }
        with atomic_file(path, binary=True, overwrite=False) as handle:
            torch.save(payload, handle)

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Restore weights, optimizer, RNG and controller settings; return the progress."""

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["schema"] != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported controller checkpoint schema {payload['schema']!r}")
        if payload["base_identity"] != self.controller.base_identity:
            raise ValueError("controller checkpoint was trained on a different base model")
        parameters = {
            name: value
            for name, value in self.controller.model.named_parameters()
            if value.requires_grad
        }
        saved = payload["parameters"]
        # Optimizer state indexes parameters by traversal order, not by name.
        if list(parameters) != list(saved):
            raise ValueError("controller checkpoint trainable parameters differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved[name])
        self.controller.restore_policy_stamp(PolicyStamp(**payload["policy"]))
        self.controller.temperature = payload["temperature"]
        self.controller.max_image_pixels = payload["max_image_pixels"]
        self.controller.observation_mode = payload.get("observation_mode", "rgb")
        self._optimizer_parking.discard()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()
        # Loaded once the parameters are on their device, so moment tensors land there.
        self._pending_optimizer_state = payload["optimizer"]
        restore_rng_state(payload["rng"])
        return payload["progress"]

    async def update(self, traces: list[dict[str, Any]]) -> dict[str, Any]:
        """One clipped policy-gradient step over a group of fresh on-policy episodes."""

        validated = [
            Episode.training_steps(trace, policy=self.controller.policy_stamp) for trace in traces
        ]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, trace in enumerate(traces):
            groups[canonical_json_sha256(trace["task"], allow_nan=False)].append(index)
        if not traces or any(len(indices) < 2 for indices in groups.values()):
            raise ValueError("controller updates need at least two fresh episodes per task")
        baselines = {}
        for indices in groups.values():
            for index in indices:
                # Other rollouts of the same task: an action-independent baseline.
                baselines[index] = sum(
                    traces[other]["discounted_return"] for other in indices if other != index
                ) / (len(indices) - 1)
        entropies, stop_probabilities = [], []
        self.optimizer.zero_grad(set_to_none=True)
        max_error, loss_sum, decision_count = 0.0, 0.0, 0
        try:
            await self.controller.activate()
            self._optimizer_parking.restore()
            if self._pending_optimizer_state is not None:
                self.optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
            for index, (task, steps) in enumerate(validated):
                for observation, decision, value in steps:
                    log_prob = self.controller.replay_log_prob(
                        task, observation, decision
                    ).reshape(1)
                    distribution = decision.replay.get("action_log_probs")
                    if distribution is not None:
                        entropies.append(-sum(math.exp(v) * v for v in distribution))
                        stop_probabilities.append(
                            sum(
                                math.exp(v)
                                for action, v in zip(task.actions, distribution, strict=True)
                                if action.kind == "stop"
                            )
                        )
                    error = abs(log_prob.detach().item() - decision.old_log_prob)
                    max_error = max(max_error, error)
                    if error > self.replay_tolerance:
                        raise ValueError(f"controller replay parity failed: {error}")
                    advantage = (value - baselines[index]) * traces[index][
                        "gamma"
                    ] ** observation.step
                    segment = SegmentSignal(
                        name="controller",
                        distribution="categorical",
                        log_prob=log_prob,
                        old_log_prob=log_prob.new_tensor([decision.old_log_prob]),
                        mask=torch.ones_like(log_prob),
                    )
                    loss, _ = self.algorithm.compute_loss(
                        AlgorithmInput(
                            signals=TrajectorySignalBatch(
                                segments={"controller": segment},
                                group_ids=torch.zeros(1, dtype=torch.long, device=log_prob.device),
                            ),
                            advantages=log_prob.new_tensor([advantage]),
                        )
                    )
                    # Sum decisions within an episode, average over episodes, so a
                    # longer episode does not change the denominator.
                    scaled_loss = loss / len(traces)
                    scaled_loss.backward()
                    loss_sum += scaled_loss.detach().item()
                    decision_count += 1
            norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, self.max_grad_norm, error_if_nonfinite=True
            )
            if norm.item() > 0:
                self.optimizer.step()
                self.controller.advance_policy()
            return {
                "episodes": len(traces),
                "decisions": decision_count,
                "task_groups": len(groups),
                "loss": loss_sum,
                "gradient_norm": norm.item(),
                "replay_abs_error_max": max_error,
                "policy_version": self.controller.policy_stamp.version,
                "optimizer_stepped": norm.item() > 0,
                "action_distribution_decisions": len(entropies),
                "action_entropy_mean": sum(entropies) / len(entropies) if entropies else None,
                "stop_probability_mean": sum(stop_probabilities) / len(stop_probabilities)
                if stop_probabilities
                else None,
            }
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._optimizer_parking.park_tensors(self.optimizer.state)
            await self.controller.park()
