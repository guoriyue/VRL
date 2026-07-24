"""Build trainer rollout batches from trajectory-backed generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation import GenerationOutput
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.rewards import RewardScoringInput
from vrl.trajectory import (
    RewardView,
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryStoragePolicy,
    apply_trajectory_storage_policy,
    named_tensor,
    role_tensor,
)


@dataclass(slots=True)
class RolloutBatchBuildContext:
    """Non-engine metadata needed while building a trainer RolloutBatch."""

    metadata: dict[str, Any]
    device: Any | None = None
    kl_reward_coef: float = 0.0
    trajectory_storage_policy: TrajectoryStoragePolicy = field(
        default_factory=TrajectoryStoragePolicy,
    )
    # Resolved family PolicySemantics.trajectory_layout, threaded from the
    # collector's ModelFamilyEntry. "multisegment_token" selects the per-segment
    # AR pack path; None falls back to the generic all-categorical heuristic.
    trajectory_layout: str | None = None


class TrajectoryRolloutBatchBuilder:
    """Convert one trajectory-backed GenerationOutput into reward and trainer inputs."""

    def __init__(
        self,
        output: GenerationOutput,
        context: RolloutBatchBuildContext,
    ) -> None:
        self.output = output
        self.context = context
        self.trajectory = apply_trajectory_storage_policy(
            self._require_output_trajectory(output),
            context.trajectory_storage_policy,
        )
        self.output.trajectory = self.trajectory

    def reward_scoring_input(self) -> RewardScoringInput:
        reward_outputs = self.reward_outputs()
        return RewardScoringInput(
            outputs=reward_outputs,
            source_request_id=self.output.request_id,
            sample_rows=tuple(self.output.sample_rows),
            metadata=dict(self.context.metadata),
            device=self._infer_device(reward_outputs),
        )

    def reward_outputs(self) -> Any:
        """Return the selected artifact normalized to [0, 1] per the reward view's range."""

        view = self._reward_view()
        reward_output = self._reward_output(view)
        if isinstance(reward_output, torch.Tensor) and reward_output.dtype == torch.uint8:
            # Worker-side wire packing (see decode_denoise_result): decoded
            # video crosses the wire as uint8. k/255 reconstruction round-trips
            # bit-exactly through every downstream to_uint8 quantization, so
            # reward scores are unchanged.
            return reward_output.float() / 255.0
        if view.value_range == "tanh":
            reward_output = ((reward_output + 1.0) * 0.5).clamp(0.0, 1.0)
        return reward_output

    def build(self, rewards_raw: torch.Tensor) -> RolloutBatch:
        """Convert the engine output and reward tensor into a trainer batch."""

        trainable = self._trainable_segments()
        if not trainable:
            raise RuntimeError(
                "generation-only trajectory cannot build a trainer RolloutBatch: "
                "no trainable policy segment or replay facts were recorded",
            )
        if self._is_multisegment_categorical(trainable):
            return self._pack_ar_multisegment(rewards_raw)
        segment = self._primary_trainable_segment()
        if segment.distribution == "flow_matching" or (
            segment.distribution == "gaussian" and segment.modality == "latent"
        ):
            return self._pack_diffusion(segment, rewards_raw)
        if segment.distribution in ("categorical", "gaussian"):
            return self._pack_ar_tokens(segment, rewards_raw)
        raise NotImplementedError(
            "trajectory rollout collection does not support distribution="
            f"{segment.distribution!r}",
        )

    def _pack_diffusion(
        self,
        segment: TrajectorySegment,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        observations = role_tensor(segment, "observation").value
        kl_tensor = named_tensor(segment, "kl").value
        device = observations.device

        if self.context.kl_reward_coef > 0:
            kl_per_sample = kl_tensor.reshape(kl_tensor.shape[0], -1).sum(dim=1)
            rewards_adjusted = rewards_raw.to(device) - self.context.kl_reward_coef * kl_per_sample
        else:
            rewards_adjusted = rewards_raw.to(device)

        rollout_context = dict(self.trajectory.context)
        if self.context.metadata:
            rollout_context["reward_metadata"] = dict(self.context.metadata)
        runtime_debug = self.output.extra.get("runtime_debug")
        if runtime_debug is not None:
            rollout_context["runtime_debug"] = runtime_debug

        return RolloutBatch(
            rewards=rewards_adjusted,
            group_ids=self._group_ids(device=device),
            extras={},
            context=rollout_context,
            trajectory=self.trajectory,
        )

    def _pack_ar_tokens(
        self,
        segment: TrajectorySegment,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        prompt_ids = named_tensor(segment, "prompt_input_ids").value
        device = self.context.device or prompt_ids.device

        return RolloutBatch(
            rewards=rewards_raw.to(device),
            group_ids=self._group_ids(device=device),
            extras={},
            context=dict(self.trajectory.context),
            trajectory=self.trajectory,
        )

    def _pack_ar_multisegment(
        self,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        primary = self._primary_trainable_segment()

        token_ids = role_tensor(primary, "action").value
        device = self.context.device or token_ids.device

        return RolloutBatch(
            rewards=rewards_raw.to(device),
            group_ids=self._group_ids(device=device),
            extras={},
            context=dict(self.trajectory.context),
            trajectory=self.trajectory,
        )

    def _reward_output(self, view: RewardView) -> Any:
        if view.tensor_refs:
            if len(view.tensor_refs) != 1:
                raise RuntimeError(
                    f"RewardView {view.name!r} must expose exactly one tensor_ref "
                    "for collector reward scoring",
                )
            return self._tensor_value_from_ref(view.tensor_refs[0])

        output_ref = view.metadata.get("output_ref")
        if output_ref == "GenerationOutput.output":
            return self.output.output
        raise RuntimeError(
            f"RewardView {view.name!r} has no tensor_refs and no supported output_ref",
        )

    def _reward_view(self) -> RewardView:
        reward_views = self.trajectory.reward_views
        if len(reward_views) == 1:
            return next(iter(reward_views.values()))
        if not reward_views:
            raise RuntimeError(
                f"TrajectoryBatch {self.trajectory.request_id!r} has no reward views",
            )
        raise RuntimeError(
            f"TrajectoryBatch {self.trajectory.request_id!r} has multiple reward "
            "views; every generated trajectory must declare exactly one scoring view",
        )

    def _tensor_value_from_ref(self, ref: str) -> Any:
        segment_name, tensor_name = ref.split(".", 1)
        try:
            segment = self.trajectory.segments[segment_name]
            return segment.tensors[tensor_name].value
        except KeyError as exc:
            raise RuntimeError(
                f"RewardView references unknown trajectory tensor {ref!r}",
            ) from exc

    def _trainable_segments(self) -> list[TrajectorySegment]:
        return [segment for segment in self.trajectory.segments.values() if segment.trainable]

    def _primary_trainable_segment(
        self,
    ) -> TrajectorySegment:
        primary_name = self.trajectory.primary_segment
        if primary_name is None:
            raise RuntimeError("TrajectoryBatch has no primary trainable segment")
        segment = self.trajectory.segments.get(primary_name)
        if segment is None or not segment.trainable:
            raise RuntimeError(
                "TrajectoryBatch.primary_segment must reference a trainable segment",
            )
        return segment

    def _group_ids(self, *, device: Any) -> torch.Tensor:
        return torch.tensor(
            [row.prompt_index for row in self.output.sample_rows],
            dtype=torch.long,
            device=device,
        )

    def _is_multisegment_categorical(
        self,
        trainable: list[TrajectorySegment],
    ) -> bool:
        if self.context.trajectory_layout == "multisegment_token":
            return True
        return len(trainable) > 1 and all(
            segment.distribution == "categorical" for segment in trainable
        )

    def _infer_device(self, value: Any) -> Any:
        if self.context.device is not None:
            return self.context.device
        device = getattr(value, "device", None)
        if device is not None:
            return device
        return "cpu"

    @staticmethod
    def _require_output_trajectory(output: GenerationOutput) -> TrajectoryBatch:
        trajectory = output.trajectory
        if not isinstance(trajectory, TrajectoryBatch):
            raise RuntimeError(
                f"GenerationOutput {output.request_id!r} is missing TrajectoryBatch",
            )
        return trajectory


__all__ = [
    "RolloutBatchBuildContext",
    "TrajectoryRolloutBatchBuilder",
]
