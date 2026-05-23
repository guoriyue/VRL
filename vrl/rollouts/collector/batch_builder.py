"""Build trainer rollout batches from trajectory-backed generation outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation import GenerationOutput
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.artifacts import (
    RewardArtifactPolicy,
    extract_reward_artifact,
)
from vrl.rollouts.collector.rewards import RewardScoringInput
from vrl.trajectory import (
    RewardView,
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryStoragePolicy,
    TrajectoryTensor,
    apply_trajectory_storage_policy,
    build_training_view,
)


@dataclass(slots=True)
class RolloutBatchBuildContext:
    """Non-engine metadata needed while building a trainer RolloutBatch."""

    metadata: dict[str, Any]
    device: Any | None = None
    kl_reward: float = 0.0
    rescale_to_unit: bool = False
    reward_view_name: str | None = None
    trajectory_storage_policy: TrajectoryStoragePolicy = field(
        default_factory=TrajectoryStoragePolicy,
    )
    reward_artifact_policy: RewardArtifactPolicy = field(default_factory=RewardArtifactPolicy)
    extra: dict[str, Any] = field(default_factory=dict)


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

    def reward_scoring_input(
        self,
        metadata: Mapping[str, Any],
    ) -> RewardScoringInput:
        reward_outputs = self.reward_outputs()
        return RewardScoringInput(
            outputs=reward_outputs,
            prompts=[row.prompt for row in self.output.sample_rows],
            metadata=dict(metadata),
            device=self._infer_device(reward_outputs),
            expected_count=len(self.output.sample_rows),
        )

    def reward_outputs(self) -> Any:
        """Return the explicitly selected generated artifact for reward scoring."""

        segment = self._primary_trainable_segment(preferred=self._primary_segment_name())
        reward_output = self._reward_output()
        if segment.distribution == "categorical" and self.context.rescale_to_unit:
            reward_output = (reward_output + 1.0) * 0.5
            reward_output = reward_output.clamp(0.0, 1.0)
        return reward_output

    def build(self, rewards_raw: torch.Tensor) -> RolloutBatch:
        """Convert the engine output and reward tensor into a trainer batch."""

        trainable = self._trainable_segments()
        if self._is_multisegment_categorical(trainable):
            return self._pack_ar_multisegment(trainable, rewards_raw)
        segment = self._primary_trainable_segment(preferred=self._primary_segment_name())
        if segment.distribution == "flow_matching":
            return self._pack_diffusion(segment, rewards_raw)
        if segment.distribution == "categorical":
            return self._pack_ar_discrete(segment, rewards_raw)
        if segment.distribution == "gaussian":
            return self._pack_ar_continuous(segment, rewards_raw)
        raise NotImplementedError(
            "trajectory rollout collection does not support distribution="
            f"{segment.distribution!r}",
        )

    def _pack_diffusion(
        self,
        segment: TrajectorySegment,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        observations = self._role_tensor(segment, "observation").value
        actions = self._role_tensor(segment, "action").value
        kl_tensor = self._named_tensor(segment, "kl").value
        device = observations.device

        if self.context.kl_reward > 0:
            rewards_adjusted = (
                rewards_raw.to(device)
                - self.context.kl_reward * kl_tensor.sum(dim=1)
            )
        else:
            rewards_adjusted = rewards_raw.to(device)

        rollout_context = dict(self.trajectory.context)
        if self.context.metadata:
            rollout_context["reward_metadata"] = dict(self.context.metadata)
        runtime_debug = self.output.extra.get("runtime_debug")
        if runtime_debug is not None:
            rollout_context["runtime_debug"] = runtime_debug

        return RolloutBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=torch.ones(observations.shape[0], dtype=torch.bool, device=device),
            group_ids=self._group_ids(device=device),
            extras={},
            context=rollout_context,
            videos=self.output.output,
            prompts=[row.prompt for row in self.output.sample_rows],
            trajectory=self.trajectory,
            training_view=build_training_view(
                self.trajectory,
                primary_segment=segment.name,
            ),
        )

    def _pack_ar_discrete(
        self,
        segment: TrajectorySegment,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        token_ids = self._role_tensor(segment, "action").value
        prompt_ids = self._named_tensor(segment, "prompt_input_ids").value
        device = self.context.device or prompt_ids.device
        images = self.output.output

        return RolloutBatch(
            observations=prompt_ids.unsqueeze(1),
            actions=token_ids,
            rewards=rewards_raw.to(device),
            dones=torch.ones(
                len(self.output.sample_rows),
                dtype=torch.bool,
                device=device,
            ),
            group_ids=self._group_ids(device=device),
            extras={},
            context=dict(self.trajectory.context),
            videos=images.unsqueeze(2),
            prompts=[row.prompt for row in self.output.sample_rows],
            trajectory=self.trajectory,
            training_view=build_training_view(
                self.trajectory,
                primary_segment=segment.name,
            ),
        )

    def _pack_ar_continuous(
        self,
        segment: TrajectorySegment,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        tokens = self._role_tensor(segment, "action").value
        prompt_ids = self._named_tensor(segment, "prompt_input_ids").value
        device = self.context.device or prompt_ids.device
        images = self.output.output

        return RolloutBatch(
            observations=prompt_ids.unsqueeze(1),
            actions=tokens,
            rewards=rewards_raw.to(device),
            dones=torch.ones(
                len(self.output.sample_rows),
                dtype=torch.bool,
                device=device,
            ),
            group_ids=self._group_ids(device=device),
            extras={},
            context=dict(self.trajectory.context),
            videos=images.unsqueeze(2),
            prompts=[row.prompt for row in self.output.sample_rows],
            trajectory=self.trajectory,
            training_view=build_training_view(
                self.trajectory,
                primary_segment=segment.name,
            ),
        )

    def _pack_ar_multisegment(
        self,
        trainable: list[TrajectorySegment],
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        primary_name = self._primary_segment_name() or "final_image"
        primary = self.trajectory.segments.get(primary_name)
        if primary is None or not primary.trainable:
            primary = trainable[-1]
            primary_name = primary.name

        token_ids = self._role_tensor(primary, "action").value
        prompt_ids = self._optional_named_tensor(primary, "prompt_input_ids")
        if prompt_ids is None:
            prompt_ids = torch.zeros(
                token_ids.shape[0],
                1,
                dtype=torch.long,
                device=token_ids.device,
            )
        device = self.context.device or prompt_ids.device
        final_image = self._decoded_tensor("final_image")
        if final_image is None:
            final_image = self.output.output

        rollout_context = dict(self.trajectory.context)
        rollout_context.pop("primary_segment", None)
        rollout_context.pop("segment_names", None)
        return RolloutBatch(
            observations=prompt_ids.unsqueeze(1),
            actions=token_ids,
            rewards=rewards_raw.to(device),
            dones=torch.ones(
                len(self.output.sample_rows),
                dtype=torch.bool,
                device=device,
            ),
            group_ids=self._group_ids(device=device),
            extras={},
            context={
                **rollout_context,
                "r1_segment_names": tuple(
                    name
                    for name, segment in self.trajectory.segments.items()
                    if segment.distribution == "categorical"
                ),
            },
            videos=final_image.unsqueeze(2),
            prompts=[row.prompt for row in self.output.sample_rows],
            trajectory=self.trajectory,
            training_view=build_training_view(
                self.trajectory,
                primary_segment=primary_name,
            ),
        )

    def _reward_output(self) -> Any:
        view = self._reward_view()
        if view.tensor_refs:
            if len(view.tensor_refs) != 1:
                raise RuntimeError(
                    f"RewardView {view.name!r} must expose exactly one tensor_ref "
                    "for collector reward scoring",
                )
            return self._tensor_value_from_ref(view.tensor_refs[0])

        output_ref = view.metadata.get("output_ref")
        if output_ref == "GenerationOutput.output":
            return extract_reward_artifact(self.output)
        raise RuntimeError(
            f"RewardView {view.name!r} has no tensor_refs and no supported output_ref",
        )

    def _reward_view(self) -> RewardView:
        reward_views = self.trajectory.reward_views
        requested = self.context.reward_view_name
        if requested:
            view = reward_views.get(requested)
            if view is None:
                raise RuntimeError(
                    f"TrajectoryBatch {self.trajectory.request_id!r} has no "
                    f"reward view {requested!r}; available={sorted(reward_views)}",
                )
            return view
        if len(reward_views) == 1:
            return next(iter(reward_views.values()))
        if not reward_views:
            raise RuntimeError(
                f"TrajectoryBatch {self.trajectory.request_id!r} has no reward views",
            )
        raise RuntimeError(
            f"TrajectoryBatch {self.trajectory.request_id!r} has multiple reward "
            "views; set RolloutBatchBuildContext.reward_view_name",
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
        return [
            segment
            for segment in self.trajectory.segments.values()
            if segment.trainable
        ]

    def _primary_trainable_segment(
        self,
        *,
        preferred: str | None = None,
    ) -> TrajectorySegment:
        if preferred is not None:
            segment = self.trajectory.segments.get(preferred)
            if segment is not None and segment.trainable:
                return segment
        for segment in self.trajectory.segments.values():
            if segment.trainable:
                return segment
        raise RuntimeError("TrajectoryBatch has no trainable segment")

    def _role_tensor(
        self,
        segment: TrajectorySegment,
        role: str,
    ) -> TrajectoryTensor:
        matches = [
            tensor
            for tensor in segment.tensors.values()
            if tensor.role == role
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"segment {segment.name!r} requires exactly one role {role!r}, "
                f"found {len(matches)}",
            )
        return matches[0]

    def _named_tensor(
        self,
        segment: TrajectorySegment,
        name: str,
    ) -> TrajectoryTensor:
        try:
            return segment.tensors[name]
        except KeyError as exc:
            raise RuntimeError(
                f"segment {segment.name!r} is missing tensor {name!r}",
            ) from exc

    def _optional_named_tensor(
        self,
        segment: TrajectorySegment,
        name: str,
    ) -> Any | None:
        tensor = segment.tensors.get(name)
        return None if tensor is None else tensor.value

    def _group_ids(self, *, device: Any) -> torch.Tensor:
        group_ids = self.trajectory.group_ids
        if isinstance(group_ids, torch.Tensor):
            return group_ids.to(device=device, dtype=torch.long)
        return torch.tensor(group_ids, dtype=torch.long, device=device)

    def _primary_segment_name(self) -> str | None:
        value = self.trajectory.context.get("primary_segment")
        return value if isinstance(value, str) else None

    def _is_multisegment_categorical(
        self,
        trainable: list[TrajectorySegment],
    ) -> bool:
        if (
            self.trajectory.family == "janus_pro_r1"
            or self.trajectory.task == "ar_t2i_r1"
        ):
            return True
        return len(trainable) > 1 and all(
            segment.distribution == "categorical"
            for segment in trainable
        )

    def _decoded_tensor(self, name: str) -> Any | None:
        decoded = self.trajectory.segments.get("decoded")
        if decoded is None:
            return None
        tensor = decoded.tensors.get(name)
        return None if tensor is None else tensor.value

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
