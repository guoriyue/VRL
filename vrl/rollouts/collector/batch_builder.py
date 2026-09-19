"""Build trainer rollout batches from trajectory-backed generation outputs.

Owns the trajectory-format knowledge the collector itself must not carry:
which segment is trainable, which reward view scores, and how a denoise
trajectory packs into the engine-neutral ``RolloutBatch``. One builder per ``GenerationOutput`` produces both the ``RewardSample`` inputs for
``RewardRuntime.score`` and, once scores return, the trainer batch — keeping
``collector.core`` purely about phase ordering and GPU handoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from vrl.generation import GenerationOutput
from vrl.rewards import RewardSample
from vrl.rewards.types import REWARD_GROUP_ID_METADATA_KEY
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.storage import TrajectoryStoragePolicy
from vrl.trajectory.types import TrajectorySegment
from vrl.utils.media_reference import MediaReference


@dataclass(slots=True)
class RolloutBatchBuildContext:
    """Non-engine metadata needed while building a trainer RolloutBatch."""

    metadata: dict[str, Any]
    device: Any | None = None
    trajectory_storage_policy: TrajectoryStoragePolicy = field(
        default_factory=TrajectoryStoragePolicy,
    )


class TrajectoryRolloutBatchBuilder:
    """Convert one trajectory-backed GenerationOutput into reward and trainer inputs."""

    def __init__(
        self,
        output: GenerationOutput,
        context: RolloutBatchBuildContext,
    ) -> None:
        self.output = output
        self.context = context
        self.trajectory = context.trajectory_storage_policy.apply_to_trajectory_(output.trajectory)
        self.output.trajectory = self.trajectory

    def reward_samples(self) -> tuple[RewardSample, ...]:
        """Build reward-owned samples from this generation output."""

        reward_outputs = self.reward_outputs()
        batch_size = self._batch_size(reward_outputs)
        if len(self.output.sample_rows) != batch_size:
            raise ValueError(
                "reward sample-row/output batch mismatch: "
                f"sample_rows={len(self.output.sample_rows)}, outputs={batch_size}",
            )
        samples: list[RewardSample] = []
        if REWARD_GROUP_ID_METADATA_KEY in self.context.metadata:
            raise ValueError(
                f"reward metadata key {REWARD_GROUP_ID_METADATA_KEY!r} is collector-owned",
            )
        for index, row in enumerate(self.output.sample_rows):
            metadata = dict(self.context.metadata)
            metadata[REWARD_GROUP_ID_METADATA_KEY] = (
                f"{self.output.request_id}:prompt:{row.prompt_index}"
            )
            samples.append(
                RewardSample(
                    prompt=row.prompt,
                    output=reward_outputs[index],
                    sample_id=row.sample_id,
                    metadata=metadata,
                ),
            )
        return tuple(samples)

    def reward_outputs(self) -> Any:
        """Return the selected artifact normalized to [0, 1] per the reward view's range."""

        reward_views = self.trajectory.reward_views
        if not reward_views:
            raise RuntimeError(
                f"TrajectoryBatch {self.trajectory.request_id!r} has no reward views",
            )
        if len(reward_views) != 1:
            raise RuntimeError(
                f"TrajectoryBatch {self.trajectory.request_id!r} has multiple reward "
                "views; every generated trajectory must declare exactly one scoring view",
            )
        view = next(iter(reward_views.values()))

        if view.tensor_refs:
            if len(view.tensor_refs) != 1:
                raise RuntimeError(
                    f"RewardInputSpec {view.name!r} must expose exactly one tensor_ref "
                    "for collector reward scoring",
                )
            ref = view.tensor_refs[0]
            segment_name, tensor_name = ref.split(".", 1)
            try:
                reward_output = self.trajectory.segments[segment_name].tensors[tensor_name].value
            except KeyError as exc:
                raise RuntimeError(
                    f"RewardInputSpec references unknown trajectory tensor {ref!r}",
                ) from exc
        elif view.metadata.get("output_ref") == "GenerationOutput.output":
            reward_output = self.output.output
        else:
            raise RuntimeError(
                f"RewardInputSpec {view.name!r} has no tensor_refs and no supported output_ref",
            )

        if (
            isinstance(reward_output, list)
            and reward_output
            and all(isinstance(value, MediaReference) for value in reward_output)
        ):
            # Preserve range metadata without resolving bytes on the driver.
            return [replace(value, value_range=view.value_range) for value in reward_output]
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

        trainable = [segment for segment in self.trajectory.segments.values() if segment.trainable]
        if not trainable:
            raise RuntimeError(
                "generation-only trajectory cannot build a trainer RolloutBatch: "
                "no trainable policy segment or replay facts were recorded",
            )
        segment = self._primary_trainable_segment()
        if segment.distribution == "flow_matching" or (
            segment.distribution == "gaussian" and segment.modality == "latent"
        ):
            return self._pack_diffusion(segment, rewards_raw)
        raise NotImplementedError(
            "trajectory rollout collection does not support distribution="
            f"{segment.distribution!r}",
        )

    def _pack_diffusion(
        self,
        segment: TrajectorySegment,
        rewards_raw: torch.Tensor,
    ) -> RolloutBatch:
        observations = segment.role_tensor("observation").value
        device = observations.device
        rewards_adjusted = rewards_raw.to(device)

        rollout_context = dict(self.trajectory.context)
        if self.context.metadata:
            rollout_context["reward_metadata"] = dict(self.context.metadata)
        runtime_debug = self.output.runtime_debug
        if runtime_debug is not None:
            rollout_context["runtime_debug"] = runtime_debug

        return RolloutBatch(
            rewards=rewards_adjusted,
            group_ids=self._group_ids(device=device),
            extras={},
            context=rollout_context,
            trajectory=self.trajectory,
        )

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

    @staticmethod
    def _batch_size(value: Any) -> int:
        shape = getattr(value, "shape", None)
        if shape is not None:
            if len(shape) == 0:
                raise ValueError("reward outputs must have a batch dimension")
            return int(shape[0])
        try:
            return len(value)
        except TypeError as exc:
            raise TypeError("reward outputs must expose shape[0] or len()") from exc


__all__ = [
    "RolloutBatchBuildContext",
    "TrajectoryRolloutBatchBuilder",
]
