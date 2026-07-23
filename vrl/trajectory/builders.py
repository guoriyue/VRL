"""Builders that project family outputs into the trajectory contract."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.trajectory.types import (
    ReplayInput,
    TrajectoryAxis,
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.trajectory.validation import (
    TrajectoryValidator,
    tensor_ref,
)
from vrl.trajectory.views import RewardView


class _TrajectoryBatchBuilder:
    """Project one generation request and its sample rows into trajectory batches."""

    _DROP = object()

    def __init__(
        self,
        *,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
    ) -> None:
        self.request = request
        self.sample_rows = sample_rows

    def build_diffusion_trajectory(
        self,
        *,
        observations: Any,
        actions: Any,
        old_log_prob: Any,
        timesteps: Any,
        kl: Any,
        replay_tensors: dict[str, Any],
        context: dict[str, Any],
    ) -> TrajectoryBatch:
        """Build a denoise-step trajectory from shared diffusion rollout tensors."""

        batch_size = len(self.sample_rows)
        timestep_count = int(old_log_prob.shape[1])
        device = getattr(old_log_prob, "device", None)
        tensors: dict[str, TrajectoryTensor] = {
            "observations": TrajectoryTensor(
                "observations",
                observations,
                ("sample", "denoise"),
                "observation",
            ),
            "actions": TrajectoryTensor(
                "actions",
                actions,
                ("sample", "denoise"),
                "action",
            ),
            "old_log_prob": TrajectoryTensor(
                "old_log_prob",
                old_log_prob,
                ("sample", "denoise"),
                "old_log_prob",
            ),
            "mask": TrajectoryTensor(
                "mask",
                torch.ones_like(old_log_prob),
                ("sample", "denoise"),
                "mask",
            ),
            "timesteps": TrajectoryTensor(
                "timesteps",
                timesteps,
                ("sample", "denoise"),
                "replay_input",
            ),
            "kl": TrajectoryTensor(
                "kl",
                kl,
                ("sample", "denoise"),
                "replay_input",
            ),
        }
        replay_tensor_names: list[str] = []
        for name, value in replay_tensors.items():
            if name in tensors:
                continue
            if not self._sample_aligned(value):
                continue
            tensors[name] = TrajectoryTensor(name, value, ("sample",), "replay_input")
            replay_tensor_names.append(name)
        replay_tensor_refs = (
            tensor_ref("denoise", "observations"),
            tensor_ref("denoise", "actions"),
            tensor_ref("denoise", "timesteps"),
            *(tensor_ref("denoise", name) for name in replay_tensor_names),
        )

        reward_modality = self._reward_modality_for_task()
        trajectory = TrajectoryBatch(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=list(self.sample_rows),
            group_ids=self._prompt_group_ids(device=device),
            axes={
                "sample": TrajectoryAxis("sample", "sample", batch_size),
                "denoise": TrajectoryAxis("denoise", "denoise_step", timestep_count),
            },
            segments={
                "denoise": TrajectorySegment(
                    name="denoise",
                    modality="latent",
                    trainable=True,
                    distribution="flow_matching",
                    tensors=tensors,
                    reward_view=reward_modality,
                    replay_inputs={
                        "logprob": ReplayInput(
                            name="logprob",
                            tensor_refs=replay_tensor_refs,
                        ),
                    },
                )
            },
            reward_views={
                reward_modality: RewardView(
                    name=reward_modality,
                    value_range="unit",
                    metadata={"output_ref": "GenerationOutput.output"},
                )
            },
            metrics=TrajectoryMetrics(
                num_samples=batch_size,
                axis_lengths={"sample": batch_size, "denoise": timestep_count},
                values={"num_steps": timestep_count},
            ),
            context=self._serializable_context(context),
        )
        return TrajectoryValidator(trajectory).validate_batch()

    def build_chunk_autoregressive_denoise_trajectory(
        self,
        *,
        observations: Any,
        actions: Any,
        old_log_prob: Any,
        mask: Any,
        timesteps: Any,
        finalized_chunk_latents: Any,
        replay_tensors: dict[str, Any],
        context: dict[str, Any],
        kl: Any | None = None,
    ) -> TrajectoryBatch:
        """Build a trainable chunk-autoregressive denoise trajectory.

        Unlike a full-sequence denoise trace, the policy decisions here have two
        independent logical axes: which temporal chunk is being produced and which
        stochastic transition inside that chunk is replayed.  Every trainable fact
        therefore starts with ``[sample, temporal_chunk, denoise_transition]``.
        Terminal chunk latents are kept separately at ``[sample, temporal_chunk]``
        so replay can reconstruct the cache/conditioning boundary between chunks.
        """

        batch_size = len(self.sample_rows)
        chunk_count, transition_count = self._chunk_denoise_shape(old_log_prob)
        for name, value in (
            ("observations", observations),
            ("actions", actions),
            ("mask", mask),
            ("timesteps", timesteps),
        ):
            self._require_shape_prefix(
                name,
                value,
                (batch_size, chunk_count, transition_count),
            )
        self._require_shape_prefix(
            "finalized_chunk_latents",
            finalized_chunk_latents,
            (batch_size, chunk_count),
        )
        if kl is None:
            kl = torch.zeros_like(old_log_prob)
        else:
            self._require_shape_prefix(
                "kl",
                kl,
                (batch_size, chunk_count, transition_count),
            )

        transition_axes = ("sample", "temporal_chunk", "denoise_transition")
        tensors: dict[str, TrajectoryTensor] = {
            "observations": TrajectoryTensor(
                "observations",
                observations,
                transition_axes,
                "observation",
            ),
            "actions": TrajectoryTensor(
                "actions",
                actions,
                transition_axes,
                "action",
            ),
            "old_log_prob": TrajectoryTensor(
                "old_log_prob",
                old_log_prob,
                transition_axes,
                "old_log_prob",
            ),
            "mask": TrajectoryTensor(
                "mask",
                mask,
                transition_axes,
                "mask",
            ),
            "timesteps": TrajectoryTensor(
                "timesteps",
                timesteps,
                transition_axes,
                "replay_input",
            ),
            "kl": TrajectoryTensor(
                "kl",
                kl,
                transition_axes,
                "replay_input",
            ),
            "finalized_chunk_latents": TrajectoryTensor(
                "finalized_chunk_latents",
                finalized_chunk_latents,
                ("sample", "temporal_chunk"),
                "replay_input",
            ),
        }
        replay_tensor_names: list[str] = []
        for name, value in replay_tensors.items():
            if name in tensors:
                continue
            axes = self._chunk_replay_axes(
                value,
                chunk_count=chunk_count,
                transition_count=transition_count,
            )
            if axes is None:
                continue
            tensors[name] = TrajectoryTensor(name, value, axes, "replay_input")
            replay_tensor_names.append(name)

        reward_modality = self._reward_modality_for_task()
        replay_tensor_refs = (
            tensor_ref("denoise", "observations"),
            tensor_ref("denoise", "actions"),
            tensor_ref("denoise", "timesteps"),
            tensor_ref("denoise", "finalized_chunk_latents"),
            *(tensor_ref("denoise", name) for name in replay_tensor_names),
        )
        device = getattr(old_log_prob, "device", None)
        trajectory = TrajectoryBatch(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=list(self.sample_rows),
            group_ids=self._prompt_group_ids(device=device),
            axes={
                "sample": TrajectoryAxis("sample", "sample", batch_size),
                "temporal_chunk": TrajectoryAxis(
                    "temporal_chunk",
                    "temporal_chunk",
                    chunk_count,
                ),
                "denoise_transition": TrajectoryAxis(
                    "denoise_transition",
                    "denoise_transition",
                    transition_count,
                ),
            },
            segments={
                "denoise": TrajectorySegment(
                    name="denoise",
                    modality="latent",
                    trainable=True,
                    distribution="gaussian",
                    tensors=tensors,
                    reward_view=reward_modality,
                    replay_inputs={
                        "logprob": ReplayInput(
                            name="logprob",
                            tensor_refs=replay_tensor_refs,
                        ),
                    },
                    metadata={
                        "temporal_chunk_axis": "temporal_chunk",
                        "transition_axis": "denoise_transition",
                    },
                )
            },
            reward_views={
                reward_modality: RewardView(
                    name=reward_modality,
                    value_range="unit",
                    metadata={"output_ref": "GenerationOutput.output"},
                )
            },
            metrics=TrajectoryMetrics(
                num_samples=batch_size,
                axis_lengths={
                    "sample": batch_size,
                    "temporal_chunk": chunk_count,
                    "denoise_transition": transition_count,
                },
                values={
                    "num_temporal_chunks": chunk_count,
                    "num_denoise_transitions": transition_count,
                },
            ),
            context={
                **self._serializable_context(context),
                "trajectory_mode": "trainable_chunk_denoise",
            },
        )
        return TrajectoryValidator(trajectory).validate_batch()

    def build_chunk_autoregressive_generation_trajectory(
        self,
        *,
        output: Any,
        temporal_chunk_count: int,
        context: dict[str, Any],
    ) -> TrajectoryBatch:
        """Record chunk-autoregressive generation without inventing RL actions.

        Some upstream runtimes expose the generated artifact and temporal chunk
        count but not the stochastic transition facts needed for exact replay.  In
        that case the artifact remains useful for reward/evaluation, while the
        segment is explicitly non-trainable and contains no fake log-probability.
        """

        batch_size = len(self.sample_rows)
        if temporal_chunk_count < 1:
            raise ValueError("temporal_chunk_count must be >= 1")
        output_rows = self._leading_length(output)
        if output_rows is not None and output_rows != batch_size:
            raise ValueError(
                f"output has {output_rows} rows, expected {batch_size}",
            )
        reward_modality = self._reward_modality_for_task()
        device = getattr(output, "device", None)
        trajectory = TrajectoryBatch(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=list(self.sample_rows),
            group_ids=self._prompt_group_ids(device=device),
            axes={
                "sample": TrajectoryAxis("sample", "sample", batch_size),
                "temporal_chunk": TrajectoryAxis(
                    "temporal_chunk",
                    "temporal_chunk",
                    temporal_chunk_count,
                ),
            },
            segments={
                "generated_chunks": TrajectorySegment(
                    name="generated_chunks",
                    modality=reward_modality,
                    trainable=False,
                    distribution="deterministic",
                    tensors={
                        "output": TrajectoryTensor(
                            "output",
                            output,
                            ("sample",),
                            "replay_input",
                        ),
                    },
                    reward_view=reward_modality,
                    metadata={"temporal_chunk_axis": "temporal_chunk"},
                )
            },
            reward_views={
                reward_modality: RewardView(
                    name=reward_modality,
                    tensor_refs=(tensor_ref("generated_chunks", "output"),),
                    value_range="unit",
                )
            },
            metrics=TrajectoryMetrics(
                num_samples=batch_size,
                axis_lengths={
                    "sample": batch_size,
                    "temporal_chunk": temporal_chunk_count,
                },
                values={"num_temporal_chunks": temporal_chunk_count},
            ),
            context={
                **self._serializable_context(context),
                "trajectory_mode": "generation_only",
            },
        )
        return TrajectoryValidator(trajectory).validate_batch()

    def build_ar_discrete_trajectory(
        self,
        *,
        token_ids: Any,
        token_log_probs: Any,
        token_mask: Any,
        prompt_input_ids: Any,
        prompt_attention_mask: Any,
        uncond_input_ids: Any,
        uncond_attention_mask: Any,
        context: dict[str, Any],
    ) -> TrajectoryBatch:
        """Build a discrete image-token trajectory from Janus-Pro rollout tensors."""

        batch_size = len(self.sample_rows)
        token_count = int(token_ids.shape[1])
        device = getattr(token_ids, "device", None)
        trajectory = TrajectoryBatch(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=list(self.sample_rows),
            group_ids=self._prompt_group_ids(device=device),
            axes={
                "sample": TrajectoryAxis("sample", "sample", batch_size),
                "token": TrajectoryAxis("token", "discrete_token", token_count),
            },
            segments={
                "image_tokens": TrajectorySegment(
                    name="image_tokens",
                    modality="image",
                    trainable=True,
                    distribution="categorical",
                    tensors={
                        "token_ids": TrajectoryTensor(
                            "token_ids",
                            token_ids,
                            ("sample", "token"),
                            "action",
                        ),
                        "old_log_prob": TrajectoryTensor(
                            "old_log_prob",
                            token_log_probs,
                            ("sample", "token"),
                            "old_log_prob",
                        ),
                        "token_mask": TrajectoryTensor(
                            "token_mask",
                            token_mask,
                            ("sample", "token"),
                            "mask",
                        ),
                        "prompt_input_ids": TrajectoryTensor(
                            "prompt_input_ids",
                            prompt_input_ids,
                            ("sample",),
                            "replay_input",
                        ),
                        "prompt_attention_mask": TrajectoryTensor(
                            "prompt_attention_mask",
                            prompt_attention_mask,
                            ("sample",),
                            "replay_input",
                        ),
                        "uncond_input_ids": TrajectoryTensor(
                            "uncond_input_ids",
                            uncond_input_ids,
                            ("sample",),
                            "replay_input",
                        ),
                        "uncond_attention_mask": TrajectoryTensor(
                            "uncond_attention_mask",
                            uncond_attention_mask,
                            ("sample",),
                            "replay_input",
                        ),
                    },
                    reward_view="image",
                    replay_inputs={
                        "logprob": ReplayInput(
                            name="logprob",
                            tensor_refs=(
                                tensor_ref("image_tokens", "token_ids"),
                                tensor_ref("image_tokens", "prompt_input_ids"),
                                tensor_ref("image_tokens", "prompt_attention_mask"),
                                tensor_ref("image_tokens", "uncond_input_ids"),
                                tensor_ref("image_tokens", "uncond_attention_mask"),
                            ),
                        ),
                    },
                )
            },
            reward_views={
                "image": RewardView(
                    name="image",
                    value_range="tanh",
                    metadata={"output_ref": "GenerationOutput.output"},
                )
            },
            metrics=TrajectoryMetrics(
                num_samples=batch_size,
                axis_lengths={"sample": batch_size, "token": token_count},
                values={"num_tokens": token_count},
            ),
            context=self._serializable_context(context),
        )
        return TrajectoryValidator(trajectory).validate_batch()

    def build_ar_continuous_trajectory(
        self,
        *,
        tokens: Any,
        saved_noise: Any,
        token_log_probs: Any,
        token_mask: Any,
        prompt_input_ids: Any,
        prompt_attention_mask: Any,
        uncond_input_ids: Any,
        uncond_attention_mask: Any,
        context: dict[str, Any],
    ) -> TrajectoryBatch:
        """Build a continuous image-token trajectory from NextStep rollout tensors."""

        batch_size = len(self.sample_rows)
        token_count = int(token_log_probs.shape[1])
        device = getattr(token_log_probs, "device", None)
        segments: dict[str, TrajectorySegment] = {
            "image_tokens": TrajectorySegment(
                name="image_tokens",
                modality="image",
                trainable=True,
                distribution="gaussian",
                tensors={
                    "tokens": TrajectoryTensor(
                        "tokens",
                        tokens,
                        ("sample", "token"),
                        "action",
                    ),
                    "old_log_prob": TrajectoryTensor(
                        "old_log_prob",
                        token_log_probs,
                        ("sample", "token"),
                        "old_log_prob",
                    ),
                    "token_mask": TrajectoryTensor(
                        "token_mask",
                        token_mask,
                        ("sample", "token"),
                        "mask",
                    ),
                    "saved_noise": TrajectoryTensor(
                        "saved_noise",
                        saved_noise,
                        ("sample", "token"),
                        "replay_input",
                    ),
                    "prompt_input_ids": TrajectoryTensor(
                        "prompt_input_ids",
                        prompt_input_ids,
                        ("sample",),
                        "replay_input",
                    ),
                    "prompt_attention_mask": TrajectoryTensor(
                        "prompt_attention_mask",
                        prompt_attention_mask,
                        ("sample",),
                        "replay_input",
                    ),
                    "uncond_input_ids": TrajectoryTensor(
                        "uncond_input_ids",
                        uncond_input_ids,
                        ("sample",),
                        "replay_input",
                    ),
                    "uncond_attention_mask": TrajectoryTensor(
                        "uncond_attention_mask",
                        uncond_attention_mask,
                        ("sample",),
                        "replay_input",
                    ),
                },
                reward_view="image",
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=(
                            tensor_ref("image_tokens", "tokens"),
                            tensor_ref("image_tokens", "saved_noise"),
                            tensor_ref("image_tokens", "prompt_input_ids"),
                            tensor_ref("image_tokens", "prompt_attention_mask"),
                            tensor_ref("image_tokens", "uncond_input_ids"),
                            tensor_ref("image_tokens", "uncond_attention_mask"),
                        ),
                    ),
                },
            )
        }
        trajectory = TrajectoryBatch(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=list(self.sample_rows),
            group_ids=self._prompt_group_ids(device=device),
            axes={
                "sample": TrajectoryAxis("sample", "sample", batch_size),
                "token": TrajectoryAxis("token", "continuous_token", token_count),
            },
            segments=segments,
            reward_views={
                "image": RewardView(
                    name="image",
                    value_range="tanh",
                    metadata={"output_ref": "GenerationOutput.output"},
                )
            },
            metrics=TrajectoryMetrics(
                num_samples=batch_size,
                axis_lengths={"sample": batch_size, "token": token_count},
                values={"num_tokens": token_count},
            ),
            context=self._serializable_context(context),
        )
        return TrajectoryValidator(trajectory).validate_batch()

    def build_ar_multisegment_trajectory(
        self,
        *,
        segments: dict[str, dict[str, Any]],
        primary_segment: str,
        context: dict[str, Any],
    ) -> TrajectoryBatch:
        """Build a multi-segment categorical trajectory without flattening segments."""

        if not segments:
            raise ValueError("segments must be non-empty")
        batch_size = len(self.sample_rows)
        first_segment = next(iter(segments.values()))
        device = getattr(first_segment.get("token_ids"), "device", None)

        axes = {"sample": TrajectoryAxis("sample", "sample", batch_size)}
        trajectory_segments: dict[str, TrajectorySegment] = {}
        for name, payload in segments.items():
            token_ids = payload["token_ids"]
            old_log_prob = payload.get("token_log_probs")
            if old_log_prob is None:
                old_log_prob = payload.get("old_log_probs")
            if old_log_prob is None:
                old_log_prob = torch.zeros_like(token_ids, dtype=torch.float32)
            token_mask = payload.get("token_mask")
            if token_mask is None:
                token_mask = torch.ones_like(old_log_prob, dtype=torch.float32)

            axis_name = f"{name}_token"
            token_count = int(token_ids.shape[1])
            axes[axis_name] = TrajectoryAxis(axis_name, "discrete_token", token_count)
            modality = (
                "image" if bool(payload.get("visual", not name.endswith("_text"))) else "text"
            )
            trainable = self._segment_trainable(
                self.request.sampling.get("train_segments"), name, payload
            )
            trajectory_segments[name] = TrajectorySegment(
                name=name,
                modality=modality,
                trainable=trainable,
                distribution="categorical",
                tensors={
                    "token_ids": TrajectoryTensor(
                        "token_ids",
                        token_ids,
                        ("sample", axis_name),
                        "action",
                    ),
                    "old_log_prob": TrajectoryTensor(
                        "old_log_prob",
                        old_log_prob,
                        ("sample", axis_name),
                        "old_log_prob",
                    ),
                    "token_mask": TrajectoryTensor(
                        "token_mask",
                        token_mask,
                        ("sample", axis_name),
                        "mask",
                    ),
                    "prompt_embeds": TrajectoryTensor(
                        "prompt_embeds",
                        payload["prompt_embeds"],
                        ("sample",),
                        "replay_input",
                    ),
                    "attention_mask": TrajectoryTensor(
                        "attention_mask",
                        payload["attention_mask"],
                        ("sample",),
                        "replay_input",
                    ),
                    "prompt_attention_mask": TrajectoryTensor(
                        "prompt_attention_mask",
                        payload.get("prompt_attention_mask", payload["attention_mask"]),
                        ("sample",),
                        "replay_input",
                    ),
                },
                reward_view="image" if name == primary_segment else None,
                advantage_scope="segment",
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=(
                            tensor_ref(name, "token_ids"),
                            tensor_ref(name, "prompt_embeds"),
                            tensor_ref(name, "attention_mask"),
                        ),
                    ),
                },
                metadata={
                    "visual": bool(payload.get("visual", modality == "image")),
                    "cfg": bool(payload.get("cfg", False)),
                    "train": trainable,
                },
            )

        axis_lengths = {
            name: axis.length for name, axis in axes.items() if axis.length is not None
        }
        trajectory = TrajectoryBatch(
            request_id=self.request.request_id,
            family=self.request.family,
            task=self.request.task,
            sample_rows=list(self.sample_rows),
            group_ids=self._prompt_group_ids(device=device),
            axes=axes,
            segments=trajectory_segments,
            reward_views={
                "image": RewardView(
                    name="image",
                    value_range="tanh",
                    metadata={"output_ref": "GenerationOutput.output"},
                )
            },
            metrics=TrajectoryMetrics(
                num_samples=batch_size,
                axis_lengths=axis_lengths,
                values={"num_segments": len(segments)},
            ),
            context={
                **self._serializable_context(context),
                "primary_segment": primary_segment,
                "segment_names": tuple(segments),
            },
        )
        return TrajectoryValidator(trajectory).validate_batch()

    def _prompt_group_ids(
        self,
        *,
        device: Any,
    ) -> Any:
        return torch.tensor(
            [row.prompt_index for row in self.sample_rows],
            dtype=torch.long,
            device=device,
        )

    def _chunk_denoise_shape(self, value: Any) -> tuple[int, int]:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 3:
            raise ValueError(
                "old_log_prob must have leading [sample, temporal_chunk, "
                "denoise_transition] dimensions",
            )
        batch_size = len(self.sample_rows)
        if int(shape[0]) != batch_size:
            raise ValueError(
                f"old_log_prob has {shape[0]} rows, expected {batch_size}",
            )
        chunk_count = int(shape[1])
        transition_count = int(shape[2])
        if chunk_count < 1 or transition_count < 1:
            raise ValueError(
                "old_log_prob temporal_chunk and denoise_transition dimensions must be >= 1",
            )
        return chunk_count, transition_count

    @staticmethod
    def _require_shape_prefix(name: str, value: Any, expected: tuple[int, ...]) -> None:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < len(expected):
            raise ValueError(f"{name} must have leading dimensions {expected}")
        actual = tuple(int(length) for length in shape[: len(expected)])
        if actual != expected:
            raise ValueError(f"{name} has leading dimensions {actual}, expected {expected}")

    def _chunk_replay_axes(
        self,
        value: Any,
        *,
        chunk_count: int,
        transition_count: int,
    ) -> tuple[str, ...] | None:
        batch_size = len(self.sample_rows)
        shape = getattr(value, "shape", None)
        if shape is not None:
            if len(shape) >= 3 and tuple(int(length) for length in shape[:3]) == (
                batch_size,
                chunk_count,
                transition_count,
            ):
                return ("sample", "temporal_chunk", "denoise_transition")
            if len(shape) >= 2 and tuple(int(length) for length in shape[:2]) == (
                batch_size,
                chunk_count,
            ):
                return ("sample", "temporal_chunk")
            if len(shape) >= 1 and int(shape[0]) == batch_size:
                return ("sample",)
            return None
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return ("sample",)
        return None

    @staticmethod
    def _leading_length(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    def _sample_aligned(self, value: Any) -> bool:
        batch_size = len(self.sample_rows)
        shape = getattr(value, "shape", None)
        if shape is not None:
            return len(shape) >= 1 and int(shape[0]) == batch_size
        return isinstance(value, (list, tuple)) and len(value) == batch_size

    def _serializable_context(self, value: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, inner in value.items():
            serializable = self._serializable_value(inner)
            if serializable is not self._DROP:
                out[key] = serializable
        return out

    def _serializable_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return value
        if isinstance(value, tuple):
            values = [self._serializable_value(inner) for inner in value]
            return tuple(inner for inner in values if inner is not self._DROP)
        if isinstance(value, list):
            values = [self._serializable_value(inner) for inner in value]
            return [inner for inner in values if inner is not self._DROP]
        if isinstance(value, dict):
            return self._serializable_context(value)
        return self._DROP

    def _reward_modality_for_task(self) -> str:
        # i2v is image-conditioned but still emits video frames, so it scores as video.
        if self.request.task in {"t2v", "i2v", "v2w", "t2w"}:
            return "video"
        return "image"

    @staticmethod
    def _segment_trainable(value: Any, name: str, payload: dict[str, Any]) -> bool:
        if "train" in payload:
            return bool(payload["train"])
        if "enabled" in payload:
            return bool(payload["enabled"])
        if value is None:
            return bool(payload.get("visual", True))
        if isinstance(value, dict):
            return bool(value.get(name, False))
        if isinstance(value, str):
            return name == value
        if isinstance(value, (list, tuple, set, frozenset)):
            return name in value
        return bool(value)


def build_diffusion_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    observations: Any,
    actions: Any,
    old_log_prob: Any,
    timesteps: Any,
    kl: Any,
    replay_tensors: dict[str, Any],
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a denoise-step trajectory from shared diffusion rollout tensors."""

    return _TrajectoryBatchBuilder(
        request=request,
        sample_rows=sample_rows,
    ).build_diffusion_trajectory(
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=timesteps,
        kl=kl,
        replay_tensors=replay_tensors,
        context=context,
    )


def build_chunk_autoregressive_denoise_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    observations: Any,
    actions: Any,
    old_log_prob: Any,
    mask: Any,
    timesteps: Any,
    finalized_chunk_latents: Any,
    replay_tensors: dict[str, Any],
    context: dict[str, Any],
    kl: Any | None = None,
) -> TrajectoryBatch:
    """Build a trainable chunk-autoregressive denoise trajectory.

    Unlike a full-sequence denoise trace, the policy decisions here have two
    independent logical axes: which temporal chunk is being produced and which
    stochastic transition inside that chunk is replayed.  Every trainable fact
    therefore starts with ``[sample, temporal_chunk, denoise_transition]``.
    Terminal chunk latents are kept separately at ``[sample, temporal_chunk]``
    so replay can reconstruct the cache/conditioning boundary between chunks.
    """

    return _TrajectoryBatchBuilder(
        request=request,
        sample_rows=sample_rows,
    ).build_chunk_autoregressive_denoise_trajectory(
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        mask=mask,
        timesteps=timesteps,
        finalized_chunk_latents=finalized_chunk_latents,
        replay_tensors=replay_tensors,
        context=context,
        kl=kl,
    )


def build_chunk_autoregressive_generation_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    output: Any,
    temporal_chunk_count: int,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Record chunk-autoregressive generation without inventing RL actions.

    Some upstream runtimes expose the generated artifact and temporal chunk
    count but not the stochastic transition facts needed for exact replay.  In
    that case the artifact remains useful for reward/evaluation, while the
    segment is explicitly non-trainable and contains no fake log-probability.
    """

    return _TrajectoryBatchBuilder(
        request=request,
        sample_rows=sample_rows,
    ).build_chunk_autoregressive_generation_trajectory(
        output=output,
        temporal_chunk_count=temporal_chunk_count,
        context=context,
    )


def build_ar_discrete_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    token_ids: Any,
    token_log_probs: Any,
    token_mask: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    uncond_input_ids: Any,
    uncond_attention_mask: Any,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a discrete image-token trajectory from Janus-Pro rollout tensors."""

    return _TrajectoryBatchBuilder(
        request=request,
        sample_rows=sample_rows,
    ).build_ar_discrete_trajectory(
        token_ids=token_ids,
        token_log_probs=token_log_probs,
        token_mask=token_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        uncond_input_ids=uncond_input_ids,
        uncond_attention_mask=uncond_attention_mask,
        context=context,
    )


def build_ar_continuous_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    tokens: Any,
    saved_noise: Any,
    token_log_probs: Any,
    token_mask: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    uncond_input_ids: Any,
    uncond_attention_mask: Any,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a continuous image-token trajectory from NextStep rollout tensors."""

    return _TrajectoryBatchBuilder(
        request=request,
        sample_rows=sample_rows,
    ).build_ar_continuous_trajectory(
        tokens=tokens,
        saved_noise=saved_noise,
        token_log_probs=token_log_probs,
        token_mask=token_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        uncond_input_ids=uncond_input_ids,
        uncond_attention_mask=uncond_attention_mask,
        context=context,
    )


def build_ar_multisegment_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    segments: dict[str, dict[str, Any]],
    primary_segment: str,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a multi-segment categorical trajectory without flattening segments."""

    return _TrajectoryBatchBuilder(
        request=request,
        sample_rows=sample_rows,
    ).build_ar_multisegment_trajectory(
        segments=segments,
        primary_segment=primary_segment,
        context=context,
    )


__all__ = [
    "build_ar_continuous_trajectory",
    "build_ar_discrete_trajectory",
    "build_ar_multisegment_trajectory",
    "build_chunk_autoregressive_denoise_trajectory",
    "build_chunk_autoregressive_generation_trajectory",
    "build_diffusion_trajectory",
]
