"""Shared executor scaffolding for full-sequence denoise generation families.

Worker-side half of the full-sequence binding: everything here runs in the
process that owns the model (encode -> prepare -> denoise -> decode per sample
batch), adapting the regime-independent denoise loop
(``vrl/generation/steps/denoise``) to ``execution``'s batch-executor contract.
The driver-side half lives in ``gather.py`` (model-free reassembly), with
``layout.py`` holding the request parsing both halves share.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from vrl.generation.bindings.full_sequence_denoise.layout import (
    DiffusionRequestLayout,
    DiffusionSamplingParams,
)
from vrl.generation.execution.executor_base import BatchExecutorBase
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import BatchCompletionCallback
from vrl.generation.protocols import GenerationBatchGatherer
from vrl.generation.steps.denoise.config import DenoiseLoopConfig
from vrl.generation.steps.denoise.loop import (
    DenoiseLoopResult,
    run_denoise_loop,
)
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    VideoGenerationRequest,
)
from vrl.trajectory.storage import (
    TrajectoryStoragePolicy,
    apply_value_storage_policy,
    trajectory_storage_policy_from_cfg,
    trajectory_tensor_bytes,
)
from vrl.utils.cuda_memory import (
    cuda_peak_allocated_bytes,
    cuda_peak_allocated_mb,
    reset_cuda_peak,
)
from vrl.utils.media import load_reference_image, to_uint8


@dataclass(slots=True)
class DiffusionBatchResult:
    """Output of one fused diffusion sample batch."""

    batch: GenerationSampleBatch
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    video: Any
    replay_tensors: dict[str, Any]
    context: dict[str, Any]
    # Display/provenance-only: emitted through per-batch runtime debug metrics.
    peak_memory_mb: float | None = None
    # Binding-local memory reading consumed and cleared at the worker boundary.
    # None off-CUDA.
    memory: dict[str, int] | None = None
    # Display/provenance-only: emitted through per-batch runtime debug metrics.
    stage_durations: dict[str, float] = field(default_factory=dict)
    # Display/provenance-only: emitted through per-batch runtime debug metrics.
    engine_counters: dict[str, Any] = field(default_factory=dict)


class ReferenceConditionedBatches:
    """Reference-image threading for per-batch encode/prepare.

    Cosmos Predict2 Video2World and Wan 2.1 I2V condition every batch on a
    reference image carried by its ``GenerationInput``. The two executors had
    copy-pasted these
    hooks; ``build_batch_encoded`` stays family-specific because the encoded
    payloads genuinely differ (Wan carries ``image_embeds``).
    """

    model: Any

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Encode text plus the active reference-image conditioning for one batch."""

        reference_image = self._reference_image_for_chunk(generation_request, batch)
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            video_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            reference_image=reference_image,
        )

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Thread the active reference image into family prepare_sampling."""

        del encoded, video_request, params
        return {
            "reference_image": self._reference_image_for_chunk(
                generation_request,
                batch,
            ),
        }

    def _reference_image_for_chunk(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> Any:
        ref = request.inputs[batch.prompt_index].reference_image
        if ref is None:
            raise ValueError(
                f"{request.family} requires reference_image for prompt index {batch.prompt_index}",
            )
        return load_reference_image(ref)


class DiffusionBatchExecutorBase(BatchExecutorBase):
    """Common GenerationRequest -> diffusion GenerationOutput execution path."""

    family: str
    task: str
    model: Any
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int | None = None
    sde_type: str = "flow_grpo"

    def __init__(
        self,
        model: Any,
        *,
        gatherer: GenerationBatchGatherer | None = None,
    ) -> None:
        super().__init__(gatherer=gatherer)
        self.model = model

    # -- protocol ------------------------------------------------------

    @property
    def layout(self) -> DiffusionRequestLayout:
        return DiffusionRequestLayout(
            default_num_frames=self.default_num_frames,
            default_fps=self.default_fps,
            default_max_sequence_length=self.default_max_sequence_length,
            sde_type=self.sde_type,
        )

    def parse_sampling_params(self, request: GenerationRequest) -> DiffusionSamplingParams:
        return self.layout.parse_sampling_params(request)

    def build_denoise_config(
        self,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> DenoiseLoopConfig:
        """Build the SDE denoise config for one sample batch."""

        layout = self.layout
        return DenoiseLoopConfig(
            sample_start=batch.sample_start,
            sample_count=batch.sample_count,
            seed=params.model_request.seed,
            sde=params.sde,
            sde_window=layout.select_sde_window(params),
            denoise_mode=params.denoise_mode,
            teacache=params.teacache,
        )

    def forward_plan_pipelined(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        plan: EnginePlan,
        *,
        completion_callback: BatchCompletionCallback | None = None,
    ) -> GenerationOutput:
        """In-process software-pipelined variant of forward_plan: batch N+1's
        produce (encode->prepare->denoise->decode, GPU compute on the default
        stream) overlaps batch N's teardown (the GPU->CPU result copy + host
        packing, on a copy stream), hiding the per-batch copy+CPU boundary behind
        the next batch's denoise. BIT-EXACT to forward_plan: same per-batch stage
        methods (via forward_batch), value-preserving side-stream copy,
        and the SAME order-preserving gather_batches — so the gathered output is
        identical; only the wall-clock changes.

        This is the executor-level entry for the single-GPU stage-overlap lever; the
        Ray worker calls it per-request (all of a request's batches on one worker)
        instead of dispatching one monolithic forward_batch per batch.
        """

        from vrl.generation.execution.pipeline import forward_batches_pipelined

        batches = forward_batches_pipelined(
            self,
            request,
            plan.sample_batches,
            completion_callback=completion_callback,
        )
        return self.gather_batches(request, sample_rows, batches)

    def forward_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> DiffusionBatchResult:
        """Run the canonical diffusion batch flow and prepare its wire payload."""

        return self._forward_chunk(
            request,
            batch,
            execute_steps=None,
            apply_wire_storage=True,
        )

    def forward_probe_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
        *,
        execute_steps: int,
    ) -> DiffusionBatchResult:
        """Run a truncated canonical batch for startup memory sizing."""

        if execute_steps < 1:
            raise ValueError("execute_steps must be >= 1")
        return self._forward_chunk(
            request,
            batch,
            execute_steps=execute_steps,
            apply_wire_storage=False,
        )

    def _forward_chunk(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
        *,
        execute_steps: int | None,
        apply_wire_storage: bool,
    ) -> DiffusionBatchResult:
        from vrl.utils.profiling import profile_range

        stage_durations: dict[str, float] = {}
        params = self.parse_sampling_params(request)
        video_request = params.model_request

        started = time.perf_counter()
        with profile_range("generation.prompt_encode"):
            encoded = self.encode_prompt_for_batch(
                generation_request=request,
                video_request=video_request,
                params=params,
                batch=batch,
            )
        stage_durations["encode"] = time.perf_counter() - started

        started = time.perf_counter()
        batch_encoded = self.build_batch_encoded(
            encoded=encoded,
            generation_request=request,
            video_request=video_request,
            params=params,
            batch=batch,
        )
        prepare_kwargs = self.build_prepare_kwargs(
            encoded=encoded,
            generation_request=request,
            video_request=video_request,
            params=params,
            batch=batch,
        )
        config = self.build_denoise_config(params, batch)
        if execute_steps is not None:
            config = replace(config, execute_steps=execute_steps)
        state = self.prepare_denoise_state(
            request=video_request,
            encoded=batch_encoded,
            config=config,
            prepare_kwargs=prepare_kwargs,
        )
        stage_durations["prepare_latent"] = time.perf_counter() - started

        started = time.perf_counter()
        denoise_result = self.run_denoise_steps(
            state=state,
            config=config,
        )
        stage_durations["denoise"] = time.perf_counter() - started

        started = time.perf_counter()
        batch_result = self.decode_denoise_result(
            batch=batch,
            config=config,
            denoise_result=denoise_result,
            stage_durations=stage_durations,
        )
        batch_result.stage_durations["decode"] = time.perf_counter() - started
        if apply_wire_storage:
            return self.apply_wire_storage_policy(request, batch_result)
        return batch_result

    def apply_wire_storage_policy(
        self,
        request: GenerationRequest,
        batch_result: DiffusionBatchResult,
    ) -> DiffusionBatchResult:
        """Apply rollout.trajectory_storage BEFORE tensors cross the wire.

        The same policy is re-applied driver-side when the trajectory batch is
        built (idempotent there); applying it here is what turns a dtype
        downcast into actual worker->driver transfer savings. The default
        preserve/preserve policy is a no-op, keeping the GRPO baseline
        bit-for-bit.
        """

        policy = trajectory_storage_policy_from_cfg(
            request.sampling.get("trajectory_storage"),
        )
        if policy == TrajectoryStoragePolicy():
            return batch_result
        batch_result.observations = apply_value_storage_policy(
            batch_result.observations,
            policy,
        )
        batch_result.actions = apply_value_storage_policy(batch_result.actions, policy)
        batch_result.log_probs = apply_value_storage_policy(batch_result.log_probs, policy)
        batch_result.timesteps = apply_value_storage_policy(batch_result.timesteps, policy)
        batch_result.kl = apply_value_storage_policy(batch_result.kl, policy)
        batch_result.replay_tensors = apply_value_storage_policy(
            batch_result.replay_tensors,
            policy,
        )
        return batch_result

    def prepare_denoise_state(
        self,
        *,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        config: DenoiseLoopConfig,
        prepare_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Prepare latent state for one diffusion sample batch."""

        from vrl.utils.profiling import profile_range

        model = self.model
        with profile_range("generation.prepare_sampling"):
            state = model.prepare_sampling(request, encoded, **(prepare_kwargs or {}))
        batch_rows = state.latents.shape[0]
        if int(batch_rows) != config.sample_count:
            raise ValueError(
                "Diffusion denoise batch produced "
                f"{batch_rows} rows, expected {config.sample_count}",
            )
        return state

    def run_denoise_steps(
        self,
        *,
        state: Any,
        config: DenoiseLoopConfig,
    ) -> DenoiseLoopResult:
        """Adapt the full-sequence denoise binding to the step-owned loop."""

        return run_denoise_loop(
            model=self.model,
            state=state,
            config=config,
        )

    def decode_denoise_result(
        self,
        *,
        batch: GenerationSampleBatch,
        config: DenoiseLoopConfig,
        denoise_result: DenoiseLoopResult,
        stage_durations: dict[str, float] | None = None,
    ) -> DiffusionBatchResult:
        """Decode the final latents and pack one diffusion batch result."""

        from vrl.utils.profiling import profile_range

        model = self.model
        state = denoise_result.state
        # Byte-admission shadow: the decode + pack spike is the second memory
        # phase, measured separately from the denoise plateau.
        reset_cuda_peak()
        with profile_range("generation.decode_latents"):
            video = model.decode_latents(state.latents)
        # Pack decoded video as uint8 before it crosses the worker->driver
        # wire: every downstream consumer (reward models, mp4 artifacts)
        # quantizes to uint8 anyway, so fp32 here is 4x wasted transfer
        # (~474MB/group). The driver reconstructs [0, 1] floats as k/255,
        # which round-trips bit-exactly through the same to_uint8 formula.
        # Training tensors (latents/log_probs/replay) are NOT touched.
        if isinstance(video, torch.Tensor) and video.is_floating_point():
            video = to_uint8(video)
        replay_tensors = model.export_replay_tensors(state)
        # Carry the rollout proposal mean alongside the model's replay tensors so
        # it concatenates + lands under the denoise segment like old_log_prob,
        # readable at replay via replay_tensor_dict("denoise"). Only present when
        # a trust-region recipe opted in (return_prev_sample_mean).
        if denoise_result.prev_sample_means is not None:
            replay_tensors = {
                **replay_tensors,
                "old_prev_sample_mean": denoise_result.prev_sample_means,
            }
        # Frozen reference noise_pred per step (cache_ref_noise_pred). Lands under
        # the denoise segment like old_log_prob; the SDE evaluator reads it at
        # replay via replay_tensor_dict("denoise") to skip the ref forward.
        if denoise_result.ref_noise_preds is not None:
            replay_tensors = {
                **replay_tensors,
                "ref_noise_pred": denoise_result.ref_noise_preds,
            }
        context = dict(model.export_batch_context(state))

        decode_peak_bytes = cuda_peak_allocated_bytes()
        decode_peak_mb = cuda_peak_allocated_mb()
        memory = None
        if denoise_result.memory is not None and decode_peak_bytes is not None:
            memory = {**denoise_result.memory, "decode_peak_bytes": decode_peak_bytes}
        phase_peaks = [
            peak for peak in (denoise_result.peak_memory_mb, decode_peak_mb) if peak is not None
        ]

        return DiffusionBatchResult(
            batch=batch,
            observations=denoise_result.observations,
            actions=denoise_result.actions,
            log_probs=denoise_result.log_probs,
            timesteps=denoise_result.timesteps,
            kl=denoise_result.kl,
            video=video,
            replay_tensors=replay_tensors,
            context=context,
            peak_memory_mb=max(phase_peaks) if phase_peaks else None,
            memory=memory,
            stage_durations=dict(stage_durations or {}),
            engine_counters={
                **denoise_result.engine_counters,
                "diffusion_replay_tensor_bytes": trajectory_tensor_bytes(replay_tensors),
                "diffusion_video_bytes": trajectory_tensor_bytes(video),
            },
        )

    # -- family hooks --------------------------------------------------

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Encode prompt conditioning for a single prompt batch."""

        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            video_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            request=video_request,
        )

    # Encoded keys copied through UNREPEATED by the default build_batch_encoded.
    # For batch-shared tensors whose leading dim is not a batch axis (FLUX's
    # ``text_ids`` is ``[seq, 3]``), the generic repeat would corrupt the shape,
    # so the family lists them here instead of overriding the whole method.
    # Non-tensor values (PIL reference images, python lists) already pass
    # through ``repeat_batch`` untouched and need no listing.
    batch_passthrough_keys: tuple[str, ...] = ()

    def build_batch_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Build per-sample encoded tensors for one sample batch."""

        del generation_request, video_request, params
        passthrough = set(self.batch_passthrough_keys)
        return {
            key: (
                value
                if key in passthrough
                else self.layout.repeat_batch(value, batch.sample_count)
            )
            for key, value in encoded.items()
        }

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any] | None:
        """Return additional family kwargs for model.prepare_sampling."""

        del encoded, generation_request, video_request, params, batch
        return None


__all__ = [
    "DiffusionBatchExecutorBase",
    "DiffusionBatchResult",
    "DiffusionRequestLayout",
    "DiffusionSamplingParams",
    "GenericDiffusionBatchExecutor",
    "ReferenceConditionedBatches",
]


class GenericDiffusionBatchExecutor(DiffusionBatchExecutorBase):
    """Generic batch executor for pure-data diffusion families.

    A family whose executor overrides no method (no ``build_batch_encoded`` /
    ``encode_prompt_for_batch``) is pure configuration: ``family`` / ``task``
    plus a few ``default_*`` values. Rather than ship a boilerplate subclass,
    it declares a ``model.executor`` block in its model config yaml and
    dispatches here; the launcher reads that block wholesale into these
    constructor kwargs (family/task come from the registry entry, the worker
    injects family/task from the launch contract). Families with real
    per-batch tensor logic (cosmos predict2/2.5, cosmos3, echo, wan i2v) keep
    their own subclass.
    """

    def __init__(
        self,
        model: Any,
        *,
        family: str,
        task: str,
        gatherer: GenerationBatchGatherer | None = None,
        num_frames: int = 1,
        max_sequence_length: int | None = None,
        fps: int | None = None,
        batch_passthrough_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(model, gatherer=gatherer)
        self.family = family
        self.task = task
        self.default_num_frames = int(num_frames)
        self.default_max_sequence_length = (
            None if max_sequence_length is None else int(max_sequence_length)
        )
        self.default_fps = None if fps is None else int(fps)
        self.batch_passthrough_keys = tuple(batch_passthrough_keys)
