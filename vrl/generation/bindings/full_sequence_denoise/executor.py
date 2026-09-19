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
from pathlib import Path
from typing import Any

import torch

from vrl.generation.bindings.full_sequence_denoise.layout import (
    DenoiseRequestLayout,
    DenoiseSamplingParams,
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
    DenoiseRequest,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.trajectory.storage import (
    TrajectoryStoragePolicy,
    trajectory_tensor_bytes,
)
from vrl.utils.cuda_memory import (
    cuda_peak_allocated_bytes,
    reset_cuda_peak,
)
from vrl.utils.media import to_uint8
from vrl.utils.tensors import expand_tensor_to_batch
from vrl.utils.validation import require_int


@dataclass(slots=True)
class DenoiseBatchResult:
    """Output of one fused diffusion sample batch."""

    batch: GenerationSampleBatch
    # The denoise path once, ``(sample, num_steps + 1, *latent)``: observations
    # are ``latents[:, :-1]`` and actions ``latents[:, 1:]``. The gatherer
    # slices the two views after concatenation, so one storage crosses the
    # worker->driver wire and one lives on the driver instead of two.
    latents: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    # Decoded media (uint8), or sample-ordered references from a Ray actor.
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

    # The two trajectory views the loop result also exposes; readers outside
    # the wire (probes, family tests) keep naming them by role.
    @property
    def observations(self) -> Any:
        return self.latents[:, :-1]

    @property
    def actions(self) -> Any:
        return self.latents[:, 1:]

    # The media a reward scores, by one name across the family result types
    # (the Ray adapter replaces it with references for online scoring).
    @property
    def reward_media(self) -> Any:
        return self.video

    @reward_media.setter
    def reward_media(self, value: Any) -> None:
        self.video = value


class ReferenceConditionedBatches:
    """Reference-image threading for per-batch encode/prepare.

    Cosmos Predict2 Video2World and Wan 2.1 I2V condition every batch on a
    reference image carried by its ``GenerationInput``. Shared hooks load and
    pass that image to encoding and sampling preparation. Tensor expansion is
    owned by the base executor; families may specialize the encoded payload.
    """

    model: Any

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        model_request: DenoiseRequest,
        params: Any,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Encode text plus the active reference-image conditioning for one batch."""

        reference_image = self._reference_image_for_batch(generation_request, batch)
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            model_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            reference_image=reference_image,
        )

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Thread the active reference image into family prepare_sampling."""

        reference_image = encoded.get("reference_image")
        if reference_image is None:
            reference_image = self._reference_image_for_batch(generation_request, batch)
        return {"reference_image": reference_image}

    def _reference_image_for_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> Any:
        ref = request.inputs[batch.prompt_index].reference_image
        if ref is None:
            raise ValueError(
                f"{request.family} requires reference_image for prompt index {batch.prompt_index}",
            )
        if not isinstance(ref, (str, Path)) or not ref:
            return ref
        from PIL import Image

        with Image.open(ref) as image:
            return image.convert("RGB")


class DenoiseBatchExecutorBase(BatchExecutorBase):
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
    def layout(self) -> DenoiseRequestLayout:
        return DenoiseRequestLayout(
            default_num_frames=self.default_num_frames,
            default_fps=self.default_fps,
            default_max_sequence_length=self.default_max_sequence_length,
            sde_type=self.sde_type,
        )

    def parse_sampling_params(self, request: GenerationRequest) -> DenoiseSamplingParams:
        return self.layout.parse_sampling_params(request)

    def build_denoise_config(
        self,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> DenoiseLoopConfig:
        """Build the SDE denoise config for one sample batch."""

        return DenoiseLoopConfig(
            sample_start=batch.sample_start,
            sample_count=batch.sample_count,
            seed=params.model_request.seed,
            sde=params.sde,
            # Use the parsed window without drawing again inside the loop.
            # _forward_batch parses each batch; request-owned randomness keeps
            # the selected window identical across those re-parses.
            sde_window=params.sde_window,
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
        """Per-request variant of forward_plan: every batch of the request runs
        on this worker in one call, each result copied to pinned CPU memory
        before the next batch is produced. BIT-EXACT to forward_plan: the same
        per-batch stage methods (via forward_batch) and the SAME order-preserving
        merge_generation_batches, so the gathered output is identical; only the
        per-batch dispatch overhead disappears.

        The Ray worker calls this per request (all of a request's batches on one
        worker) instead of dispatching one forward_batch RPC per batch.
        """

        batches = self.forward_batches_pipelined(
            request,
            plan.sample_batches,
            completion_callback=completion_callback,
        )
        return self.merge_generation_batches(request, sample_rows, batches)

    def forward_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> DenoiseBatchResult:
        """Run the canonical diffusion batch flow and prepare its wire payload."""

        return self.apply_wire_storage_policy(
            request,
            self._forward_batch(request, batch, execute_steps=None),
        )

    def forward_probe_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
        *,
        execute_steps: int,
    ) -> DenoiseBatchResult:
        """Run a truncated canonical batch for startup memory sizing."""

        require_int(execute_steps, path="execute_steps", minimum=1)
        return self._forward_batch(request, batch, execute_steps=execute_steps)

    def _forward_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
        *,
        execute_steps: int | None,
    ) -> DenoiseBatchResult:
        from vrl.utils.profiling import profile_range

        stage_durations: dict[str, float] = {}
        params = self.parse_sampling_params(request)

        started = time.perf_counter()
        with profile_range("generation.prompt_encode"):
            encoded = self.encode_prompt_for_batch(
                generation_request=request,
                model_request=params.model_request,
                params=params,
                batch=batch,
            )
        stage_durations["encode"] = time.perf_counter() - started

        started = time.perf_counter()
        batch_encoded = self.expand_conditioning_to_batch(
            encoded=encoded,
            generation_request=request,
            batch=batch,
        )
        prepare_kwargs = self.build_prepare_kwargs(
            encoded=encoded,
            generation_request=request,
            batch=batch,
        )
        config = self.build_denoise_config(params, batch)
        if execute_steps is not None:
            config = replace(config, execute_steps=execute_steps)
        state = self.prepare_denoise_state(
            request=params.model_request,
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
        )
        stage_durations["decode"] = time.perf_counter() - started
        batch_result.stage_durations = stage_durations
        return batch_result

    def apply_wire_storage_policy(
        self,
        request: GenerationRequest,
        batch_result: DenoiseBatchResult,
    ) -> DenoiseBatchResult:
        """Apply rollout.trajectory_storage BEFORE tensors cross the wire.

        The same policy is re-applied driver-side when the trajectory batch is
        built (idempotent there); applying it here is what turns a dtype
        downcast into actual worker->driver transfer savings. The default
        preserve/preserve policy is a no-op, keeping the GRPO baseline
        bit-for-bit.
        """

        policy = request.trajectory_storage
        if policy is None or policy == TrajectoryStoragePolicy():
            return batch_result
        batch_result.latents = policy.apply_to_value(batch_result.latents)
        batch_result.log_probs = policy.apply_to_value(batch_result.log_probs)
        batch_result.timesteps = policy.apply_to_value(batch_result.timesteps)
        batch_result.kl = policy.apply_to_value(batch_result.kl)
        batch_result.replay_tensors = policy.apply_to_value(batch_result.replay_tensors)
        return batch_result

    def prepare_denoise_state(
        self,
        *,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        config: DenoiseLoopConfig,
        prepare_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Prepare latent state for one diffusion sample batch."""

        from vrl.utils.profiling import profile_range

        model = self.model
        if request.seed is not None:
            # Match the denoise generator's batch offset. Reusing the request
            # seed makes every one-sample native batch start from identical noise.
            request = replace(request, seed=request.seed + config.sample_start)
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
    ) -> DenoiseBatchResult:
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
        # The stochastic-window bounds [lo, hi) actually used for this batch's
        # rollout, one row per sample so chunk concatenation is trivial. The
        # trainer's timestep_selection="sde_window" reads it to train exactly
        # the steps that were policy actions — ODE steps outside the window
        # were deterministic and must never receive surrogate loss. Absent when
        # no window is configured (all steps stochastic).
        if config.sde_window is not None:
            window_lo, window_hi = config.sde_window
            replay_tensors = {
                **replay_tensors,
                "sde_window": torch.tensor(
                    [[int(window_lo), int(window_hi)]],
                    dtype=torch.int64,
                ).repeat(batch.sample_count, 1),
            }
        context = dict(model.export_batch_context(state))

        decode_peak_bytes = cuda_peak_allocated_bytes()
        memory = None
        peak_memory_mb = None
        if denoise_result.memory is not None and decode_peak_bytes is not None:
            memory = {**denoise_result.memory, "decode_peak_bytes": decode_peak_bytes}
            peak_memory_mb = max(memory["denoise_peak_bytes"], decode_peak_bytes) / (1024 * 1024)

        return DenoiseBatchResult(
            batch=batch,
            latents=denoise_result.latents,
            log_probs=denoise_result.log_probs,
            timesteps=denoise_result.timesteps,
            kl=denoise_result.kl,
            video=video,
            replay_tensors=replay_tensors,
            context=context,
            peak_memory_mb=peak_memory_mb,
            memory=memory,
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
        model_request: DenoiseRequest,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Encode prompt conditioning for a single prompt batch."""

        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            model_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            request=model_request,
        )

    # Encoded keys copied through UNREPEATED by the default expand_conditioning_to_batch.
    # For batch-shared tensors whose leading dim is not a batch axis (FLUX's
    # ``text_ids`` is ``[seq, 3]``), the generic repeat would corrupt the shape,
    # so the family lists them here instead of overriding the whole method.
    # Non-tensor values (PIL reference images, python lists) already pass
    # through input preparation untouched and need no listing.
    batch_passthrough_keys: tuple[str, ...] = ()

    def expand_conditioning_to_batch(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Expand already-encoded conditioning to this batch, preserving shared fields."""

        del generation_request
        passthrough = set(self.batch_passthrough_keys)
        batch_encoded: dict[str, Any] = {}
        for key, value in encoded.items():
            if key not in passthrough and isinstance(value, torch.Tensor) and value.ndim > 0:
                try:
                    value = expand_tensor_to_batch(value, batch.sample_count, materialize=True)
                except ValueError as error:
                    raise ValueError(f"encoded field {key!r}: {error}") from error
            batch_encoded[key] = value
        return batch_encoded

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any] | None:
        """Return additional family kwargs for model.prepare_sampling."""

        del encoded, generation_request, batch
        return None


__all__ = [
    "DenoiseBatchExecutorBase",
    "DenoiseBatchResult",
    "DenoiseRequestLayout",
    "DenoiseSamplingParams",
    "GenericDenoiseBatchExecutor",
    "ReferenceConditionedBatches",
]


class GenericDenoiseBatchExecutor(DenoiseBatchExecutorBase):
    """Generic batch executor for pure-data diffusion families.

    A family whose executor overrides no method (no ``expand_conditioning_to_batch`` /
    ``encode_prompt_for_batch``) is pure configuration: ``family`` / ``task``
    plus a few ``default_*`` values. Rather than ship a boilerplate subclass,
    it declares a ``model.executor`` block in its model config yaml and
    dispatches here; the launcher reads that block wholesale into these
    constructor kwargs (family/task come from the registry entry, the worker
    injects family/task from the launch contract). Families with real
    per-batch input preparation (cosmos predict2/2.5, cosmos3, echo, wan i2v) keep
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
        self.default_num_frames = num_frames
        self.default_max_sequence_length = max_sequence_length
        self.default_fps = fps
        self.batch_passthrough_keys = tuple(batch_passthrough_keys)
