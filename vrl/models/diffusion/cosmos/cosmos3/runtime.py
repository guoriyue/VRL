"""Cosmos3 Omni vision generator family runtime.

NOTE — no ``runner.py``: predict2's runner massages the SD3-shaped backbone
contract + applies EDM ``/sigma`` finalize. Cosmos3's transformer takes a ~10-kwarg
interleaved joint sequence and returns a 3-tuple of lists, which does not fit
``DiffusionBackboneInput``; CFG + velocity-masking run inline in
``Cosmos3Model.forward_step`` (see model.py). A runner would be a thin no-op layer
with no protocol boundary, so it is intentionally omitted.

NOTE — replay bundle loads a FULL pipeline, not a transformer-only minimal model:
the replay model needs the pipeline's ``_prepare_text_segment`` /
``_prepare_vision_segment`` to rebuild the step-invariant packed_static. A
transformer-only "option B" (persist+replay the packed-static index tensors)
is the memory optimization follow-up; correctness comes first.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.loader import apply_rollout_quantization
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
from vrl.models.runtime_config import extract_runtime_spec
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
COSMOS3_FAMILY_CAPABILITY = diffusion_family_capability("cosmos3", "t2v")


def extract_cosmos3_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    return extract_runtime_spec(cfg, device, weight_dtype, task_variant="text2world")


def _apply_train_knobs(model: Any, spec: RuntimeBuildSpec) -> bool:
    use_lora = spec.use_lora
    if use_lora:
        model.apply_lora(spec)
    else:
        model.apply_full_finetune()
    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])
    return use_lora


def build_cosmos3_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    from vrl.models.diffusion.cosmos.cosmos3.model import Cosmos3Model

    logger.info("Building cosmos3 runtime bundle from %s", spec.model_name_or_path)
    model = Cosmos3Model.from_spec(spec)
    use_lora = _apply_train_knobs(model, spec)
    apply_rollout_quantization(model, spec)
    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        runtime_caps={},
        metadata={
            "model_path": spec.model_name_or_path,
            "family": COSMOS3_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **full_generation_bundle_metadata(),
            **apply_generation_memory_policy(
                model,
                memory_config=getattr(spec, "memory", None),
                owner="Cosmos3 VAE",
            ),
        },
    )


def build_cosmos3_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    from vrl.models.diffusion.cosmos.cosmos3.model import Cosmos3Model, Cosmos3ReplayModel

    logger.info("Building cosmos3 replay runtime bundle from %s", spec.model_name_or_path)
    # Reuse from_spec's pipeline loader, then wrap pipeline-shell in the replay model
    # (it needs the segment builders to rebuild packed_static at recompute time).
    driver = Cosmos3Model.from_spec(spec)
    model = Cosmos3ReplayModel(
        pipeline_shell=driver.pipeline,
        scheduler=driver.scheduler,
        device=spec.device,
    )
    use_lora = _apply_train_knobs(model, spec)

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        runtime_caps={},
        metadata={
            "model_path": spec.model_name_or_path,
            "family": COSMOS3_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **minimal_replay_bundle_metadata(),
        },
    )


def build_cosmos3_runtime_bundle_from_cfg(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBundle:
    return build_cosmos3_runtime_bundle(extract_cosmos3_runtime_spec(cfg, device, weight_dtype))


def build_cosmos3_replay_runtime_bundle_from_cfg(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBundle:
    return build_cosmos3_replay_runtime_bundle(extract_cosmos3_runtime_spec(cfg, device, weight_dtype))


class Cosmos3ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Cosmos3 Omni text-to-video rollouts.

    Strictly ``samples_per_chunk=1``: the Cosmos3OmniPipeline packs one sample at a
    time (no native B>1 path), so the chunk executor pins a single sample.
    """

    family: str = "cosmos3"
    task: str = "t2v"
    family_capability = COSMOS3_FAMILY_CAPABILITY
    default_num_frames: int = 93
    default_fps: int | None = 24
    default_max_sequence_length: int = 512

    def __init__(self, model: Any, *, samples_per_chunk: int = 1) -> None:
        del samples_per_chunk  # cosmos3 is strictly batch=1 (pipeline constraint)
        self.model = model
        self.default_samples_per_chunk = 1

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            guidance_scale=params.base.guidance_scale,
            num_frames=video_request.frame_count,
            height=video_request.height,
            width=video_request.width,
            fps=video_request.fps or 24,
        )

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        # batch=1: input_ids are python lists (not tensors); pass through unchanged.
        del generation_request, video_request, params, chunk
        return dict(encoded)


__all__ = [
    "Cosmos3ChunkExecutor",
    "build_cosmos3_replay_runtime_bundle",
    "build_cosmos3_replay_runtime_bundle_from_cfg",
    "build_cosmos3_runtime_bundle",
    "build_cosmos3_runtime_bundle_from_cfg",
    "extract_cosmos3_runtime_spec",
]
