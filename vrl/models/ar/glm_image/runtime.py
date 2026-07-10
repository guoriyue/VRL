"""GLM-Image family runtime for Ray rollout workers."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.ar import ARChunkInputs, ARDiscreteChunkExecutorBase
from vrl.generation.capabilities import FamilyCapability
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.ar.build import (
    ar_model_config_base,
    build_ar_runtime_bundle,
    extract_ar_runtime_spec,
)
from vrl.models.ar.capabilities import ar_discrete_family_capability
from vrl.models.ar.glm_image.model import (
    GlmImageConfig,
    GlmImageModel,
    GlmImageReplayModel,
    glm_image_token_num,
)
from vrl.models.ar.glm_image.runner import GlmImageTokenRunner
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

GLM_IMAGE_FAMILY_CAPABILITY = ar_discrete_family_capability("glm_image", "ar_t2i")


def build_glm_image_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared AR bundle assembly."""

    config = _glm_image_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=GlmImageModel(GlmImageConfig(**config)),
        capability=GLM_IMAGE_FAMILY_CAPABILITY,
    )


def build_glm_image_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build a GLM-Image trainer replay bundle without the DiT decode stack."""

    config = _glm_image_config_from_runtime_spec(spec)
    return build_ar_runtime_bundle(
        spec,
        model=GlmImageReplayModel(GlmImageConfig(**config)),
        capability=GLM_IMAGE_FAMILY_CAPABILITY,
        replay=True,
    )


def extract_glm_image_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any | None = None,
) -> RuntimeBuildSpec:
    """Slice GLM-Image runtime construction fields out of a whole RL cfg."""

    return extract_ar_runtime_spec(
        cfg,
        device,
        weight_dtype,
        ar_task="ar_t2i",
        default_model_path="zai-org/GLM-Image",
    )


# GLM-Image LoRA defaults; applied at read time so the carried ``model.lora``
# block only needs the values it overrides (same shape as the emu3 stub).
_GLM_IMAGE_LORA_DEFAULTS: dict[str, Any] = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ("q_proj", "v_proj"),
    "dropout": 0.0,
    "init": "gaussian",
}


def _glm_image_config_from_runtime_spec(spec: RuntimeBuildSpec) -> dict[str, Any]:
    sampling_config = spec.sampling_config or {}
    config = ar_model_config_base(spec, _GLM_IMAGE_LORA_DEFAULTS)

    for key in (
        "temperature",
        "top_p",
        "image_height",
        "image_width",
        "decode_num_inference_steps",
        "decode_guidance_scale",
    ):
        if key in sampling_config:
            config[key] = sampling_config[key]

    return config


class GlmImageChunkExecutor(ARDiscreteChunkExecutorBase):
    """AR executor for GLM-Image text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``temperature``: float — AR sampling temperature (checkpoint default 0.9).
    - ``top_p``: float — nucleus filtering (checkpoint default 0.75).
    - ``image_height`` / ``image_width``: int — target pixel size, multiples
      of 32; the processor derives the large + preview token grids from it.
    - ``max_text_length``: int — pad prompts to this length so ``L_text``
      is constant across multi-prompt requests (REQUIRED).
    - ``decode_num_inference_steps`` / ``decode_guidance_scale`` — frozen
      DiT decode segment knobs (postprocess only, never trained).
    - ``seed``: int | None — when set, ``torch.manual_seed`` is applied
      per chunk for parity tests.

    Family specifics vs Emu3:

    - NO AR-side CFG: the reference generation is plain sampling (see the
      runner docstring), so there is no ``guidance_scale`` sampling knob and
      no uncond branch — ``uncond_input_ids`` in the trajectory are zeros
      that satisfy the shared discrete schema (llamagen precedent); replay
      never reads them.
    - The token count is fully determined by the pixel target
      (``prev_h*prev_w + token_h*token_w``), so there is no
      ``image_token_num`` knob.
    - The trajectory context carries ``image_height``/``image_width``
      (replay rebuilds the mrope position schedule from them).
    """

    family: str = "glm_image"
    _runner_cls = GlmImageTokenRunner
    _runner_attention_family = "glm_image"
    task: str = "ar_t2i"
    family_capability: FamilyCapability = GLM_IMAGE_FAMILY_CAPABILITY

    def __init__(self, model: Any) -> None:
        """Construct the executor.

        Args:
          model: a ``GlmImageModel`` (or a stub exposing the same interface:
            ``processor``, ``device``, ``language_model``,
            ``encode_generation_prompts``, runner-step primitives, and
            ``decode_image_tokens``).
        """
        self.model = model

    # -- protocol ------------------------------------------------------

    def _ar_runner(self, request: GenerationRequest) -> GlmImageTokenRunner:
        """Build the GLM-Image runner without a shared attention backend.

        DOCUMENTED DEVIATION (llamagen precedent): GLM-Image decode positions
        are 3-axis mrope schedules and its decoder block carries
        post_self_attn/post_mlp layernorms — the shared ``vllm_paged``
        backend hand-walks 2-layernorm LLaMA-style blocks with 1D rope, and
        ``torch_native`` cannot inject position ids, so the runner drives the
        HF trunk + DynamicCache itself. An explicit ``attention_backend``
        request is rejected instead of silently ignored.
        """
        backend = request.sampling.get("attention_backend")
        if backend is not None:
            raise ValueError(
                "glm_image does not support request.sampling.attention_backend="
                f"{backend!r}: mrope position schedules are driven natively "
                "inside the family runner."
            )
        return GlmImageTokenRunner(self.model)

    def prepare_chunk_inputs(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ARChunkInputs:
        """Encode the single-branch prompt and wire the mrope decode loop."""

        sampling = request.sampling

        temperature = float(sampling.get("temperature", self.model.config.temperature))
        top_p = float(sampling.get("top_p", self.model.config.top_p))
        if "max_text_length" not in sampling:
            raise ValueError("request.sampling.max_text_length is required")
        max_text_length = int(sampling["max_text_length"])
        image_height = int(sampling.get("image_height", self.model.config.image_height))
        image_width = int(sampling.get("image_width", self.model.config.image_width))
        decode_steps = sampling.get("decode_num_inference_steps")
        decode_guidance = sampling.get("decode_guidance_scale")

        repeated_prompts = [chunk.prompt] * chunk.sample_count
        prompt_ids, prompt_mask, (token_h, token_w, prev_h, prev_w) = (
            self.model.encode_generation_prompts(
                repeated_prompts,
                max_text_length=max_text_length,
                image_height=image_height,
                image_width=image_width,
            )
        )
        cond_embeds = self._embed(prompt_ids)

        total_token_num = glm_image_token_num(image_height, image_width)
        return ARChunkInputs(
            max_new_tokens=total_token_num,
            decode_dtype=str(cond_embeds.dtype),
            init_args=(cond_embeds, prompt_mask),
            init_kwargs={
                "token_h": token_h,
                "token_w": token_w,
                "prev_h": prev_h,
                "prev_w": prev_w,
                "temperature": temperature,
                "top_p": top_p,
            },
            image_decode_kwargs={
                "height": image_height,
                "width": image_width,
                "prompts": repeated_prompts,
                "num_inference_steps": (
                    None if decode_steps is None else int(decode_steps)
                ),
                "guidance_scale": (
                    None if decode_guidance is None else float(decode_guidance)
                ),
            },
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            # No AR-side uncond branch exists (no CFG). Zeros keep the shared
            # discrete trajectory schema satisfied; replay never reads them.
            uncond_input_ids=torch.zeros_like(prompt_ids),
            uncond_attention_mask=torch.zeros_like(prompt_mask),
            context={
                "temperature": temperature,
                "top_p": top_p,
                "image_height": image_height,
                "image_width": image_width,
                "image_token_num": total_token_num,
            },
            # One single-branch prefill (no CFG uncond branch). Every
            # generated position is a free codebook draw, so the default
            # all-ones token mask is correct.
            prefill_forwards=1,
        )


__all__ = [
    "GLM_IMAGE_FAMILY_CAPABILITY",
    "GlmImageChunkExecutor",
    "build_glm_image_replay_runtime_bundle",
    "build_glm_image_runtime_bundle",
    "extract_glm_image_runtime_spec",
]
