"""Qwen-Image-2.1 batch adapter for optional ordered image conditioning."""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence_denoise import (
    DenoiseBatchExecutorBase,
    ReferenceConditionedBatches,
)
from vrl.generation.bindings.full_sequence_denoise.layout import DenoiseSamplingParams
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import DenoiseRequest, GenerationRequest


class QwenImage21BatchExecutor(ReferenceConditionedBatches, DenoiseBatchExecutorBase):
    """Load each prompt's references once, before expanding its sample group.

    Text-to-image takes no reference; editing takes any number, in order, with
    alpha kept (the VAE encodes RGBA). The model card documents up to ten, but
    that is a quality envelope, not a pipeline limit, so it is not enforced. The references and the family's
    ``reference_resolution`` / ``output_mode`` sampling fields all go to
    ``encode_prompt``, which fixes the latent prefix for the whole denoise.
    """

    family = "qwen_image_21"
    task = "t2i"
    min_reference_images = 0
    max_reference_images = None
    reference_image_mode = "RGBA"

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        model_request: DenoiseRequest,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        from vrl.config.sampling_schema import QwenImage21SamplingSection

        item = generation_request.inputs[batch.prompt_index]
        if item.reference_video:
            raise ValueError("Qwen-Image-2.1 accepts reference images, not reference_video")
        options = QwenImage21SamplingSection.revalidate(
            generation_request.sampling, section="sampling"
        )
        return self.model.encode_prompt(
            item.prompt,
            model_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            reference_images=self._reference_images_for_batch(generation_request, batch),
            reference_resolution=options.reference_resolution,
            output_mode=options.output_mode,
        )

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any] | None:
        """References live in the encoded latent prefix; sampling needs no extra kwargs."""

        del encoded, generation_request, batch
        return None
