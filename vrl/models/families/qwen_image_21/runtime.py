"""Qwen-Image-2.1 batch adapter for optional ordered image conditioning."""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence_denoise import DenoiseBatchExecutorBase
from vrl.generation.bindings.full_sequence_denoise.layout import DenoiseSamplingParams
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import DenoiseRequest, GenerationRequest


class QwenImage21BatchExecutor(DenoiseBatchExecutorBase):
    """Load each prompt's references once, before expanding its sample group."""

    family = "qwen_image_21"
    task = "t2i"

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        model_request: DenoiseRequest,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        from PIL import Image

        from vrl.config.sampling_schema import QwenImage21SamplingSection

        item = generation_request.inputs[batch.prompt_index]
        if item.reference_video:
            raise ValueError("Qwen-Image-2.1 accepts reference images, not reference_video")
        paths = item.reference_images
        if len(paths) > 10:
            raise ValueError("Qwen-Image-2.1 accepts at most 10 reference images")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGBA"))
        options = QwenImage21SamplingSection.revalidate(
            generation_request.sampling, section="sampling"
        )
        return self.model.encode_prompt(
            item.prompt,
            model_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            reference_images=images,
            reference_resolution=options.reference_resolution,
            output_mode=options.output_mode,
        )
