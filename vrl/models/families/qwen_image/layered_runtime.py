"""Qwen-Image-Layered batch adapter: one source image per prompt, ``layers`` RGBA outputs."""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchExecutorBase,
    ReferenceConditionedBatches,
)
from vrl.generation.bindings.full_sequence.layout import DenoiseSamplingParams
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest


class QwenImageLayeredBatchExecutor(ReferenceConditionedBatches, DenoiseBatchExecutorBase):
    """Encode the prompt's one source image into condition tokens before its sample group.

    The source is loaded with alpha; ``sampling.input_mode`` decides in the model
    whether that alpha is kept or replaced by opaque. ``sampling.layers`` is the
    per-request layer budget and reaches ``prepare_sampling`` here.
    """

    family = "qwen_image_layered"
    task = "t2i"
    min_reference_images = 1
    max_reference_images = 1
    reference_image_mode = "RGBA"

    @staticmethod
    def _options(generation_request: GenerationRequest) -> Any:
        from vrl.config.sampling_schema import QwenImageLayeredSamplingSection

        return QwenImageLayeredSamplingSection.revalidate(
            generation_request.sampling, section="sampling"
        )

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        item = generation_request.inputs[batch.prompt_index]
        if item.reference_video:
            raise ValueError(
                "Qwen-Image-Layered decomposes a reference image, not reference_video"
            )
        options = self._options(generation_request)
        return self.model.encode_prompt(
            item.prompt,
            params.model_request.negative_prompt or None,
            **params.text_encode_kwargs(),
            reference_image=self._reference_image_for_batch(generation_request, batch),
            height=params.model_request.height,
            width=params.model_request.width,
            input_mode=options.input_mode,
        )

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """The source already lives in ``encoded``; sampling only needs the layer budget."""

        del encoded, batch
        return {"layers": self._options(generation_request).layers}


__all__ = ["QwenImageLayeredBatchExecutor"]
