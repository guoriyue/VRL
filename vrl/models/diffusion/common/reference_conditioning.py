"""Chunk hooks shared by reference-conditioned video executors (V2W / I2V)."""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.utils.media import load_reference_image


class ReferenceConditionedChunks:
    """Reference-image threading for per-chunk encode/prepare.

    Cosmos Predict2 Video2World and Wan 2.1 I2V condition every chunk on a
    reference image that can arrive globally (executor attribute) or
    per-sample (request metadata). The two executors had copy-pasted these
    hooks; ``build_chunk_encoded`` stays family-specific because the encoded
    payloads genuinely differ (Wan carries ``image_embeds``).
    """

    model: Any
    reference_image: Any

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Encode text plus the active reference-image conditioning for one chunk."""

        reference_image = self._reference_image_for_request(generation_request)
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=params.base.max_sequence_length,
            guidance_scale=params.base.guidance_scale,
            reference_image=reference_image,
        )

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Thread the active reference image into family prepare_sampling."""

        del encoded, video_request, params, chunk
        return {
            "reference_image": self._reference_image_for_request(
                generation_request,
            ),
        }

    def _reference_image_for_request(self, request: GenerationRequest) -> Any:
        return load_reference_image(
            request.metadata.get("reference_image", self.reference_image),
        )


__all__ = ["ReferenceConditionedChunks"]
