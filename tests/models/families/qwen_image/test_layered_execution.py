"""Layer decomposition crosses encode, sample expansion, SDE collection, decode and gather."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageLayeredPipeline
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN_IN_CHANNELS,
    TINY_QWEN_JOINT_DIM,
    build_tiny_transformer,
    stamp_model_precision,
)
from vrl.generation.bindings.full_sequence.gather import DenoiseBatchGatherer
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.qwen_image.layered import QwenImageLayeredPipelineModel
from vrl.models.families.qwen_image.layered_runtime import QwenImageLayeredBatchExecutor
from vrl.trajectory.reader import TrajectoryReader

_Z = TINY_QWEN_IN_CHANNELS // 4


def test_layered_executor_encodes_the_source_once_and_gathers_rgba_layers(tmp_path) -> None:
    encoded_sources: list[torch.Tensor] = []

    def encode(pixels: torch.Tensor):
        encoded_sources.append(pixels.clone())
        latent = pixels[:, :_Z, :, ::8, ::8]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(latents: torch.Tensor, *, return_dict: bool):
        b, _, _, h, w = latents.shape
        pixels = torch.full((b, 4, 1, h * 8, w * 8), -1.0)
        pixels[:, 1] = 1.0  # bright green, alpha (channel 3) fully transparent
        return (pixels,)

    def encode_prompt(*, prompt, num_images_per_prompt, max_sequence_length, device):
        del num_images_per_prompt, max_sequence_length, device
        torch.manual_seed(len(prompt))
        return torch.randn(1, 3, TINY_QWEN_JOINT_DIM), torch.ones(1, 3, dtype=torch.int64)

    pipe = SimpleNamespace(
        transformer=build_tiny_transformer("qwen_image", use_additional_t_cond=True),
        vae=SimpleNamespace(
            dtype=torch.float32,
            encode=encode,
            decode=decode,
            config=SimpleNamespace(
                z_dim=_Z,
                input_channels=4,
                latents_mean=[0.0] * _Z,
                latents_std=[1.0] * _Z,
                temperal_downsample=[False, True, True],
            ),
        ),
        vae_scale_factor=8,
        latent_channels=_Z,
        image_processor=VaeImageProcessor(vae_scale_factor=16),
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        encode_prompt=encode_prompt,
        _pack_latents=QwenImageLayeredPipeline._pack_latents,
    )
    pipe.prepare_latents = lambda *args, **kwargs: QwenImageLayeredPipeline.prepare_latents(
        pipe, *args, **kwargs
    )
    model = QwenImageLayeredPipelineModel(pipeline=pipe, device=torch.device("cpu"))
    stamp_model_precision(model)
    executor = QwenImageLayeredBatchExecutor(model, gatherer=DenoiseBatchGatherer())

    source = tmp_path / "room.png"
    Image.new("RGB", (40, 24), (200, 30, 30)).save(source)
    request = GenerationRequest(
        request_id="decompose",
        family="qwen_image_layered",
        task="t2i",
        inputs=[GenerationInput(prompt="a clean room", reference_images=[str(source)])],
        samples_per_prompt=2,
        sampling={
            "height": 32,
            "width": 48,
            "num_steps": 3,
            "guidance_scale": 1.0,
            "layers": 3,
            "seed": 11,
        },
    )

    with torch.no_grad():
        output = executor.forward_request(request)

    # The source is resized to the request geometry, given opaque alpha, encoded once.
    assert len(encoded_sources) == 1
    assert encoded_sources[0].shape == (1, 4, 1, 32, 48)
    assert (encoded_sources[0][:, 3] == 1.0).all()
    # Three RGBA layers per sample, uint8, transported as one tensor.
    assert output.output.shape == (2, 3, 4, 32, 48)
    assert output.output.dtype == torch.uint8
    assert (output.output[:, :, 3] == 0).all() and (output.output[:, :, 1] == 255).all()
    # Replay tensors carry the condition tokens and the exact clean latents of
    # the packed (layers + 1)-frame sequence; the layer budget rides the context.
    reader = TrajectoryReader(output.trajectory)
    tokens = (32 // 16) * (48 // 16)
    assert reader.tensor_value("denoise", "condition_latents").shape == (
        2,
        tokens,
        TINY_QWEN_IN_CHANNELS,
    )
    latents_clean, _ = reader.forward_process_replay("denoise", 0)
    assert latents_clean.shape == (
        2,
        4 * tokens,
        TINY_QWEN_IN_CHANNELS,
    )
    assert output.trajectory.context["layers"] == 3
