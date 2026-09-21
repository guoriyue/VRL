"""Editing crosses encode, sample expansion, SDE collection, decode, and gather."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN21_CONTEXT_DIM,
    TINY_QWEN21_IN_CHANNELS,
    build_tiny_qwen_image_21_transformer,
    stamp_model_precision,
)
from vrl.generation.bindings.full_sequence_denoise.gather import DenoiseBatchGatherer
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.qwen_image_21.model import QwenImage21Model
from vrl.models.families.qwen_image_21.runtime import QwenImage21BatchExecutor


def test_edit_executor_encodes_rgba_references_once_per_batch_and_preserves_output(
    tmp_path,
) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from diffusers.image_processor import VaeImageProcessor

    # Fake only the large encoders/VAE; real transformer, processor, latent
    # packing, scheduler, SDE loop, and driver gather execute on CPU.
    encoded_images = []
    prompt_images = []

    def encode_image(image):
        encoded_images.append(image.clone())
        latent = torch.nn.functional.avg_pool2d(image[:, :, 0], 16).repeat(1, 2, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent.unsqueeze(2)))

    def decode_image(latents, *, return_dict):
        b, _, _, h, w = latents.shape
        pixels = torch.ones(b, 4, 1, h * 16, w * 16)
        pixels[:, 3] = -1  # Fully transparent VAE output, normalized alpha zero.
        return (pixels,)

    def encode_prompt(*, prompt, image, device, num_images_per_prompt):
        prompt_images.append(image)
        slots = sum(im.width * im.height // 1024 for im in image or [])
        embeds = torch.randn(1, slots + 3, TINY_QWEN21_CONTEXT_DIM)
        mask = torch.tensor([[False] * 3 + [True] * slots])
        return embeds, None, mask

    class Pipeline:
        prepare_latents = QwenImage21Pipeline.prepare_latents
        _encode_vae_image = QwenImage21Pipeline._encode_vae_image
        _pack_latents = staticmethod(QwenImage21Pipeline._pack_latents)
        _unpack_latents = staticmethod(QwenImage21Pipeline._unpack_latents)

    pipe = Pipeline()
    pipe.transformer = build_tiny_qwen_image_21_transformer()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    pipe.vae_scale_factor = 16
    pipe.latent_channels = TINY_QWEN21_IN_CHANNELS
    pipe.vae = SimpleNamespace(
        dtype=torch.float32,
        encode=encode_image,
        decode=decode_image,
        config=SimpleNamespace(z_dim=8, latents_mean=[0] * 8, latents_std=[1] * 8),
    )
    pipe.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=8)
    pipe.processor = SimpleNamespace(
        image_processor=SimpleNamespace(size={"shortest_edge": 1024, "longest_edge": 1048576})
    )
    pipe.encode_prompt = encode_prompt
    model = QwenImage21Model(pipeline=pipe, device=torch.device("cpu"))
    stamp_model_precision(model)
    executor = QwenImage21BatchExecutor(model, gatherer=DenoiseBatchGatherer())
    source, leaf = tmp_path / "source.png", tmp_path / "leaf.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(source)
    Image.new("RGBA", (128, 32), (0, 255, 0, 255)).save(leaf)
    request = GenerationRequest(
        request_id="edit",
        family="qwen_image_21",
        task="t2i",
        inputs=[GenerationInput(prompt="edit", reference_images=[str(source), str(leaf)])],
        samples_per_prompt=2,
        initial_noise_seeds=[42, 42],
        sampling={
            "height": 64,
            "width": 64,
            "num_steps": 3,
            "guidance_scale": 1.0,
            "reference_resolution": 64,
            "output_mode": "rgba",
            "seed": 17,
        },
    )
    with torch.no_grad():
        output = executor.forward_plan(
            request, request.sample_rows(), EnginePlan.from_request(request)
        )
    # Both samples share initial noise, while each reference is VAE-encoded only once.
    assert len(encoded_images) == 2
    assert encoded_images[0].shape == (1, 4, 1, 64, 64)
    assert encoded_images[1].shape == (1, 4, 1, 32, 128)
    assert (encoded_images[0][:, 3] == -1).all()
    assert [im.size for im in prompt_images[0]] == [(64, 64), (128, 32)]
    assert output.output.shape == (2, 4, 64, 64)
    assert output.output.dtype == torch.uint8
    assert (output.output[:, 3] == 0).all()
    # A later text-only RGB request must reset both reference and decode state.
    request.inputs = [GenerationInput(prompt="plain image")]
    request.sampling["output_mode"] = "rgb"
    with torch.no_grad():
        plain = executor.forward_plan(
            request, request.sample_rows(), EnginePlan.from_request(request)
        )
    assert plain.output.shape == (2, 3, 64, 64)
    assert (plain.output == 255).all()
    assert prompt_images[-1] is None
    # Fail before VAE work if the vision processor would independently resize
    # an undersized reference and break the shared token geometry.
    pipe.processor.image_processor.size["shortest_edge"] = 65536
    request.inputs = [GenerationInput(prompt="edit", reference_image=str(source))]
    with pytest.raises(ValueError, match="checkpoint vision processor's pixel limits"):
        executor.forward_plan(request, request.sample_rows(), EnginePlan.from_request(request))
    assert len(encoded_images) == 2
