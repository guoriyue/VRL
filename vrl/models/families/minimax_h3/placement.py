"""Explicit H3 block placement; not a selectable distributed trainer strategy."""

from __future__ import annotations

from typing import Any

import torch

from vrl.models.interfaces.runtime import ModelBuild


def transformer_device_map(
    transformer: Any, *, root_device: int, block_devices: tuple[int, ...]
) -> dict[str, int]:
    """Keep functional packing/selection local and move complete blocks only."""
    devices = (root_device, *block_devices)
    if any(type(device) is not int or device < 0 for device in devices):
        raise ValueError("H3 placement requires explicit nonnegative CUDA indices")
    if len(block_devices) != len(transformer.transformer_blocks):
        raise ValueError("H3 placement requires exactly one device per transformer block")
    mapping = {
        name: root_device
        for name, _ in transformer.named_children()
        if name != "transformer_blocks"
    }
    mapping.update(
        {f"transformer_blocks.{index}": device for index, device in enumerate(block_devices)}
    )
    return mapping


def load_partitioned_transformer(build: ModelBuild, block_devices: tuple[int, ...]) -> Any:
    """Load shards directly to their owners, preserving upstream FP32 exceptions."""
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    build.require_replay()
    if not build.use_lora:
        raise ValueError("Partitioned H3 replay currently requires LoRA")
    if build.precision.quantization or (build.torch_compile or {}).get("enable"):
        raise ValueError("Partitioned H3 replay does not yet support quantization or compile")
    root = torch.device(build.device)
    if root.type != "cuda" or root.index is None:
        raise ValueError("Partitioned H3 replay requires an explicit CUDA root device")
    config = MiniMaxH3Transformer3DModel.load_config(
        build.model_name_or_path, subfolder="transformer", **build.pretrained_kwargs
    )
    with init_empty_weights(include_buffers=True):
        skeleton = MiniMaxH3Transformer3DModel.from_config(config)
    mapping = transformer_device_map(skeleton, root_device=root.index, block_devices=block_devices)
    del skeleton
    return MiniMaxH3Transformer3DModel.from_pretrained(
        build.model_name_or_path,
        subfolder="transformer",
        torch_dtype=build.parameter_dtype,
        device_map=mapping,
        low_cpu_mem_usage=True,
        **build.pretrained_kwargs,
    )


def text_encoder_device_map(
    encoder: Any, *, root_device: int, layer_devices: tuple[int, ...]
) -> dict[str, int]:
    """Partition decoder layers while retaining vision and token inputs on root."""
    if any(type(device) is not int or device < 0 for device in (root_device, *layer_devices)):
        raise ValueError("H3 encoder placement requires explicit nonnegative CUDA indices")
    language_model = encoder.model.language_model
    if len(layer_devices) != len(language_model.layers):
        raise ValueError("H3 encoder placement requires exactly one device per decoder layer")
    mapping = {name: root_device for name, _ in encoder.named_children() if name != "model"}
    mapping.update(
        {
            f"model.{name}": root_device
            for name, _ in encoder.model.named_children()
            if name != "language_model"
        }
    )
    mapping.update(
        {
            f"model.language_model.{name}": root_device
            for name, _ in language_model.named_children()
            if name != "layers"
        }
    )
    mapping.update(
        {
            f"model.language_model.layers.{index}": device
            for index, device in enumerate(layer_devices)
        }
    )
    return mapping


def load_partitioned_text_encoder(
    build: ModelBuild, *, root_device: int, layer_devices: tuple[int, ...]
) -> Any:
    """Load the frozen H3 conditioner directly onto its explicit CUDA owners."""
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    rollout = build.require_rollout()
    config = Qwen3VLConfig.from_pretrained(
        build.model_name_or_path, subfolder="text_encoder", **build.pretrained_kwargs
    )
    with init_empty_weights(include_buffers=True):
        skeleton = Qwen3VLForConditionalGeneration(config)
    mapping = text_encoder_device_map(
        skeleton, root_device=root_device, layer_devices=layer_devices
    )
    del skeleton
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        build.model_name_or_path,
        subfolder="text_encoder",
        dtype=rollout.prompt_encoder_dtype or build.parameter_dtype,
        device_map=mapping,
        **build.pretrained_kwargs,
    )
    encoder.requires_grad_(False)
    return encoder.eval()
