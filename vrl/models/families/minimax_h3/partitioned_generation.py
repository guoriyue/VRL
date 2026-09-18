"""Explicit staged H3 generation; no public Ray/trainer strategy dispatch yet."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import torch

from vrl.models.families.minimax_h3.model import (
    MiniMaxH3Components,
    MiniMaxH3Model,
    build_flow_scheduler_class,
)
from vrl.models.families.minimax_h3.placement import (
    load_partitioned_text_encoder,
    load_partitioned_transformer,
    park_partitioned_text_encoder,
)
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle


@dataclass(frozen=True)
class H3GenerationPlacement:
    transformer_blocks: tuple[int, ...]
    encoder_root: int
    encoder_layers: tuple[int, ...]
    video_vae: int
    audio_vae: int
    park_encoder_for_decode: bool = True

    def validate(self, build: ModelBuild) -> None:
        rollout = build.require_rollout()
        if type(self.park_encoder_for_decode) is not bool:
            raise ValueError("park_encoder_for_decode must be a boolean")
        root = torch.device(build.device)
        devices = (
            *self.transformer_blocks,
            self.encoder_root,
            *self.encoder_layers,
            self.video_vae,
            self.audio_vae,
        )
        if (
            root.type != "cuda"
            or root.index is None
            or any(type(d) is not int or d < 0 for d in devices)
        ):
            raise ValueError("Partitioned H3 requires explicit nonnegative CUDA devices")
        if not self.transformer_blocks or not self.encoder_layers:
            raise ValueError("Partitioned H3 requires nonempty block and layer ownership")
        policy_devices = {root.index, *self.transformer_blocks}
        encoder_devices = {self.encoder_root, *self.encoder_layers}
        if (
            policy_devices & encoder_devices
            or not {self.video_vae, self.audio_vae} <= encoder_devices
        ):
            raise ValueError(
                "H3 encoder must be disjoint from policy; VAEs must reuse encoder devices"
            )
        if (
            not build.use_lora
            or build.precision.quantization
            or (build.torch_compile or {}).get("enable")
        ):
            raise ValueError("Partitioned H3 generation requires eager, unquantized LoRA")
        if rollout.pipeline_offload_mode != "none":
            raise ValueError("Partitioned H3 owns its staged offload lifecycle")


class PartitionedH3GenerationModel(MiniMaxH3Model):
    # The frozen base is already placed block-by-block across devices.
    trainable_roots_preplaced = True

    def __init__(self, *, pipeline: Any, device: Any, placement: H3GenerationPlacement):
        super().__init__(pipeline=pipeline, device=device)
        self.placement = placement

    @classmethod
    def from_partitioned_build(cls, build: ModelBuild, placement: H3GenerationPlacement):
        from diffusers import ModularPipeline

        placement.validate(build)
        transformer = load_partitioned_transformer(
            replace(build, rollout=None, generation_memory=None), placement.transformer_blocks
        )
        encoder = load_partitioned_text_encoder(
            build, root_device=placement.encoder_root, layer_devices=placement.encoder_layers
        )
        pipeline = ModularPipeline.from_pretrained(
            build.model_name_or_path, workflow="t2va", **build.pretrained_kwargs
        )
        names = ["vae", "audio_vae", "tokenizer", "processor", "scheduler", "audio_scheduler"]
        pipeline.load_components(names=names, torch_dtype=torch.float32, **build.pretrained_kwargs)
        missing = [name for name in names if getattr(pipeline, name, None) is None]
        if missing:
            raise RuntimeError(f"H3 component loading failed: {missing}")
        for vae in (pipeline.vae, pipeline.audio_vae):
            vae.requires_grad_(False).to("cpu", dtype=torch.float32)
        components = MiniMaxH3Components(
            transformer=transformer,
            text_encoder=encoder,
            vae=pipeline.vae,
            audio_vae=pipeline.audio_vae,
            tokenizer=pipeline.tokenizer,
            processor=pipeline.processor,
            scheduler=build_flow_scheduler_class().from_config(dict(pipeline.scheduler.config)),
            audio_scheduler=pipeline.audio_scheduler,
        )
        return cls(pipeline=components, device=build.device, placement=placement)

    def decode_latents(self, latents):
        with self._decode_encoder_context():
            try:
                self.pipeline.vae.to(torch.device("cuda", self.placement.video_vae))
                return super().decode_latents(latents)
            finally:
                self.pipeline.vae.to("cpu")

    def decode_audio(self, audio_rows):
        with self._decode_encoder_context():
            try:
                self.pipeline.audio_vae.to(torch.device("cuda", self.placement.audio_vae))
                return super().decode_audio(audio_rows)
            finally:
                self.pipeline.audio_vae.to("cpu")

    def _decode_encoder_context(self):
        if self.placement.park_encoder_for_decode:
            return park_partitioned_text_encoder(self.pipeline.text_encoder)
        return nullcontext()

    def move_frozen_components(self, device):
        raise RuntimeError(
            "Partitioned H3 owns component placement; whole-component moves are unsupported"
        )


def build_partitioned_h3_generation_runtime_bundle(
    build: ModelBuild, placement: H3GenerationPlacement
) -> RuntimeBundle:
    """Apply the shared native LoRA/memory passes to an explicitly placed model."""
    from vrl.models.steps.denoise.build import build_denoise_runtime_bundle

    placement.validate(build)

    class ConfiguredModel(PartitionedH3GenerationModel):
        @classmethod
        def from_build(cls, resolved):
            return cls.from_partitioned_build(resolved, placement)

    return build_denoise_runtime_bundle(build, model_cls=ConfiguredModel)
