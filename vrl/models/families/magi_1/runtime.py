"""MAGI-1 registry/runtime boundary."""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseExecutorBase,
)
from vrl.generation.protocols import ChunkGatherer
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle


class Magi1ChunkExecutor(ChunkAutoregressiveDenoiseExecutorBase):
    """Generation-only chunk-autoregressive executor for MAGI-1 4.5B."""

    family = "magi_1"
    task = "t2v"
    default_samples_per_chunk = 1

    def __init__(
        self,
        model: Any,
        *,
        gatherer: ChunkGatherer | None = None,
        samples_per_chunk: int = 1,
    ) -> None:
        # The official pipeline accepts one prompt/sample and owns one process.
        if int(samples_per_chunk) != 1:
            raise ValueError("MAGI-1 requires samples_per_chunk=1")
        super().__init__(model, gatherer=gatherer)


def build_magi_1_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build a rollout bundle backed by the isolated official subprocess."""

    if build.family != "magi_1":
        raise ValueError(
            f"MAGI-1 rollout builder received family {build.family!r}",
        )
    _validate_magi_1_precision(build)
    if build.use_lora:
        raise RuntimeError(
            "MAGI-1 does not support VRL-managed LoRA: its official runtime is "
            "generation-only and runs out of process",
        )
    if build.precision.quantization is not None:
        raise RuntimeError(
            "MAGI-1 ignores VRL rollout quantization; select an official MAGI "
            "quantized config/checkpoint instead",
        )
    if build.torch_compile is not None:
        raise RuntimeError(
            "MAGI-1 ignores VRL torch_compile because the official model runs "
            "in an isolated subprocess",
        )

    from vrl.models.families.magi_1.model import Magi1SubprocessModel

    model = Magi1SubprocessModel.from_build(build)
    return RuntimeBundle(
        model=model,
        trainable_modules={},
        scheduler=None,
        raw_handle=model,
        precision=build.precision,
        loads_full_generation_modules=True,
    )


def _validate_magi_1_precision(build: ModelBuild) -> None:
    """Reject VRL precision knobs the official subprocess cannot honor."""

    import torch

    rollout = build.require_rollout()
    if build.parameter_dtype != torch.bfloat16 or build.precision.dtype != "bf16":
        raise ValueError(
            "MAGI-1 official 4.5B DiT fixes params_dtype=torch.bfloat16; "
            "precision.rollout.dtype and runtime parameter dtype must both be bf16",
        )
    if rollout.prompt_encoder_dtype != torch.float32:
        raise ValueError(
            "MAGI-1 official T5 runtime fixes torch_dtype=torch.float32; "
            "precision.rollout.prompt_encoders.dtype must be fp32",
        )
    if build.precision.outer_autocast:
        raise ValueError(
            "MAGI-1 runs outside VRL autocast; precision.rollout.outer_autocast must be false",
        )
    if build.precision.float32_precision != "ieee":
        raise ValueError(
            "MAGI-1 official subprocess does not apply VRL TF32 policy; "
            "precision.float32_precision must be ieee",
        )


__all__ = [
    "Magi1ChunkExecutor",
    "build_magi_1_runtime_bundle",
]
