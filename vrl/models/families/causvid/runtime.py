"""Runtime assembly and generation executor for CausVid."""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseExecutorBase,
)
from vrl.models.interfaces.runtime import (
    ModelBuild,
    RuntimeBundle,
    full_generation_bundle_metadata,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class CausVidChunkExecutor(ChunkAutoregressiveDenoiseExecutorBase):
    """Prompt/sample transport around CausVid's family-owned causal loop."""

    family: str = "causvid"
    task: str = "t2v"

    def __init__(self, model: Any, *, samples_per_chunk: int = 1) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


def build_causvid_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build full CausVid rollout state from pinned source and weight artifacts."""

    from vrl.models.families.causvid.model import CausVidModel
    from vrl.models.loader import (
        apply_rollout_quantization,
        validate_rollout_quantization_support,
    )
    from vrl.models.precision import apply_float32_precision
    from vrl.models.steps.denoise.common.vae_decode_memory import (
        apply_generation_memory_policy,
    )

    build.require_rollout()
    validate_rollout_quantization_support(build)
    logger.info("Building CausVid rollout runtime from %s", build.model_name_or_path)
    model = CausVidModel.from_build(build)
    if build.use_lora:
        model.apply_lora(build)
        apply_rollout_quantization(model, build)
        if build.precision.quantization:
            model.transformer.to(model.device)
    else:
        apply_rollout_quantization(model, build)
        model.apply_full_finetune()

    compile_config = build.torch_compile or {}
    if compile_config.get("enable"):
        model.torch_compile_transformer(compile_config["mode"])
    if build.num_steps is not None:
        model.set_num_steps(build.num_steps)
    apply_generation_memory_policy(
        model,
        memory_config=build.memory,
        owner="CausVid model",
    )
    apply_float32_precision(build.precision.float32_precision)
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        precision=build.precision,
        metadata=full_generation_bundle_metadata(),
    )


def build_causvid_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build the minimal transformer-only CausVid trainer runtime."""

    from vrl.models.families.causvid.model import CausVidReplayModel
    from vrl.models.steps.denoise.build import assemble_replay_bundle

    build.require_replay()
    logger.info("Building CausVid replay runtime from %s", build.model_name_or_path)
    return assemble_replay_bundle(CausVidReplayModel.from_build(build), build)


__all__ = [
    "CausVidChunkExecutor",
    "build_causvid_replay_runtime_bundle",
    "build_causvid_runtime_bundle",
]
