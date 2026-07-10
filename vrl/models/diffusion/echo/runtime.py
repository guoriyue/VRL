"""JoyAI-Echo family runtime (single-shot video flow-matching policy).

Echo is not diffusers-backed, so the replay bundle does NOT use
``load_diffusers_transformer``; it builds the velocity transformer through Echo's
own ``create_ltx2_wrapper`` (transformer only — no text encoder / VAE / audio).

The rollout bundle routes through the shared diffusion builder (2026-07-01),
which wires rollout quantization and torch.compile — both conditional no-ops
while the config leaves them off, and both still UNVALIDATED on the LTX
transformer (built against diffusers-shaped modules). Before enabling either
knob on Echo, run a real-rollout parity check on an 80GB card.
"""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
def _gemma_path_from_spec(spec: RuntimeBuildSpec) -> str:
    gemma_path = (spec.model_config or {}).get("gemma_path")
    if not gemma_path:
        raise ValueError(
            "Echo requires model.gemma_path (the Gemma-3-12B encoder directory)",
        )
    return str(gemma_path)


def build_echo_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle: Echo's velocity transformer only."""

    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.models.diffusion.echo.model import (
        EchoReplayModel,
        _resolve_echo_checkpoint,
        _resolve_gemma_dir,
    )

    logger.info("Building echo replay runtime bundle from %s", spec.model_name_or_path)
    from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper

    dtype = resolve_torch_dtype(spec.dtype)
    device = torch.device(spec.device) if spec.device is not None else torch.device("cpu")
    sampling = getattr(spec, "sampling_config", None) or {}
    echo = create_ltx2_wrapper(
        checkpoint_path=_resolve_echo_checkpoint(spec.model_name_or_path),
        gemma_path=_resolve_gemma_dir(_gemma_path_from_spec(spec)),
        device=device,
        dtype=dtype,
        video_height=int(sampling.get("height", 512)),
        video_width=int(sampling.get("width", 768)),
    )
    # Same flow-matching scheduler as the rollout side (model.py from_spec): Echo's
    # released DMD few-step sampler is bypassed; RL drives the velocity field with
    # the standard flow-matching SDE. num_train_timesteps=1000 is Echo/LTX's
    # train-time discretization (sigma = t / num_train_timesteps), and it MUST match the
    # rollout value so replay log-prob recompute reads the identical sigma table.
    model = EchoReplayModel(
        echo=echo,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        dtype=dtype,
        device=device,
    )
    # The replay scheduler feeds the SDE log-prob evaluator, which maps each
    # stored timestep -> sigma via scheduler.sigmas; populate the schedule (the
    # rollout side does this in prepare_sampling, but the trainer reads
    # bundle.scheduler directly).
    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)

    from vrl.models.diffusion.build import assemble_replay_bundle

    return assemble_replay_bundle(model, spec, family="echo")


def build_echo_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    from vrl.models.diffusion.build import extract_family_runtime_spec

    return build_echo_replay_runtime_bundle(
        extract_family_runtime_spec(cfg, device, weight_dtype),
    )


class EchoChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Echo text-to-video rollouts."""

    family: str = "echo"
    task: str = "t2v"
    # Echo's default release resolution is 1280x736, 241 frames @ 24fps; the RL
    # smoke/proof runs override these down for single-card feasibility.
    default_num_frames: int = 25
    default_fps: int | None = 24
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,
        *,
        samples_per_chunk: int = 1,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        del generation_request, video_request, params
        return self.model.encode_prompt(chunk.prompt)

__all__ = [
    "EchoChunkExecutor",
    "build_echo_replay_runtime_bundle",
    "build_echo_replay_runtime_bundle_from_cfg",
]
