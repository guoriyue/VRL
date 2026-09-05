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

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import DenoiseRequest, GenerationRequest
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def build_echo_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build the trainer replay bundle: Echo's velocity transformer only."""

    from vrl.models.families.echo.config import resolve_echo_video_dimensions

    video_height, video_width = resolve_echo_video_dimensions(build)
    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.models.families.echo.model import (
        EchoReplayModel,
        _resolve_echo_checkpoint,
        _resolve_gemma_dir,
    )

    logger.info("Building echo replay runtime bundle from %s", build.model_name_or_path)
    from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper

    dtype = build.parameter_dtype
    device = torch.device(build.device) if build.device is not None else torch.device("cpu")
    gemma_path = (build.model_config or {}).get("gemma_path")
    if not gemma_path:
        raise ValueError(
            "Echo requires model.gemma_path (the Gemma-3-12B encoder directory)",
        )
    echo = create_ltx2_wrapper(
        checkpoint_path=_resolve_echo_checkpoint(
            build.model_name_or_path,
            **build.revision_kwargs,
        ),
        gemma_path=_resolve_gemma_dir(
            str(gemma_path),
            **build.config_revision_kwargs("gemma_revision"),
        ),
        device=device,
        dtype=dtype,
        video_height=video_height,
        video_width=video_width,
    )
    # Same flow-matching scheduler as the rollout side (model.py from_build): Echo's
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
    num_steps = build.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)

    from vrl.models.steps.denoise.build import assemble_replay_bundle

    return assemble_replay_bundle(model, build)


class EchoBatchExecutor(DiffusionBatchExecutorBase):
    """Diffusion executor for Echo text-to-video rollouts."""

    family: str = "echo"
    task: str = "t2v"
    # Frame count and FPS remain request behavior. The spatial grid is instead
    # fixed when the Echo wrapper is constructed and checked below.
    default_num_frames: int = 25
    default_fps: int | None = 24

    def parse_sampling_params(self, request: GenerationRequest) -> DiffusionSamplingParams:
        """Require requests to use the wrapper's fixed latent-grid dimensions."""

        params = super().parse_sampling_params(request)
        dimensions = (params.model_request.height, params.model_request.width)
        expected = (self.model.video_height, self.model.video_width)
        if dimensions != expected:
            raise ValueError(
                f"Echo sampling dimensions {dimensions[0]}x{dimensions[1]} must equal "
                f"the model construction dimensions {expected[0]}x{expected[1]}",
            )
        return params

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: DenoiseRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        del params
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            video_request.negative_prompt or None,
        )


__all__ = [
    "EchoBatchExecutor",
    "build_echo_replay_runtime_bundle",
]
