"""MiniMax-H3 family runtime: the batch-one executor and the replay bundle builder.

NOTE — no ``runner.py``: the H3 transformer takes a packed-sequence layout
(nine kwargs) and returns a ``(video, audio)`` pair, which does not fit
``DiffusionBackboneInput``; the call and the audio side-stream update run
inline in ``MiniMaxH3Model.forward_step``.

NOTE — the replay bundle is a custom builder, not the generic recipe: replay
needs TWO schedulers (video ``shift=12``, audio ``shift=3``, both
``MiniMaxH3Scheduler`` under different subfolders), and the video one has to be
the flow-convention subclass. The generic recipe loads exactly one scheduler.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.protocols import GenerationBatchGatherer
from vrl.generation.types import DenoiseRequest, GenerationRequest
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# 5 s at 24 fps is 120 frames; the video VAE decodes ``17n + 5``, so 124.
DEFAULT_NUM_FRAMES = 124
DEFAULT_FPS = 24
DEFAULT_MAX_SEQUENCE_LENGTH = 512


def load_h3_replay_components(
    build: ModelBuild, *, block_devices: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """Transformer + the two H3 schedulers as ``MiniMaxH3ReplayModel`` kwargs.

    Shared with VDN-H3, whose replay model is the same construction plus the
    hybrid-attention graft applied afterwards.
    """

    from diffusers import MiniMaxH3Scheduler

    from vrl.models.families.minimax_h3.model import build_flow_scheduler_class
    from vrl.models.loader import load_diffusers_transformer

    if block_devices is None:
        transformer = load_diffusers_transformer(build, "MiniMaxH3Transformer3DModel")
    else:
        from vrl.models.families.minimax_h3.placement import load_partitioned_transformer

        transformer = load_partitioned_transformer(build, block_devices)
    load_kwargs = build.pretrained_kwargs
    return {
        "transformer": transformer,
        "scheduler": build_flow_scheduler_class().from_pretrained(
            build.model_name_or_path,
            subfolder="scheduler",
            **load_kwargs,
        ),
        "audio_scheduler": MiniMaxH3Scheduler.from_pretrained(
            build.model_name_or_path,
            subfolder="audio_scheduler",
            **load_kwargs,
        ),
        "device": build.device,
    }


def build_minimax_h3_replay_runtime_bundle(
    build: ModelBuild, *, block_devices: tuple[int, ...] | None = None
) -> RuntimeBundle:
    """Transformer + the two H3 schedulers; no VAE, no conditioner."""

    from vrl.models.families.minimax_h3.model import MiniMaxH3ReplayModel
    from vrl.models.steps.denoise.build import assemble_replay_bundle

    logger.info("Building minimax_h3 replay runtime bundle from %s", build.model_name_or_path)
    components = load_h3_replay_components(build, block_devices=block_devices)
    model = MiniMaxH3ReplayModel(**components)
    if block_devices is not None:
        # The training strategy must not collapse the dispatched base onto the
        # root device when it places the trainable roots.
        model.trainable_roots_preplaced = True
    num_steps = build.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    return assemble_replay_bundle(model, build)


class MiniMaxH3BatchExecutor(DiffusionBatchExecutorBase):
    """Diffusion executor for MiniMax-H3 text-to-video rollouts.

    Strictly ``samples_per_generation_batch=1``: the transformer's batch axis is
    a replication axis over ONE packed layout, and the layout is a function of
    the prompt length, so one prompt per generation batch is the native shape.
    """

    family: str = "minimax_h3"
    task: str = "t2v"
    default_num_frames: int = DEFAULT_NUM_FRAMES
    default_fps: int | None = DEFAULT_FPS
    default_max_sequence_length: int | None = DEFAULT_MAX_SEQUENCE_LENGTH

    def __init__(
        self,
        model: Any,
        *,
        gatherer: GenerationBatchGatherer | None = None,
        samples_per_generation_batch: int | None = None,
    ) -> None:
        del samples_per_generation_batch  # one packed layout per forward
        super().__init__(model, gatherer=gatherer)

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: DenoiseRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        del video_request  # distilled: no negative prompt
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            None,
            **params.text_encode_kwargs(),
        )


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "DEFAULT_NUM_FRAMES",
    "MiniMaxH3BatchExecutor",
    "build_minimax_h3_replay_runtime_bundle",
    "load_h3_replay_components",
]
