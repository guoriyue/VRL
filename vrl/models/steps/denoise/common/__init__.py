"""Shared diffusion model call helpers used by family models."""

from vrl.models.steps.denoise.common.backbone import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    EncoderAttentionMaskRunnerBase,
)
from vrl.models.steps.denoise.common.cfg import DiffusionBranch
from vrl.models.steps.denoise.common.latent_decode import (
    ChunkedLatentDecoder,
    LatentDecodePlan,
    VaeDecodeMixin,
)
from vrl.models.steps.denoise.common.masked_prompt import (
    MaskedPromptCollectorMixin,
    MaskedPromptModelMixin,
    MaskedPromptSamplingState,
    TrainTimestepMaskedPromptSamplingState,
)
from vrl.models.steps.denoise.common.replay_tensors import (
    replay_tensor,
    shared_replay_tensor,
)
from vrl.models.steps.denoise.common.timestep import (
    broadcast_spatial_timestep,
    expand_batch_timestep,
    pack_eval_timestep,
    set_mu_shifted_timesteps,
)
from vrl.utils.tensors import expand_tensor_to_batch

__all__ = [
    "ChunkedLatentDecoder",
    "DiffusionBackboneCaller",
    "DiffusionBackboneInput",
    "DiffusionBackboneRunnerBase",
    "DiffusionBranch",
    "EncoderAttentionMaskRunnerBase",
    "LatentDecodePlan",
    "MaskedPromptCollectorMixin",
    "MaskedPromptModelMixin",
    "MaskedPromptSamplingState",
    "TrainTimestepMaskedPromptSamplingState",
    "VaeDecodeMixin",
    "broadcast_spatial_timestep",
    "expand_batch_timestep",
    "expand_tensor_to_batch",
    "pack_eval_timestep",
    "replay_tensor",
    "set_mu_shifted_timesteps",
    "shared_replay_tensor",
]
