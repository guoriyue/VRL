"""REAL-CUDA group-shared start: the Wan family's own ``prepare_sampling`` with
diffusers' real ``WanPipeline.prepare_latents`` bound to a tiny pipeline shell.
The executor draws the group's one-row latent on the device and expands it;
two batch widths and an OOM-split child of the same prompt must start from
the same latent, and the SDE step noise must still differ per sample."""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from diffusers import UniPCMultistepScheduler  # noqa: E402
from diffusers.pipelines.wan.pipeline_wan import WanPipeline  # noqa: E402

from vrl.generation.bindings.full_sequence_denoise.executor import (  # noqa: E402
    DenoiseBatchExecutorBase,
)
from vrl.generation.steps.denoise.config import DenoiseLoopConfig, DenoiseSDEParams  # noqa: E402
from vrl.generation.types import DenoiseRequest  # noqa: E402
from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel  # noqa: E402

pytestmark = pytest.mark.gpu


class _Executor(DenoiseBatchExecutorBase):
    family = "wan_2_1"
    task = "t2v"


def _model(device: torch.device) -> WanT2VDiffusersModel:
    transformer = torch.nn.Linear(4, 4).to(device)
    transformer.config = SimpleNamespace(in_channels=4)
    pipe = SimpleNamespace(
        transformer=transformer,
        scheduler=UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
        ),
        vae=SimpleNamespace(config=SimpleNamespace(z_dim=4)),
        vae_scale_factor_spatial=8,
        vae_scale_factor_temporal=4,
    )
    # The real diffusers draw and its "latents given" path, on the real device.
    pipe.prepare_latents = types.MethodType(WanPipeline.prepare_latents, pipe)
    return WanT2VDiffusersModel(pipeline=pipe, device=device)


def _config(start: int, count: int, group_seed: int | None) -> DenoiseLoopConfig:
    return DenoiseLoopConfig(
        sample_start=start,
        sample_count=count,
        seed=11,
        sde=DenoiseSDEParams(noise_level=0.7, sde_type="flow_grpo"),
        sde_window=None,
        initial_noise_seed=group_seed,
    )


def test_group_shares_one_start_across_batch_widths_and_oom_split_on_cuda() -> None:
    device = torch.device("cuda")
    executor = _Executor(_model(device))
    request = DenoiseRequest(
        width=64, height=64, frame_count=5, num_steps=2, guidance_scale=1.0, seed=11
    )
    single = {"prompt_embeds": torch.zeros(1, 3, 4, device=device)}

    def latents(start: int, count: int, group_seed: int | None) -> torch.Tensor:
        config = _config(start, count, group_seed)
        initial = executor.draw_group_initial_latents(
            request=request, encoded=single, config=config
        )
        state = executor.prepare_denoise_state(
            request=request,
            encoded={"prompt_embeds": torch.zeros(count, 3, 4, device=device)},
            config=config,
            initial_latents=initial,
        )
        assert state.latents.device.type == "cuda"
        assert state.latents.shape == (count, 4, 2, 8, 8)
        return state.latents

    whole = latents(0, 4, 21)
    halves = torch.cat([latents(0, 2, 21), latents(2, 2, 21)])
    child = latents(3, 1, 21)

    assert all(torch.equal(row, whole[0]) for row in whole)
    assert torch.equal(halves, whole)
    assert torch.equal(child[0], whole[0])
    assert not torch.equal(latents(0, 4, 22)[0], whole[0])

    # Without the option each row is its own draw.
    plain = latents(0, 4, None)
    assert not torch.equal(plain[0], plain[1])
