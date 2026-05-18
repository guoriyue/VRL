from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.diffusion_fixtures import FakeDiffusionTransformer
from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2Model
from vrl.models.diffusion.cosmos.predict2_5.model import CosmosPredict25Model
from vrl.models.diffusion.sd3_5.model import SD3_5Model
from vrl.models.diffusion.wan_2_1.model import WanT2VDiffusersModel


class _FakeVAE:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.dtype = torch.float32
        self.calls: list[torch.Tensor] = []

    def decode(self, latents: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
        assert return_dict is False
        self.calls.append(latents)
        return (latents,)


class _ImageProcessor:
    def postprocess(self, image: torch.Tensor, *, output_type: str) -> torch.Tensor:
        assert output_type == "pt"
        return image


class _VideoProcessor:
    def postprocess_video(self, video: torch.Tensor, *, output_type: str) -> torch.Tensor:
        assert output_type == "pt"
        return video.permute(0, 2, 1, 3, 4)


def test_sd3_decode_latents_uses_shared_chunked_decoder_and_keeps_layout() -> None:
    vae = _FakeVAE(SimpleNamespace(scaling_factor=2.0, shift_factor=0.5))
    pipeline = SimpleNamespace(
        transformer=FakeDiffusionTransformer(),
        vae=vae,
        image_processor=_ImageProcessor(),
        device=torch.device("cpu"),
        decode_batch_size=1,
    )
    model = SD3_5Model(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(8.0).view(2, 1, 2, 2)

    image = model.decode_latents(latents)

    assert len(vae.calls) == 2
    torch.testing.assert_close(image, latents / 2.0 + 0.5)


def test_wan_decode_latents_preserves_bcthw_layout() -> None:
    vae = _FakeVAE(
        SimpleNamespace(latents_mean=[1.0], latents_std=[2.0], z_dim=1),
    )
    pipeline = SimpleNamespace(
        transformer=FakeDiffusionTransformer(),
        vae=vae,
        video_processor=_VideoProcessor(),
        device=torch.device("cpu"),
    )
    model = WanT2VDiffusersModel(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(16.0).view(2, 1, 2, 2, 2)

    video = model.decode_latents(latents)

    assert video.shape == latents.shape
    torch.testing.assert_close(video, latents * 2.0 + 1.0)


def test_cosmos_predict2_decode_latents_applies_sigma_data_and_layout() -> None:
    vae = _FakeVAE(
        SimpleNamespace(latents_mean=[1.0], latents_std=[4.0], z_dim=1),
    )
    pipeline = SimpleNamespace(
        transformer=FakeDiffusionTransformer(),
        scheduler=SimpleNamespace(config=SimpleNamespace(sigma_data=2.0)),
        vae=vae,
        video_processor=_VideoProcessor(),
        device=torch.device("cpu"),
    )
    model = CosmosPredict2Model(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(16.0).view(2, 1, 2, 2, 2)

    video = model.decode_latents(latents)

    assert video.shape == latents.shape
    torch.testing.assert_close(video, latents * 2.0 + 1.0)


def test_cosmos_predict25_decode_latents_matches_frames_and_layout() -> None:
    vae = _FakeVAE(SimpleNamespace())
    matched: list[int] = []

    def match_num_frames(video: torch.Tensor, frames: int) -> torch.Tensor:
        matched.append(frames)
        return video

    pipeline = SimpleNamespace(
        transformer=FakeDiffusionTransformer(),
        vae=vae,
        latents_mean=torch.ones(1, 1, 1, 1, 1),
        latents_std=torch.full((1, 1, 1, 1, 1), 3.0),
        vae_scale_factor_temporal=4,
        video_processor=_VideoProcessor(),
        _match_num_frames=match_num_frames,
        device=torch.device("cpu"),
    )
    model = CosmosPredict25Model(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(16.0).view(2, 1, 2, 2, 2)

    video = model.decode_latents(latents)

    assert matched == [5]
    assert video.shape == latents.shape
    torch.testing.assert_close(video, latents * 3.0 + 1.0)
