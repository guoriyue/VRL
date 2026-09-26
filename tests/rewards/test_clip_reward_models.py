"""Tests for SigLIP aesthetic and CLIP preference reward models.

Scoring runs on tiny real CLIP/SigLIP repositories (``tests/rewards/fixtures.py``):
production loaders read genuine encoders and image processors from disk, the
aesthetic head loads the shipped V2.5 asset, and PickScore's arithmetic is
checked against an independent oracle. The revision-forwarding tests keep a
recorder because a local directory has no revision to observe (see their labels).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.rewards.fixtures import (
    build_tiny_clip_repo,
    build_tiny_siglip_repo,
    shipped_aesthetic_hidden_size,
)

pytest.importorskip("transformers")


@pytest.fixture(scope="session")
def aesthetic_siglip_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("aesthetic_predictor_v2_5")
    return build_tiny_siglip_repo(tmp_path_factory.mktemp("tiny-siglip-aesthetic"))


@pytest.fixture(scope="session")
def pickscore_clip_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny CLIP whose logit scale makes PickScore's ``/26`` cancel exactly."""

    return build_tiny_clip_repo(
        tmp_path_factory.mktemp("tiny-clip-pickscore"),
        projection_dim=16,
        logit_scale_init_value=math.log(26.0),
    )


def _solid_image(value: int, size: int = 12) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8))


def test_aesthetic_matches_released_head_and_siglip_oracle(aesthetic_siglip_repo: Path) -> None:
    from importlib import resources

    from transformers import SiglipVisionModel

    from vrl.rewards.models.aesthetic import AestheticRewardModel

    model = AestheticRewardModel(
        {"device": "cpu", "dtype": "float32", "model_name": str(aesthetic_siglip_repo)},
    )
    model.prepare_for_inference()
    head = model._module.layers.scoring_head[0]
    assert tuple(head.weight.shape) == (1024, shipped_aesthetic_hidden_size())
    assert float(head.weight.detach().abs().sum()) > 0.0
    assert not model._module.training
    assert next(model._module.parameters()).dtype == torch.float32

    images = [_solid_image(0), _solid_image(255)]
    pixels = model._processor(images=images, return_tensors="pt").pixel_values
    asset = resources.files("vrl.rewards.assets").joinpath("aesthetic_predictor_v2_5.pth")
    state = torch.load(asset, map_location="cpu", weights_only=True)
    encoder = SiglipVisionModel.from_pretrained(aesthetic_siglip_repo).eval()
    with torch.no_grad():
        features = encoder(pixel_values=pixels).pooler_output
        expected = torch.nn.functional.normalize(features, dim=-1)
        # Independent evaluation of the released head; dropout is off at inference.
        for index in (0, 2, 4, 6, 8):
            expected = torch.nn.functional.linear(
                expected,
                state[f"scoring_head.{index}.weight"].float(),
                state[f"scoring_head.{index}.bias"].float(),
            )
        actual = model._module(pixels).logits
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 1)
    assert actual[0].item() != actual[1].item()
    for value, score in zip((0, 255), expected.flatten(), strict=True):
        result = model.score_media(media=torch.full((3, 12, 12), value / 255), prompt="")
        assert result["aesthetic"] == pytest.approx(score.item(), abs=1e-5)


def test_aesthetic_video_scores_three_evenly_spaced_frames(aesthetic_siglip_repo: Path) -> None:
    from vrl.rewards.models.aesthetic import AestheticRewardModel

    model = AestheticRewardModel(
        {"device": "cpu", "dtype": "float32", "model_name": str(aesthetic_siglip_repo)},
    )
    video = torch.zeros(3, 8, 12, 12)
    video[:, 2] = 1.0
    video[:, 4] = 0.5
    video[:, 6] = 0.25
    scored = model.score_media(media=video, prompt="")
    expected = [
        model.score_media(media=video[:, index], prompt="")["aesthetic"] for index in (2, 4, 6)
    ]
    assert scored == {"aesthetic": pytest.approx(sum(expected) / 3)}


@pytest.mark.real_cover(
    "tests/rewards/inference/test_in_process_runtime.py"
    "::test_real_aesthetic_score_parks_stably_across_two_cycles",
    why="Local directories ignore revisions; record the converter call to verify Hub pinning.",
)
@pytest.mark.parametrize("revision", [None, "aesthetic-immutable-revision"])
def test_aesthetic_model_passes_revision_and_packaged_head(
    monkeypatch: pytest.MonkeyPatch,
    aesthetic_siglip_repo: Path,
    revision: str | None,
) -> None:
    import aesthetic_predictor_v2_5

    from vrl.rewards.models.aesthetic import AestheticRewardModel

    calls = []
    convert = aesthetic_predictor_v2_5.convert_v2_5_from_siglip

    def record(**kwargs):
        calls.append(kwargs.copy())
        return convert(**{**kwargs, "encoder_model_name": str(aesthetic_siglip_repo)})

    monkeypatch.setattr(aesthetic_predictor_v2_5, "convert_v2_5_from_siglip", record)
    config = {"device": "cpu", "dtype": "float32", "model_revision": revision}
    AestheticRewardModel(config)._load_module()
    assert len(calls) == 1
    assert calls[0]["encoder_model_name"] == "google/siglip-so400m-patch14-384"
    assert calls[0].get("revision") == revision
    assert Path(calls[0]["predictor_name_or_path"]).name == "aesthetic_predictor_v2_5.pth"
    assert Path(calls[0]["predictor_name_or_path"]).is_file()


def test_pickscore_matches_an_independent_cosine_oracle(pickscore_clip_repo: Path) -> None:
    """With ``logit_scale == 26`` the production ``logit_scale * (t @ i.T) / 26``
    collapses to the mean matched-pair cosine similarity, which
    ``F.cosine_similarity`` computes by a different route. Change ``/26`` to
    ``/13`` and the score doubles.

    The ``.diag()`` choice is not observable here: every image shares one prompt,
    so the diagonal mean equals the full-matrix mean. That needs per-image
    prompts, which is a real end-to-end concern, not this test's.
    """

    from vrl.rewards.models.pickscore import PickScoreRewardModel

    model = PickScoreRewardModel(
        {
            "device": "cpu",
            "model_name": str(pickscore_clip_repo),
            "processor_name": str(pickscore_clip_repo),
        },
    )
    images = [_solid_image(0), _solid_image(128), _solid_image(255)]
    prompt = "a green square"

    score = model._score(prompt, images)

    clip = model._module_for_inference()
    with torch.no_grad():
        image_inputs = model._processor(images=images, return_tensors="pt")
        image_embeds = clip.get_image_features(**image_inputs).pooler_output
        text_inputs = model._processor(
            text=[prompt] * len(images), padding=True, return_tensors="pt"
        )
        text_embeds = clip.get_text_features(**text_inputs).pooler_output
        expected = torch.nn.functional.cosine_similarity(text_embeds, image_embeds, dim=-1).mean()
    assert float(clip.logit_scale.exp()) == pytest.approx(26.0, rel=1e-6)
    assert score == pytest.approx(float(expected), abs=1e-6)


def test_pickscore_score_media_dispatches_tensors_and_rejects_non_media(
    pickscore_clip_repo: Path,
) -> None:
    """An image scores itself, a video its middle frame; invalid media raises."""

    from vrl.rewards.models.pickscore import PickScoreRewardModel

    model = PickScoreRewardModel(
        {
            "device": "cpu",
            "model_name": str(pickscore_clip_repo),
            "processor_name": str(pickscore_clip_repo),
        },
    )
    video = torch.zeros(3, 5, 12, 12)
    video[:, 2] = 1.0  # only the middle frame is white

    image_score = model.score_media(media=torch.ones(3, 12, 12), prompt="a square")
    video_score = model.score_media(media=video, prompt="a square")
    reference = model._score("a square", [_solid_image(255)])

    assert image_score == {"pickscore": pytest.approx(reference)}
    assert video_score == {"pickscore": pytest.approx(reference)}
    with pytest.raises(TypeError, match="reward media must be"):
        model.score_media(media="not-media", prompt="a square")


def test_pickscore_score_media_scores_the_middle_frame_of_each_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A [B,C,T,H,W] batch reaches the scorer as one RGB PIL image per sample."""
    from vrl.rewards.models.pickscore import PickScoreRewardModel

    media = torch.zeros(2, 3, 5, 12, 12)
    media[:, :, 2] = 1.0  # only the middle frame is white
    seen: list[list[object]] = []
    monkeypatch.setattr(
        PickScoreRewardModel,
        "_score",
        lambda self, prompt, images: seen.append(images) or 0.5,
    )

    result = PickScoreRewardModel({"device": "cpu"}).score_media(media=media, prompt="p")

    assert result == {"pickscore": 0.5}
    (images,) = seen
    assert [(image.mode, image.size) for image in images] == [("RGB", (12, 12))] * 2
    assert all(image.getextrema() == ((255, 255),) * 3 for image in images)


@pytest.mark.real_cover(
    None,
    why=(
        "a local directory has no revision: CLIPModel.from_pretrained(<dir>, revision=...) "
        "silently ignores the argument, so which revision reached the hub loaders can only "
        "be observed by recording the call; the real PickScore_v1 hub load has no "
        "opt-in counterpart yet"
    ),
    tracked_in="docs/sprints/done/SPRINT_reward-tiny-real-and-optional-lanes.md",
)
@pytest.mark.parametrize(
    ("processor_revision", "model_revision"),
    [
        (None, None),
        ("processor-immutable-revision", "model-immutable-revision"),
    ],
)
def test_pickscore_passes_optional_revisions_to_matching_loaders(
    monkeypatch: pytest.MonkeyPatch,
    processor_revision: str | None,
    model_revision: str | None,
) -> None:
    """Processor and model revisions remain independent optional boundaries."""
    import transformers

    from vrl.rewards.models.pickscore import PickScoreRewardModel

    calls: list[tuple[str, str, dict[str, str]]] = []

    class _FakeProcessor:
        pass

    class _FakeClip:
        def eval(self) -> _FakeClip:
            return self

        def to(self, *args, **kwargs) -> _FakeClip:
            return self

    def load_processor(name: str, **kwargs: str) -> _FakeProcessor:
        calls.append(("processor", name, kwargs))
        return _FakeProcessor()

    def load_clip(name: str, **kwargs: str) -> _FakeClip:
        calls.append(("model", name, kwargs))
        return _FakeClip()

    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        staticmethod(load_processor),
    )
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", staticmethod(load_clip))
    config = {"device": "cpu"}
    if processor_revision is not None:
        config["processor_revision"] = processor_revision
    if model_revision is not None:
        config["model_revision"] = model_revision

    PickScoreRewardModel(config)._load_module()

    processor_kwargs = {"revision": processor_revision} if processor_revision is not None else {}
    model_kwargs = {"revision": model_revision} if model_revision is not None else {}
    assert calls == [
        ("processor", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", processor_kwargs),
        ("model", "yuvalkirstain/PickScore_v1", model_kwargs),
    ]
