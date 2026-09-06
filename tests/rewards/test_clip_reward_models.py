"""Tests for CLIP-backed in-process reward models.

Scoring runs on tiny real CLIP repositories (``tests/rewards/fixtures.py``): the
production loaders read a genuine ``CLIPModel``/``CLIPProcessor`` from disk, the
aesthetic head loads the shipped LAION asset, and PickScore's arithmetic is
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

from tests.rewards.fixtures import build_tiny_clip_repo, shipped_aesthetic_projection_dim

pytest.importorskip("transformers")


@pytest.fixture(scope="session")
def aesthetic_clip_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny CLIP whose projection width matches the shipped aesthetic head."""

    return build_tiny_clip_repo(
        tmp_path_factory.mktemp("tiny-clip-aesthetic"),
        projection_dim=shipped_aesthetic_projection_dim(),
        logit_scale_init_value=0.0,
    )


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


def test_aesthetic_model_loads_the_shipped_head_over_a_real_clip(
    aesthetic_clip_repo: Path,
) -> None:
    """The LAION head really loads and the projected CLIP feature really drives it.

    A zero-weight head returns 0.0 for every input, so only a genuinely loaded
    head can make two images score differently. Scores are compared, never
    pinned: their values depend on the tiny CLIP's random init.
    """

    from vrl.rewards.models.aesthetic import AestheticRewardModel

    model = AestheticRewardModel(
        {"device": "cpu", "dtype": "float32", "model_name": str(aesthetic_clip_repo)},
    )
    model.prepare_for_inference()

    head = model._module.mlp.layers[0]
    assert tuple(head.weight.shape) == (1024, shipped_aesthetic_projection_dim())
    assert float(head.weight.detach().abs().sum()) > 0.0

    black = model.score_media(media=torch.zeros(3, 12, 12), prompt="")
    white = model.score_media(media=torch.ones(3, 12, 12), prompt="")
    assert black["aesthetic"] != white["aesthetic"]
    # Batched images come back as one score per image (the ``.squeeze(1)`` contract).
    assert model._module([_solid_image(0), _solid_image(255)]).shape == (2,)


@pytest.mark.real_cover(
    "tests/rewards/inference/test_in_process_runtime.py"
    "::test_real_aesthetic_score_parks_stably_across_two_cycles",
    why=(
        "a local directory has no revision: CLIPModel.from_pretrained(<dir>, revision=...) "
        "silently ignores the argument, so which revision reached the hub loaders can only "
        "be observed by recording the call; the counterpart loads the real hub checkpoint"
    ),
)
@pytest.mark.parametrize("revision", [None, "aesthetic-immutable-revision"])
def test_aesthetic_model_passes_optional_revision_to_clip_loaders(
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    """The model and processor resolve the same optional CLIP revision."""
    import transformers

    from vrl.rewards.models.aesthetic import AestheticRewardModel

    calls: list[tuple[str, str, dict[str, str]]] = []

    class _FakeClip(torch.nn.Module):
        pass

    class _FakeProcessor:
        pass

    def load_clip(name: str, **kwargs: str) -> _FakeClip:
        calls.append(("model", name, kwargs))
        return _FakeClip()

    def load_processor(name: str, **kwargs: str) -> _FakeProcessor:
        calls.append(("processor", name, kwargs))
        return _FakeProcessor()

    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", staticmethod(load_clip))
    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        staticmethod(load_processor),
    )
    config = {"device": "cpu", "dtype": "float32"}
    if revision is not None:
        config["model_revision"] = revision

    AestheticRewardModel(config)._load_module()

    expected_kwargs = {"revision": revision} if revision is not None else {}
    assert calls == [
        ("model", "openai/clip-vit-large-patch14", expected_kwargs),
        ("processor", "openai/clip-vit-large-patch14", expected_kwargs),
    ]


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
    """NCHW tensors reach the scorer as PIL images; unknown media scores 0.0."""

    from vrl.rewards.models.pickscore import PickScoreRewardModel

    model = PickScoreRewardModel(
        {
            "device": "cpu",
            "model_name": str(pickscore_clip_repo),
            "processor_name": str(pickscore_clip_repo),
        },
    )
    batch = torch.zeros(2, 3, 12, 12)
    batch[1] = 1.0

    scored = model.score_media(media=batch, prompt="a square")
    reference = model._score("a square", [_solid_image(0), _solid_image(255)])

    assert scored == {"pickscore": pytest.approx(reference)}
    assert model.score_media(media="not-media", prompt="a square") == {"pickscore": 0.0}


@pytest.mark.parametrize("layout", ["BCTHW", "BTCHW"])
def test_pickscore_score_media_scores_the_middle_frame_of_a_video(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both video layouts reach the scorer as one RGB PIL image per sample."""
    from vrl.rewards.models.pickscore import PickScoreRewardModel

    frames = torch.zeros(2, 3, 5, 12, 12)
    frames[:, :, 2] = 1.0  # only the middle frame is white
    media = frames if layout == "BCTHW" else frames.permute(0, 2, 1, 3, 4)
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
