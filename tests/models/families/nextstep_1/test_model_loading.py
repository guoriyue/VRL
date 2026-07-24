from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


def test_nextstep_rollout_resolves_the_pinned_snapshot(monkeypatch) -> None:
    from vrl.models.families.nextstep_1.config import NextStep1Config
    from vrl.models.families.nextstep_1.model import NextStep1Model
    from vrl.models.steps.token import loader

    calls: list[tuple[str, str | None]] = []
    pipeline_kwargs: dict = {}

    class Pipeline:
        def __init__(self, **kwargs):
            pipeline_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "gen_pipeline", SimpleNamespace(NextStepPipeline=Pipeline))
    monkeypatch.setattr(
        loader,
        "resolve_hf_checkpoint_dir",
        lambda path, *, revision=None, **kwargs: (
            calls.append((path, revision)) or "/cache/immutable-snapshot"
        ),
    )
    owner = SimpleNamespace(
        config=NextStep1Config(
            revision="immutable-revision",
            vae_revision="immutable-vae-revision",
            device="cpu",
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    NextStep1Model._load_pipeline(owner)

    assert calls == [
        ("stepfun-ai/NextStep-1.1", "immutable-revision"),
        ("stepfun-ai/NextStep-1-f8ch16-Tokenizer", "immutable-vae-revision"),
    ]
    assert pipeline_kwargs["model_name_or_path"] == "/cache/immutable-snapshot"
    assert pipeline_kwargs["vae_name_or_path"] == "/cache/immutable-snapshot"


def test_nextstep_rollout_preserves_an_unversioned_local_path(monkeypatch, tmp_path) -> None:
    from vrl.models.families.nextstep_1.config import NextStep1Config
    from vrl.models.families.nextstep_1.model import NextStep1Model
    from vrl.models.steps.token import loader

    pipeline_kwargs: dict = {}
    calls: list[tuple[str, str | None]] = []

    class Pipeline:
        def __init__(self, **kwargs):
            pipeline_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "gen_pipeline", SimpleNamespace(NextStepPipeline=Pipeline))
    monkeypatch.setattr(
        loader,
        "resolve_hf_checkpoint_dir",
        lambda path, *, revision=None, **kwargs: calls.append((path, revision)) or str(tmp_path),
    )
    owner = SimpleNamespace(
        config=NextStep1Config(model_path=str(tmp_path), revision=None, device="cpu"),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    NextStep1Model._load_pipeline(owner)

    assert pipeline_kwargs["model_name_or_path"] == str(tmp_path)
    assert calls == [
        (str(tmp_path), None),
        ("stepfun-ai/NextStep-1-f8ch16-Tokenizer", None),
    ]


def _decode_only_model(*, decoded_size: int):
    from vrl.models.families.nextstep_1.model import NextStep1Model

    model = object.__new__(NextStep1Model)
    torch.nn.Module.__init__(model)

    class UpstreamModel:
        @staticmethod
        def unpatchify(tokens, *, h, w):
            return torch.zeros(tokens.shape[0], 4, h, w)

    class VAE:
        dtype = torch.float32

        @staticmethod
        def decode(latent):
            return SimpleNamespace(
                sample=torch.zeros(latent.shape[0], 3, decoded_size, decoded_size),
            )

    object.__setattr__(
        model,
        "_pipeline",
        SimpleNamespace(
            model=UpstreamModel(),
            scaling_factor=1.0,
            shift_factor=0.0,
        ),
    )
    model.vae = VAE()
    return model


def test_nextstep_decode_enforces_requested_geometry() -> None:
    model = _decode_only_model(decoded_size=32)
    tokens = torch.zeros(2, 4, 8)

    decoded = model.decode_image_tokens(tokens, image_size=32)

    assert decoded.shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match="requested 64x64, decoded 32x32"):
        model.decode_image_tokens(tokens, image_size=64)


def test_nextstep_decode_rejects_non_square_token_grid() -> None:
    model = _decode_only_model(decoded_size=32)

    with pytest.raises(ValueError, match="square grid"):
        model.decode_image_tokens(torch.zeros(1, 3, 8), image_size=32)
