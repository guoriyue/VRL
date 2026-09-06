from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.families.nextstep_1.fixtures import (
    NEXTSTEP_VAE_DOWNSAMPLES,
    build_decode_only_nextstep_model,
    build_tiny_nextstep_vae,
    install_stub_nextstep_pipeline,
)


def test_nextstep_rollout_resolves_the_pinned_snapshot(monkeypatch) -> None:
    from vrl.models.families.nextstep_1.config import NextStep1Config
    from vrl.models.families.nextstep_1.model import NextStep1Model
    from vrl.models.steps.token import loader

    calls: list[tuple[str, str | None]] = []
    pipeline_kwargs = install_stub_nextstep_pipeline(monkeypatch)
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

    calls: list[tuple[str, str | None]] = []
    pipeline_kwargs = install_stub_nextstep_pipeline(monkeypatch)
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


@pytest.mark.real_cover(
    "tests/e2e/test_real_checkpoint_rl.py::test_real_checkpoint_online_rl_updates_trainable_weights",
    why=(
        "unpatchify and the pipeline assembly live in the upstream nextstep packages, which are "
        "not repository dependencies, so they stay stand-ins here; the e2e nextstep_1 case "
        "decodes through the real ones"
    ),
)
def test_nextstep_decode_enforces_requested_geometry() -> None:
    """A 4x4 token grid decodes to 32x32 through the real f8 VAE.

    The 32 is computed by diffusers from the grid side and the VAE's upsampling,
    so the mismatch error names a size the model really produced.
    """
    model = build_decode_only_nextstep_model(vae=build_tiny_nextstep_vae())
    tokens = torch.zeros(2, 16, 8)
    decoded_side = 4 * 2**NEXTSTEP_VAE_DOWNSAMPLES

    decoded = model.decode_image_tokens(tokens, image_size=decoded_side)

    assert decoded.shape == (2, 3, decoded_side, decoded_side)
    with pytest.raises(ValueError, match="requested 64x64, decoded 32x32"):
        model.decode_image_tokens(tokens, image_size=2 * decoded_side)


def test_nextstep_decode_rejects_non_square_token_grid() -> None:
    model = build_decode_only_nextstep_model(vae=build_tiny_nextstep_vae())

    with pytest.raises(ValueError, match="square grid"):
        model.decode_image_tokens(torch.zeros(1, 3, 8), image_size=32)
