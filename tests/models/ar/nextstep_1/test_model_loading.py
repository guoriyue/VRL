from __future__ import annotations

import sys
from types import SimpleNamespace

import torch


def test_nextstep_rollout_resolves_the_pinned_snapshot(monkeypatch) -> None:
    from vrl.models.ar import loader
    from vrl.models.ar.nextstep_1.model import NextStep1Config, NextStep1Model

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
    from vrl.models.ar import loader
    from vrl.models.ar.nextstep_1.model import NextStep1Config, NextStep1Model

    pipeline_kwargs: dict = {}
    calls: list[tuple[str, str | None]] = []

    class Pipeline:
        def __init__(self, **kwargs):
            pipeline_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "gen_pipeline", SimpleNamespace(NextStepPipeline=Pipeline))
    monkeypatch.setattr(
        loader,
        "resolve_hf_checkpoint_dir",
        lambda path, *, revision=None, **kwargs: (
            calls.append((path, revision)) or str(tmp_path)
        ),
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
