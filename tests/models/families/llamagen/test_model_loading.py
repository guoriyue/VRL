from __future__ import annotations

import torch


def test_llamagen_t5_loader_uses_its_independent_revision(monkeypatch) -> None:
    from transformers import AutoTokenizer, T5EncoderModel

    from vrl.models.families.llamagen.model import LlamaGenConfig, _load_t5_encoder

    calls: list[tuple[str, str, dict]] = []
    encoder = torch.nn.Linear(1, 1)
    tokenizer = object()
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("tokenizer", path, kwargs)) or tokenizer,
    )
    monkeypatch.setattr(
        T5EncoderModel,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("encoder", path, kwargs)) or encoder,
    )

    loaded_encoder, loaded_tokenizer = _load_t5_encoder(
        LlamaGenConfig(t5_revision="immutable-t5-revision"),
    )

    assert loaded_encoder is encoder
    assert loaded_tokenizer is tokenizer
    assert calls[0] == (
        "tokenizer",
        "google/flan-t5-xl",
        {"revision": "immutable-t5-revision"},
    )
    assert calls[1][0:2] == ("encoder", "google/flan-t5-xl")
    assert calls[1][2]["revision"] == "immutable-t5-revision"
