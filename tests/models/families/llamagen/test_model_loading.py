from __future__ import annotations

import pytest
import torch


def test_llamagen_t5_loader_uses_its_independent_revision(monkeypatch) -> None:
    from transformers import AutoTokenizer, T5EncoderModel

    from vrl.models.families.llamagen.config import LlamaGenConfig
    from vrl.models.families.llamagen.model import _load_t5_encoder

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


@pytest.mark.parametrize("filename", ["../outside.pt"])
def test_llamagen_member_cannot_escape_checkpoint_source(filename: str) -> None:
    from vrl.models.families.llamagen.model import _resolve_checkpoint_file

    with pytest.raises(ValueError, match="stay within its checkpoint source"):
        _resolve_checkpoint_file(
            "example/model",
            filename,
            revision="immutable",
        )
