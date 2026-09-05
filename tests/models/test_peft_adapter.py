from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from torch import nn

from vrl.models.peft_adapter import load_trainable_lora_adapter

peft = pytest.importorskip("peft")


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.v_proj = nn.Linear(2, 2, bias=False)


def _save_lora_config(path: Path) -> None:
    peft.LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.25,
        target_modules=["q_proj", "v_proj"],
    ).save_pretrained(path)


def _mutate_adapter_config(path: Path, changes: dict[str, Any]) -> None:
    config_path = path / "adapter_config.json"
    values = json.loads(config_path.read_text(encoding="utf-8"))
    values.update(changes)
    config_path.write_text(json.dumps(values), encoding="utf-8")


def _load(base: nn.Module, path: str | Path, **overrides: Any) -> Any:
    expected = {
        "expected_rank": 2,
        "expected_alpha": 4,
        "expected_dropout": 0.25,
        "expected_target_modules": ["q_proj", "v_proj"],
    }
    expected.update(overrides)
    return load_trainable_lora_adapter(base, str(path), **expected)


def test_local_adapter_load_accepts_target_order_only_difference(tmp_path: Path) -> None:
    source = peft.get_peft_model(
        _TinyModel(),
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.25,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    adapter_path = tmp_path / "adapter"
    source.save_pretrained(adapter_path)

    loaded = _load(
        _TinyModel(),
        adapter_path,
        expected_target_modules=["v_proj", "q_proj"],
    )

    config = loaded.peft_config["default"]
    assert config.r == 2
    assert config.lora_alpha == 4
    assert config.lora_dropout == 0.25
    assert config.target_modules == {"q_proj", "v_proj"}
    assert all(
        parameter.requires_grad for name, parameter in loaded.named_parameters() if "lora_" in name
    )


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"expected_rank": 3}, "rank saved=2 configured=3"),
        (
            {"expected_target_modules": ["q_proj"]},
            "target_modules saved=",
        ),
    ),
)
def test_topology_mismatch_fails_before_base_model_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    message: str,
) -> None:
    adapter_path = tmp_path / "adapter"
    _save_lora_config(adapter_path)
    load_calls = 0

    def unexpected_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("adapter weights must not load after a topology mismatch")

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", unexpected_load)
    base = _TinyModel()

    with pytest.raises(ValueError, match=message):
        _load(base, adapter_path, **override)

    assert load_calls == 0
    assert type(base.q_proj) is nn.Linear
    assert type(base.v_proj) is nn.Linear


@pytest.mark.parametrize(
    ("changes", "unsupported_field"),
    (
        ({"modules_to_save": ["head"]}, "modules_to_save"),
        ({"use_dora": True}, "use_dora"),
        ({"init_lora_weights": "olora"}, "init_lora_weights"),
    ),
)
def test_unsupported_saved_semantics_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    unsupported_field: str,
) -> None:
    adapter_path = tmp_path / "adapter"
    _save_lora_config(adapter_path)
    _mutate_adapter_config(adapter_path, changes)
    load_calls = 0

    def unexpected_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("unsupported adapter semantics must fail before loading weights")

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", unexpected_load)

    with pytest.raises(ValueError, match=unsupported_field):
        _load(_TinyModel(), adapter_path)

    assert load_calls == 0


def test_unknown_raw_adapter_config_field_fails_instead_of_following_peft_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path = tmp_path / "adapter"
    _save_lora_config(adapter_path)
    _mutate_adapter_config(adapter_path, {"future_semantic_mode": True})
    load_calls = 0

    def unexpected_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("unknown adapter semantics must fail before loading weights")

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", unexpected_load)

    with pytest.raises(ValueError, match="fields unsupported by the installed PEFT"):
        _load(_TinyModel(), adapter_path)

    assert load_calls == 0


@pytest.mark.parametrize("saved_selector", ("all-linear",))
def test_saved_target_selector_requires_an_explicit_canonical_target_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_selector: str,
) -> None:
    adapter_path = tmp_path / "adapter"
    _save_lora_config(adapter_path)
    _mutate_adapter_config(adapter_path, {"target_modules": saved_selector})
    load_calls = 0

    def unexpected_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("unsupported target selectors must fail before loading weights")

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", unexpected_load)

    with pytest.raises(ValueError, match="regex and all-linear selectors are unsupported"):
        _load(_TinyModel(), adapter_path)

    assert load_calls == 0


def test_saved_task_type_must_match_the_family_loader_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path = tmp_path / "adapter"
    _save_lora_config(adapter_path)
    _mutate_adapter_config(adapter_path, {"task_type": "CAUSAL_LM"})
    sentinel = object()
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda *_args, **_kwargs: sentinel,
    )

    assert (
        _load(
            _TinyModel(),
            adapter_path,
            expected_task_type="CAUSAL_LM",
        )
        is sentinel
    )

    with pytest.raises(
        ValueError,
        match=r"task_type saved='CAUSAL_LM' expected=None",
    ):
        _load(_TinyModel(), adapter_path)


def test_hf_adapter_config_uses_peft_hub_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path = tmp_path / "adapter"
    _save_lora_config(adapter_path)
    hub_calls: list[tuple[str, str]] = []
    sentinel = object()

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        **_kwargs: Any,
    ) -> str:
        hub_calls.append((repo_id, filename))
        return str(adapter_path / "adapter_config.json")

    def fake_from_pretrained(
        base: Any,
        model_id: str,
        **kwargs: Any,
    ) -> Any:
        assert isinstance(base, _TinyModel)
        assert model_id == "owner/adapter"
        assert type(kwargs["config"]) is peft.LoraConfig
        assert kwargs["is_trainable"] is True
        assert kwargs["adapter_name"] == "default"
        return sentinel

    monkeypatch.setattr("peft.config.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", fake_from_pretrained)

    loaded = _load(_TinyModel(), "owner/adapter")

    assert loaded is sentinel
    assert hub_calls == [("owner/adapter", "adapter_config.json")]


def test_non_lora_peft_adapter_is_rejected(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    peft.AdaLoraConfig(
        target_modules=["q_proj", "v_proj"],
        total_step=10,
    ).save_pretrained(adapter_path)

    with pytest.raises(ValueError, match="AdaLoraConfig, not a PEFT LoRA adapter"):
        _load(_TinyModel(), adapter_path)
