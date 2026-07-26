"""CPU contracts for the AR Hugging Face checkpoint adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.steps.token import loader
from vrl.models.steps.token.base import ARReplayCore


class _TinyReplayCore(ARReplayCore):
    checkpoint_subfolder = "core"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.left = nn.Linear(1, 1, bias=False)
        self.right = nn.Linear(1, 1, bias=False)
        self.loaded_state_keys: list[set[str]] = []

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        self.loaded_state_keys.append(set(state_dict))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
def test_single_file_checkpoint_loads_only_core_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
) -> None:
    from transformers import modeling_utils

    checkpoint_path = tmp_path / filename
    checkpoint_path.touch()
    calls: list[str] = []

    def fake_load_state_dict(path: str) -> dict[str, torch.Tensor]:
        calls.append(path)
        return {
            "left.weight": torch.tensor([[1.0]]),
            "right.weight": torch.tensor([[2.0]]),
            "vision_tower.weight": torch.tensor([[3.0]]),
        }

    monkeypatch.setattr(modeling_utils, "load_state_dict", fake_load_state_dict)
    core = _TinyReplayCore()

    loader.load_ar_replay_checkpoint(core, str(tmp_path))

    assert calls == [str(checkpoint_path)]
    assert core.loaded_state_keys == [{"left.weight", "right.weight"}]
    torch.testing.assert_close(core.left.weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(core.right.weight, torch.tensor([[2.0]]))


@pytest.mark.parametrize(
    ("index_filename", "extension"),
    [
        ("model.safetensors.index.json", "safetensors"),
        ("pytorch_model.bin.index.json", "bin"),
    ],
)
def test_sharded_checkpoint_opens_only_shards_with_core_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    index_filename: str,
    extension: str,
) -> None:
    from transformers import modeling_utils

    left_shard = f"core-00001-of-00003.{extension}"
    right_shard = f"core-00002-of-00003.{extension}"
    ignored_shard = f"vision-00003-of-00003.{extension}"
    weight_map = {
        "left.weight": left_shard,
        "right.weight": right_shard,
        "vision_tower.weight": ignored_shard,
    }
    (tmp_path / index_filename).write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    shard_states = {
        left_shard: {
            "left.weight": torch.tensor([[4.0]]),
            "vision_tower.extra": torch.tensor([[40.0]]),
        },
        right_shard: {
            "right.weight": torch.tensor([[5.0]]),
            "vision_tower.extra": torch.tensor([[50.0]]),
        },
    }
    calls: list[str] = []

    def fake_load_state_dict(path: str) -> dict[str, torch.Tensor]:
        filename = Path(path).name
        calls.append(filename)
        return shard_states[filename]

    monkeypatch.setattr(modeling_utils, "load_state_dict", fake_load_state_dict)
    core = _TinyReplayCore()

    loader.load_ar_replay_checkpoint(core, str(tmp_path))

    assert calls == [left_shard, right_shard]
    assert ignored_shard not in calls
    assert core.loaded_state_keys == [{"left.weight"}, {"right.weight"}]
    torch.testing.assert_close(core.left.weight, torch.tensor([[4.0]]))
    torch.testing.assert_close(core.right.weight, torch.tensor([[5.0]]))


def test_checkpoint_missing_core_keys_fails_before_loading_partial_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import modeling_utils

    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(
        modeling_utils,
        "load_state_dict",
        lambda _path: {"left.weight": torch.tensor([[1.0]])},
    )
    core = _TinyReplayCore()

    with pytest.raises(RuntimeError, match=r"missing keys: right\.weight"):
        loader.load_ar_replay_checkpoint(core, str(tmp_path))

    assert core.loaded_state_keys == []


def test_sharded_index_missing_core_key_fails_before_opening_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import modeling_utils

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"left.weight": "core.safetensors"}}),
        encoding="utf-8",
    )
    opened: list[str] = []
    monkeypatch.setattr(
        modeling_utils,
        "load_state_dict",
        lambda path: opened.append(path),
    )

    with pytest.raises(RuntimeError, match=r"missing keys: right\.weight"):
        loader.load_ar_replay_checkpoint(_TinyReplayCore(), str(tmp_path))

    assert opened == []


def test_checkpoint_missing_supported_file_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match=r"No supported _TinyReplayCore checkpoint file",
    ):
        loader.load_ar_replay_checkpoint(_TinyReplayCore(), str(tmp_path))


def test_replay_core_from_pretrained_uses_checkpoint_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoConfig

    calls: dict[str, Any] = {}

    def fake_config(model_path: str, **kwargs: Any) -> object:
        calls["config"] = (model_path, kwargs)
        return object()

    def fake_resolve(
        model_path: str,
        *,
        subfolder: str | None,
        revision: str | None,
    ) -> str:
        calls["resolve"] = (model_path, subfolder, revision)
        return "/cache/revision/core"

    def fake_load(module: nn.Module, checkpoint_dir: str) -> None:
        calls["load"] = (module, checkpoint_dir)

    monkeypatch.setattr(AutoConfig, "from_pretrained", staticmethod(fake_config))
    monkeypatch.setattr(loader, "resolve_hf_checkpoint_dir", fake_resolve)
    monkeypatch.setattr(loader, "load_ar_replay_checkpoint", fake_load)

    core = _TinyReplayCore.from_pretrained(
        "org/model",
        device="cpu",
        dtype=torch.float64,
        revision="immutable",
        trust_remote_code=True,
    )

    assert calls["config"] == (
        "org/model",
        {
            "revision": "immutable",
            "trust_remote_code": True,
            "subfolder": "core",
        },
    )
    assert calls["resolve"] == ("org/model", "core", "immutable")
    assert calls["load"] == (core, "/cache/revision/core")
    assert core.left.weight.dtype is torch.float64
    assert core.training is False
