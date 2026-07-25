from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from vrl.config.precision import RolePrecision
from vrl.families.registry import TokenFamilyBuild, get_model_family_entry
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

_LORA_ONLY_FAMILIES = (
    "janus_pro",
    "janus_pro_r1",
    "emu3",
    "glm_image",
    "llamagen",
)


class _TinyRuntimeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.language_model = nn.Linear(1, 1)

    def replay_forward(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> Any:
        return self.load_state_dict(state_dict, strict=False)

    @property
    def adapter_roots(self) -> dict[str, Any]:
        # Mirrors ARModelBase: the checkpoint root is the wrapper, the adapter
        # is one hop in on language_model.
        return {"model": self.language_model}


class _JanusReplayCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Linear(2, 2)
        self.gen_head = nn.Linear(2, 4)
        self.gen_embed = nn.Embedding(4, 2)

    def prepare_gen_img_embeds(self, image_ids: torch.Tensor) -> torch.Tensor:
        return self.gen_embed(image_ids)


class _NextStepReplayCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(2, 2)
        self.image_head = nn.Linear(2, 2)
        self.image_head.input_dim = 2
        self.image_in_projector = nn.Linear(2, 2)


def _model_build(
    family: str,
    *,
    use_lora: bool,
    for_rollout: bool,
) -> ModelBuild:
    recipe = get_model_family_entry(family).family_build
    assert isinstance(recipe, TokenFamilyBuild)
    return ModelBuild(
        model_name_or_path=recipe.default_model_path,
        revision=None,
        device="cpu",
        parameter_dtype="fp32",
        family=family,
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        model_config={"use_lora": use_lora},
        sampling_config={},
        rollout=(RolloutBuildOptions(prompt_encoder_dtype="fp32") if for_rollout else None),
    )


def _install_fake_family_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: str,
    replay: bool,
) -> list[str]:
    recipe = get_model_family_entry(family).family_build
    assert isinstance(recipe, TokenFamilyBuild)
    model_path = recipe.replay_cls if replay else recipe.model_cls
    imported_paths: list[str] = []

    def import_test_object(path: str) -> Callable[..., Any]:
        imported_paths.append(path)
        if path == recipe.config_builder:
            return lambda _build: {}
        if path == recipe.config_cls:
            return lambda **_kwargs: object()
        if path == model_path:
            return lambda _config: _TinyRuntimeModel()
        raise AssertionError(f"unexpected import path: {path}")

    monkeypatch.setattr("vrl.utils.config.import_from_path", import_test_object)
    return imported_paths


@pytest.mark.parametrize(
    ("family", "expected"),
    (
        ("janus_pro", False),
        ("janus_pro_r1", False),
        ("nextstep_1", True),
        ("emu3", False),
        ("glm_image", False),
        ("llamagen", False),
    ),
)
def test_token_family_describes_full_parameter_training_capability(
    family: str,
    expected: bool,
) -> None:
    recipe = get_model_family_entry(family).family_build

    assert isinstance(recipe, TokenFamilyBuild)
    assert recipe.supports_full_parameter_training is expected


@pytest.mark.parametrize("family", _LORA_ONLY_FAMILIES)
def test_lora_only_replay_rejects_full_parameter_training_before_family_import(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_paths: list[str] = []
    monkeypatch.setattr(
        "vrl.utils.config.import_from_path",
        lambda path: imported_paths.append(path),
    )

    with pytest.raises(
        RuntimeError,
        match="does not support full-parameter training",
    ):
        get_model_family_entry(family).build_replay(
            _model_build(family, use_lora=False, for_rollout=False),
        )

    assert imported_paths == []


@pytest.mark.parametrize("family", _LORA_ONLY_FAMILIES)
def test_lora_only_family_still_allows_base_rollout(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_paths = _install_fake_family_imports(
        monkeypatch,
        family=family,
        replay=False,
    )
    entry = get_model_family_entry(family)
    recipe = entry.family_build
    assert isinstance(recipe, TokenFamilyBuild)

    bundle = entry.build_rollout(
        _model_build(family, use_lora=False, for_rollout=True),
    )

    assert bundle.raw_handle is bundle.model
    assert imported_paths == [
        recipe.config_builder,
        recipe.config_cls,
        recipe.model_cls,
    ]


def test_nextstep_full_parameter_replay_builds_a_trainable_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_paths = _install_fake_family_imports(
        monkeypatch,
        family="nextstep_1",
        replay=True,
    )
    entry = get_model_family_entry("nextstep_1")
    recipe = entry.family_build
    assert isinstance(recipe, TokenFamilyBuild)

    bundle = entry.build_replay(
        _model_build("nextstep_1", use_lora=False, for_rollout=False),
    )

    assert any(parameter.requires_grad for parameter in bundle.model.parameters())
    assert imported_paths == [
        recipe.config_builder,
        recipe.config_cls,
        recipe.replay_cls,
    ]


@pytest.mark.parametrize(
    ("build_replay", "for_rollout", "message"),
    (
        (True, True, "replay runtime construction"),
        (False, False, "rollout runtime construction"),
    ),
)
def test_token_role_mismatch_fails_before_family_import(
    build_replay: bool,
    for_rollout: bool,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_paths: list[str] = []
    monkeypatch.setattr(
        "vrl.utils.config.import_from_path",
        lambda path: imported_paths.append(path),
    )
    entry = get_model_family_entry("nextstep_1")
    build = _model_build(
        "nextstep_1",
        use_lora=False,
        for_rollout=for_rollout,
    )

    with pytest.raises(ValueError, match=message):
        (entry.build_replay if build_replay else entry.build_rollout)(build)

    assert imported_paths == []


def _janus_replay_without_lora() -> nn.Module:
    from vrl.models.families.janus_pro.config import JanusProConfig
    from vrl.models.families.janus_pro.model import JanusProReplayModel

    return JanusProReplayModel(
        JanusProConfig(use_lora=False, device="cpu", dtype="float32"),
        mmgpt=_JanusReplayCore(),
    )


def _nextstep_replay_without_lora() -> nn.Module:
    from vrl.models.families.nextstep_1.config import NextStep1Config
    from vrl.models.families.nextstep_1.model import NextStep1ReplayModel

    return NextStep1ReplayModel(
        NextStep1Config(use_lora=False, device="cpu", dtype="float32"),
        language_model=_NextStepReplayCore(),
    )


def _emu3_replay_without_lora() -> nn.Module:
    from tests.models.families.emu3.fixtures import build_tiny_emu3_replay_model

    return build_tiny_emu3_replay_model(use_lora=False)


def _glm_image_replay_without_lora() -> nn.Module:
    from tests.models.families.glm_image.fixtures import (
        build_tiny_glm_image_replay_model,
    )

    return build_tiny_glm_image_replay_model(use_lora=False)


def _llamagen_replay_without_lora() -> nn.Module:
    from tests.models.families.llamagen.fixtures import build_tiny_llamagen_model

    return build_tiny_llamagen_model(replay=True, use_lora=False)


@pytest.mark.parametrize(
    ("family", "build_model"),
    (
        ("janus_pro", _janus_replay_without_lora),
        ("nextstep_1", _nextstep_replay_without_lora),
        ("emu3", _emu3_replay_without_lora),
        ("glm_image", _glm_image_replay_without_lora),
        ("llamagen", _llamagen_replay_without_lora),
    ),
)
def test_full_parameter_capability_matches_replay_parameter_state(
    family: str,
    build_model: Callable[[], nn.Module],
) -> None:
    recipe = get_model_family_entry(family).family_build
    assert isinstance(recipe, TokenFamilyBuild)

    has_trainable_parameters = any(
        parameter.requires_grad for parameter in build_model().parameters()
    )

    assert has_trainable_parameters is recipe.supports_full_parameter_training
