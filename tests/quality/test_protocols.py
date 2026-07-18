from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.quality.contracts import QualityProtocol
from tests.quality.preflight import (
    ASSET_ROOT,
    family_profile_path,
    load_quality_profile,
)
from vrl.config.loading import list_bundled_configs, load_config
from vrl.families.registry import FAMILY_REGISTRY


def test_every_registry_entry_has_one_test_owned_profile() -> None:
    for entry in FAMILY_REGISTRY.values():
        assert family_profile_path(entry.family).is_file()
    profile_names = {path.stem for path in (ASSET_ROOT / "families").glob("*.yaml")}
    assert profile_names == set(FAMILY_REGISTRY)


def test_production_code_does_not_import_test_quality_support() -> None:
    project_root = Path(__file__).resolve().parents[2]
    violations: list[str] = []
    for path in (project_root / "vrl").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "vrl.quality" in source or "tests.quality" in source:
            violations.append(str(path.relative_to(project_root)))
    assert violations == []


def test_every_model_preset_matches_an_approved_identity() -> None:
    names = list_bundled_configs("model")
    assert names
    for name in names:
        profile, protocol = load_quality_profile(load_config(name))
        assert profile.model_identities
        assert protocol.name


def test_every_experiment_model_matches_an_approved_identity() -> None:
    matched = 0
    for name in list_bundled_configs("experiment"):
        cfg = load_config(name)
        if cfg.get("model") is None:
            continue
        load_quality_profile(cfg)
        matched += 1
    assert matched > 0


@pytest.mark.parametrize(
    ("config_name", "revision"),
    (
        ("model/emu3/gen_9b", "062069a3301d7c62c6c93a55c9722efb9321f5f3"),
        ("model/glm_image/9b", "2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4"),
        ("model/janus_pro/1b", "960ab33191f61342a4c60ae74d8dc356a39fafcb"),
        ("model/llamagen/xl_256", "276f5c5a3d915b922899a03f1912605531574747"),
        ("model/nextstep_1/1_1", "05486ff90769ad32109d219fd4659d41671a770a"),
    ),
)
def test_ar_model_presets_and_profiles_pin_the_same_revision(
    config_name: str,
    revision: str,
) -> None:
    cfg = load_config(config_name)
    profile, _ = load_quality_profile(cfg)

    assert cfg.model.revision == revision
    assert [(identity.path, identity.revision) for identity in profile.model_identities] == [
        (str(cfg.model.path), revision),
    ]


@pytest.mark.parametrize(
    ("config_name", "field", "revision"),
    (
        (
            "model/llamagen/xl_256",
            "t5_revision",
            "7d6315df2c2fb742f0f5b556879d730926ca9001",
        ),
        (
            "model/nextstep_1/1_1",
            "vae_revision",
            "31c24c8c751e87b02d56ae6a5fe658e7da0b8388",
        ),
        (
            "model/echo/release",
            "gemma_revision",
            "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        ),
        (
            "model/cosmos/anima_preview3",
            "qwen_tokenizer_revision",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
        ),
        (
            "model/cosmos/anima_preview3",
            "t5_tokenizer_revision",
            "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
        ),
    ),
)
def test_external_model_dependencies_are_immutably_pinned(
    config_name: str,
    field: str,
    revision: str,
) -> None:
    assert load_config(config_name).model[field] == revision


def test_explicit_janus_r1_family_uses_its_multisegment_profile() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "janus_pro_r1",
                "path": "deepseek-ai/Janus-Pro-1B",
                "revision": "960ab33191f61342a4c60ae74d8dc356a39fafcb",
            },
            "algorithm": {"kind": "token_grpo_multisegment"},
        },
    )

    profile, protocol = load_quality_profile(cfg)

    assert cfg.model.family == "janus_pro_r1"
    assert profile.extends == "multisegment_token_image.yaml"
    assert profile.model_identities[0].revision == cfg.model.revision
    assert protocol.required_segments == ("initial_image", "selfcheck", "final_image")


def test_unknown_model_identity_fails_closed() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "sana", "path": "someone/unknown-sana"},
            "algorithm": {"kind": "grpo"},
        },
    )

    with pytest.raises(ValueError, match="not approved by the test fixture"):
        load_quality_profile(cfg)


def test_protocol_rejects_unknown_fields_from_typed_source_of_truth() -> None:
    _, valid = load_quality_profile(
        OmegaConf.create(
            {
                "model": {
                    "family": "sana",
                    "path": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                    "revision": "d1b54936033cd7d45410ecadd692c5c502a19a38",
                },
                "algorithm": {"kind": "grpo"},
            },
        ),
    )
    raw = asdict(valid)
    raw["typo_threshold"] = 1

    with pytest.raises(ValueError, match="unknown quality protocol field"):
        QualityProtocol.from_mapping(raw)
