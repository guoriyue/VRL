"""Family-scoped model sections: which keys each family owns, how aliases and
sibling families select their section class, and the per-family runtime
capability contract (executor / memory sections)."""

from __future__ import annotations

from dataclasses import fields

import pytest
from omegaconf import OmegaConf

from tests.config.helpers import unknown_keys
from vrl.config.model_schema import (
    LoraSection,
    ModelExecutorSection,
    ModelMemorySection,
    ModelSection,
    TorchCompileSection,
    VaeDecodeMemorySection,
)
from vrl.config.schema import (
    parse_config,
)
from vrl.models.families.causvid.config import CausVidModelSection
from vrl.models.families.cosmos.anima.config import CosmosAnimaModelSection
from vrl.models.families.cosmos.predict2_5.config import (
    CosmosPredict25ModelSection,
)
from vrl.models.families.echo.config import EchoModelSection
from vrl.models.families.flux.config import FluxModelSection
from vrl.models.families.magi_1.config import Magi1ModelSection
from vrl.models.families.names import _FAMILY_BY_ALIAS
from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR,
    SHARED_MODEL_SECTION_CLS,
    get_model_family_entry,
)
from vrl.models.families.wan_2_1.config import WanModelSection
from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# This independent fixture pins the public capability contract for every
# canonical family. Production derives executor support from its binding and
# records memory support as target-section names; the test intentionally does
# not derive either expected value from those production fields.
_SHARED_LOADER_FAMILIES = frozenset(
    {
        "sd3_5",
        "flux",
        "qwen_image",
        "sana",
        "lumina2",
        "hunyuan_video",
        "mochi",
        "hunyuan_image",
        "pixart_sigma",
        "cogvideox",
    }
)

_MODEL_RUNTIME_CAPABILITY_MATRIX = {
    "sd3_5": (True, True),
    "causvid": (False, False),
    "magi_1": (False, False),
    "flux": (True, True),
    "qwen_image": (True, True),
    "sana": (True, True),
    "lumina2": (True, True),
    "hunyuan_video": (True, True),
    "mochi": (True, True),
    "hunyuan_image": (True, True),
    "pixart_sigma": (True, True),
    "cogvideox": (True, True),
    "wan_2_1": (True, True),
    "wan_2_1_i2v": (False, True),
    "cosmos-predict2": (False, True),
    "cosmos-predict2.5": (False, True),
    "cosmos3": (False, True),
    "minimax_h3": (False, True),
    "vdn_h3": (False, True),
    "cosmos-predict2-anima": (True, True),
    "echo": (False, False),
}


# ── distributed.training strategy ─────────────────────────────────────────────


# ── model family scoped keys ──────────────────────────────────────────────────


def test_wan_model_keys_are_scoped_to_wan_family() -> None:
    """Wan trainable-topology/offload keys are accepted only for Wan families."""

    wan_cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert unknown_keys(wan_cfg) == []

    alias_cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert unknown_keys(alias_cfg) == []

    sd3_cfg = OmegaConf.create(
        {
            "model": {
                "family": "sd3_5",
                "path": "stabilityai/stable-diffusion-3.5-medium",
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert unknown_keys(sd3_cfg) == [
        "model.boundary_ratio",
        "model.offload_mode",
        "model.trainable_transformers",
    ]


def test_wan_boundary_ratio_is_source_derived_not_public_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "boundary_ratio": 0.9,
            },
        },
    )

    assert unknown_keys(cfg) == ["model.boundary_ratio"]


@pytest.mark.parametrize(
    ("family", "field"),
    [
        ("sana", "nft_previous_adapter"),
        ("cosmos-predict2", "skip_text_encoder"),
    ],
)
def test_family_owned_keys_are_unknown_for_sibling_families(family: str, field: str) -> None:
    """A key declared by one family's section is a typo for every other family."""

    cfg = OmegaConf.create({"model": {"family": family, field: True}})

    assert unknown_keys(cfg) == [f"model.{field}"]


def test_unknown_wan_offload_mode_raises() -> None:
    """Wan offload mode is a typed three-state enum, not two independent bools."""
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "offload_mode": "stream",
            },
        },
    )
    with pytest.raises(ValueError, match=r"unknown model\.offload_mode"):
        parse_config(cfg)


def test_root_retains_selected_family_model_section_and_serializes_its_fields() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "expert_lifecycle_profiling": True,
                "offload_mode": "sequential",
            },
        },
    )

    parsed = parse_config(cfg)

    assert isinstance(parsed.model, WanModelSection)
    assert parsed.model.expert_lifecycle_profiling is True
    assert parsed.model_dump()["model"]["offload_mode"] == "sequential"


def test_cosmos_predict25_keys_select_family_section() -> None:
    cfg = OmegaConf.create(
        {"model": {"family": "cosmos-predict2.5", "use_lora": True, "skip_text_encoder": True}},
    )

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, CosmosPredict25ModelSection)
    assert parsed.model.skip_text_encoder is True


def test_cosmos_anima_keys_select_family_section() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2-anima",
                "qwen_tokenizer_path": "Qwen/Qwen2.5-0.5B",
                "scheduler_shift": 3.0,
                "transformer_file": "split_files/diffusion_models/anima.safetensors",
            },
        },
    )

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, CosmosAnimaModelSection)
    assert parsed.model.scheduler_shift == 3.0
    assert parsed.model.transformer_file == "split_files/diffusion_models/anima.safetensors"


@pytest.mark.parametrize(
    ("family", "section_cls", "payload"),
    [
        ("flux", FluxModelSection, {"use_lora": True, "lora": {"dropout": 0.1}}),
        (
            "echo",
            EchoModelSection,
            {
                "gemma_path": "google/gemma-3-12b-it",
                "gemma_revision": "revision",
                "video_height": 256,
                "video_width": 256,
            },
        ),
        (
            "causvid",
            CausVidModelSection,
            {
                "accept_noncommercial_license": True,
                "base_model_path": "Wan-AI/Wan2.1-T2V-1.3B",
                "checkpoint_file": "autoregressive_checkpoint/model.pt",
            },
        ),
        (
            "magi_1",
            Magi1ModelSection,
            {
                "python_executable": "third_party/MAGI-1/.venv/bin/python",
                "source_path": "third_party/MAGI-1",
                "timeout_seconds": 3600,
            },
        ),
    ],
)
def test_family_owned_denoise_keys_select_their_public_sections(
    family: str,
    section_cls: type[ModelSection],
    payload: dict[str, object],
) -> None:

    cfg = OmegaConf.create({"model": {"family": family, **payload}})

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert type(parsed.model) is section_cls
    assert parsed.model is not None
    parsed_payload = parsed.model.model_dump(exclude_unset=True)
    assert {key: parsed_payload[key] for key in payload} == payload


def test_model_family_aliases_select_their_canonical_section_classes() -> None:
    for alias, family in _FAMILY_BY_ALIAS.items():
        canonical = parse_config(
            OmegaConf.create(
                {"model": {"family": family, "use_lora": family == "cosmos-predict2.5"}}
            ),
        )
        parsed_alias = parse_config(
            OmegaConf.create(
                {"model": {"family": alias, "use_lora": family == "cosmos-predict2.5"}}
            ),
        )

        assert type(parsed_alias.model) is type(canonical.model)
        assert parsed_alias.model.family == alias


def test_model_runtime_capability_matrix_covers_every_registered_family() -> None:
    assert set(_MODEL_RUNTIME_CAPABILITY_MATRIX) == set(FAMILY_REGISTRY)

    for family, (supports_executor, supports_memory) in _MODEL_RUNTIME_CAPABILITY_MATRIX.items():
        entry = get_model_family_entry(family)

        assert (entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR) is supports_executor
        sections = entry.runtime_capabilities.supported_model_memory_sections
        assert ("vae_decode" in sections) is supports_memory
        # Host placement of frozen components is the shared diffusers loader's;
        # families with their own from_build do not advertise it.
        assert ("cpu_resident" in sections) is (family in _SHARED_LOADER_FAMILIES)


def test_shared_nested_model_sections_preserve_explicit_falsy_presence() -> None:
    raw_model = {
        "family": "flux",
        "lora": {
            "rank": 8,
            "alpha": 16,
            "path": None,
            "target_modules": [],
            "init_lora_weights": False,
            "dropout": 0.0,
        },
        "memory": {
            "vae_decode": {
                "tiling": False,
                "slicing": None,
            },
        },
        "torch_compile": {
            "enable": False,
            "mode": None,
        },
        "executor": {
            "num_frames": 1,
            "max_sequence_length": 128,
            "fps": None,
            "batch_passthrough_keys": [],
        },
    }

    parsed = parse_config(OmegaConf.create({"model": raw_model}))

    assert isinstance(parsed.model, ModelSection)
    assert isinstance(parsed.model.lora, LoraSection)
    assert isinstance(parsed.model.memory, ModelMemorySection)
    assert isinstance(parsed.model.memory.vae_decode, VaeDecodeMemorySection)
    assert isinstance(parsed.model.torch_compile, TorchCompileSection)
    assert isinstance(parsed.model.executor, ModelExecutorSection)
    assert parsed.model.model_dump(exclude_unset=True) == raw_model


def test_generation_memory_schema_and_policy_share_one_field_vocabulary() -> None:
    assert tuple(ModelMemorySection.model_fields) == tuple(
        policy_field.name for policy_field in fields(GenerationMemoryPolicy)
    )
    assert tuple(VaeDecodeMemorySection.model_fields) == tuple(
        policy_field.name for policy_field in fields(VaeDecodeMemory)
    )


@pytest.mark.parametrize(
    ("model_fragment", "path"),
    [
        ({"lora": {"rnak": 16}}, "model.lora.rnak"),
        (
            {"memory": {"vae_decode": {"tileing": True}}},
            "model.memory.vae_decode.tileing",
        ),
    ],
)
def test_model_subtree_typos_are_named_with_complete_paths(
    model_fragment: dict[str, object],
    path: str,
) -> None:
    cfg = OmegaConf.create({"model": {"family": "flux", **model_fragment}})

    assert unknown_keys(cfg) == [path]


def _parse_error(model: dict[str, object]) -> str | None:
    try:
        parse_config(OmegaConf.create({"model": model}))
    except ValueError as error:
        return str(error)
    return None


def test_model_runtime_sections_follow_family_capabilities() -> None:
    """``model.executor`` / ``model.memory`` parse only for families that consume them."""

    for family, (supports_executor, supports_memory) in _MODEL_RUNTIME_CAPABILITY_MATRIX.items():
        executor_error = _parse_error(
            {
                "family": family,
                "use_lora": family == "cosmos-predict2.5",
                "executor": {"max_sequence_length": 123},
            },
        )
        memory_error = _parse_error(
            {
                "family": family,
                "use_lora": family == "cosmos-predict2.5",
                "memory": {"vae_decode": {"tiling": True}},
            },
        )
        if supports_executor:
            assert executor_error is None, family
        else:
            assert executor_error == f"model family {family!r} does not support model.executor"
        if supports_memory:
            assert memory_error is None, family
        else:
            assert memory_error == (
                f"model family {family!r} does not support model.memory section(s): vae_decode"
            )


@pytest.mark.parametrize("empty_value", [None, {}])
def test_empty_model_runtime_sections_are_valid_for_every_family(
    empty_value: object,
) -> None:
    for family in _MODEL_RUNTIME_CAPABILITY_MATRIX:
        parsed = parse_config(
            OmegaConf.create(
                {
                    "model": {
                        "family": family,
                        "use_lora": family == "cosmos-predict2.5",
                        "executor": empty_value,
                        "memory": empty_value,
                    },
                },
            ),
        )

        assert parsed.model is not None


def test_shared_only_families_use_the_shared_model_section() -> None:
    shared_families = [
        entry.family
        for entry in FAMILY_REGISTRY.values()
        if entry.model_section_cls == SHARED_MODEL_SECTION_CLS
    ]
    assert shared_families

    for family in shared_families:
        parsed = parse_config(
            OmegaConf.create({"model": {"family": family, "path": f"org/{family}"}}),
        )

        assert type(parsed.model) is ModelSection
        assert parsed.model.path == f"org/{family}"


def test_unknown_model_family_fails_at_typed_parse() -> None:
    cfg = OmegaConf.create({"model": {"family": "not_a_family", "path": "org/model"}})

    with pytest.raises(ValueError, match=r"unsupported model family: 'not_a_family'"):
        parse_config(cfg)


def test_present_model_section_requires_a_family() -> None:
    cfg = OmegaConf.create({"model": {"path": "org/model"}})

    with pytest.raises(ValueError, match=r"config missing required field: model\.family"):
        parse_config(cfg)
