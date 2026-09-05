"""Tests for the typed config schema boundary (vrl/config/schema.py)."""

from __future__ import annotations

import typing
from dataclasses import fields

import pytest
from omegaconf import OmegaConf

from vrl.config.model_schema import (
    LoraSection,
    ModelExecutorSection,
    ModelMemorySection,
    ModelSection,
    TorchCompileSection,
    VaeDecodeMemorySection,
)
from vrl.config.sampling_schema import (
    ARSamplingSection,
    DenoiseImageSamplingSection,
    EchoSamplingSection,
    JanusProSamplingSection,
    TextEncodedImageSamplingSection,
    VideoSamplingSection,
)
from vrl.config.schema import (
    AlgorithmConfig,
    DataConfig,
    RewardConfig,
    RolloutRuntimeSection,
    RootConfig,
    parse_config,
)
from vrl.models.families.causvid.config import CausVidModelSection
from vrl.models.families.cosmos.anima.config import CosmosAnimaModelSection
from vrl.models.families.cosmos.predict2_5.config import (
    CosmosPredict25ModelSection,
)
from vrl.models.families.echo.config import EchoModelSection
from vrl.models.families.flux.config import FluxModelSection
from vrl.models.families.janus_pro.config import JanusProModelSection
from vrl.models.families.llamagen.config import LlamaGenModelSection
from vrl.models.families.magi_1.config import Magi1ModelSection
from vrl.models.families.names import _FAMILY_BY_ALIAS
from vrl.models.families.nextstep_1.config import NextStep1ModelSection
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
    "cosmos-predict2-anima": (True, True),
    "echo": (False, False),
    "janus_pro": (False, False),
    "janus_pro_r1": (False, False),
    "nextstep_1": (False, False),
    "emu3": (False, False),
    "glm_image": (False, False),
    "llamagen": (False, False),
}


def _literal_args(annotation) -> tuple[str, ...]:
    """Flatten a ``Literal[...]`` or ``Literal[...] | None`` annotation into its
    members, so the allow-list tests derive their cases from the schema's single
    source of truth instead of hand-copying the Literal members (a copy never
    sees a newly added member, leaving it silently untested)."""
    members = typing.get_args(annotation)
    # Optional[Literal[...]]: unwrap each non-None union member's Literal args.
    if any(m is type(None) for m in members):
        return tuple(a for m in members if m is not type(None) for a in typing.get_args(m))
    return members


def _minimal_grpo_cfg(**overrides):
    base = {
        "algorithm": {"kind": "grpo"},
        "data": {
            "loader": "prompt_manifest",
            "manifest": "datasets/ocr/train.txt",
            "preprocessing": {"format": "text"},
            "sampler": {"type": "random_without_replacement"},
        },
        "rollout": {"sde": {"type": "flow_grpo"}},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _kling_video_reward_kwargs(**overrides) -> dict:
    base = {
        "sleep_offload": True,
        "reward_name": "org/model@main",
        "score_key": "overall",
        "worker_config": {"model_path": "/tmp/model"},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("value", [None, 1, 8])
def test_sampling_scheduler_batch_size_accepts_null_or_positive_integer(
    value: int | None,
) -> None:
    sampling = ARSamplingSection.model_validate({"ar_scheduler_batch_size": value})

    assert sampling.ar_scheduler_batch_size == value


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 1.5, "2"])
def test_sampling_scheduler_batch_size_rejects_coercible_or_non_positive_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer or null"):
        ARSamplingSection.model_validate({"ar_scheduler_batch_size": value})


# ── Algorithm kind discriminator ──────────────────────────────────────────────


def test_unknown_algorithm_kind_raises() -> None:
    """Checks unknown algorithm kind raises."""
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kind = "qpo"
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        parse_config(cfg)


def test_unknown_algorithm_keys_warn_and_load() -> None:
    """Removed keys, typos, and never-seen keys all warn — none of them raise."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {"algorithm": {"kind": "grpo", "adv_estimator": "dpo", "future_field": True}}
    )
    algo = AlgorithmConfig.model_validate(OmegaConf.to_container(cfg.algorithm, resolve=True))
    assert algo.kind == "grpo"  # loads fine
    unknown = find_unknown_keys(cfg)
    assert "algorithm.adv_estimator" in unknown
    assert "algorithm.future_field" in unknown


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("grpo", "flow_kl_use_dt", True),
        ("dance_grpo", "sft_weight", 0.1),
        ("flow_dppo", "add_kl_coefficient", 0.2),
        ("grpo_guard", "clip_ratio", 0.2),
        ("token_grpo", "kl_estimator", "k1"),
        ("token_grpo_multisegment", "segment_weights", [1.0]),
        ("diffusion_dpo", "beta", 5000.0),
        ("diffusion_nft", "nft_beta", 0.1),
    ],
)
def test_algorithm_keys_derive_from_selected_runtime_config(
    kind: str,
    field: str,
    value: object,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"algorithm": {"kind": kind, field: value}})
    assert find_unknown_keys(cfg) == []


@pytest.mark.parametrize(
    ("kind", "foreign_field"),
    [("grpo", "beta"), ("diffusion_dpo", "clip_ratio")],
)
def test_algorithm_keys_are_scoped_to_selected_kind(kind: str, foreign_field: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"algorithm": {"kind": kind, foreign_field: 1}})
    assert find_unknown_keys(cfg) == [f"algorithm.{foreign_field}"]


def test_algorithm_unknown_key_selector_defers_invalid_kind_to_schema() -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"algorithm": {"kind": "qpo", "future_field": 1}})
    assert find_unknown_keys(cfg) == ["algorithm.future_field"]


def test_algorithm_dispatch_covers_schema_kind_vocabulary() -> None:
    from vrl.config.algorithm import algorithm_config_class

    kinds = _literal_args(AlgorithmConfig.model_fields["kind"].annotation)
    assert kinds
    assert all(algorithm_config_class(kind) for kind in kinds)


def test_positive_kl_reward_coef_is_accepted_for_diffusion_rollouts() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kl_reward_coef = 0.25

    assert parse_config(cfg).algorithm.kl_reward_coef == 0.25


@pytest.mark.parametrize(
    "kind",
    ["token_grpo", "token_grpo_multisegment", "diffusion_dpo"],
)
def test_positive_kl_reward_coef_rejects_trajectories_without_step_kl(
    kind: str,
) -> None:
    cfg = OmegaConf.create(
        {"algorithm": {"kind": kind, "kl_reward_coef": 0.25}},
    )

    with pytest.raises(
        ValueError,
        match=r"algorithm\.kl_reward_coef > 0 requires a diffusion rollout trajectory",
    ):
        parse_config(cfg)


def test_zero_kl_reward_coef_remains_valid_for_token_rollouts() -> None:
    cfg = OmegaConf.create(
        {"algorithm": {"kind": "token_grpo", "kl_reward_coef": 0.0}},
    )

    assert parse_config(cfg).algorithm.kl_reward_coef == 0.0


@pytest.mark.parametrize(
    "value",
    [-0.1, True, "0.25", float("inf"), float("-inf"), float("nan")],
)
def test_kl_reward_coef_rejects_invalid_public_values(value: object) -> None:
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kl_reward_coef = value

    with pytest.raises(
        ValueError,
        match=r"algorithm\.kl_reward_coef must be a finite number >= 0",
    ):
        parse_config(cfg)


# ── rollout / sampling string-setting Literals ────────────────────────────────


@pytest.mark.parametrize("mode", ["native", "sde"])
def test_valid_denoise_modes_accepted(mode: str) -> None:
    """Both denoise modes validate; the Literal is the user-facing allow-list."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.denoise_mode = mode
    assert parse_config(cfg).rollout.denoise_mode == mode


def test_unknown_denoise_mode_raises() -> None:
    """An out-of-set denoise_mode is rejected at parse with the dotted path."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.denoise_mode = "bogus"
    with pytest.raises(ValueError, match=r"unknown rollout\.denoise_mode"):
        parse_config(cfg)


def test_shared_attention_backend_omission_stays_unset() -> None:
    """The runtime fallback owns the default; typed config stores no duplicate."""
    cfg = _minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {}
    sampling = parse_config(cfg).sampling

    assert type(sampling) is JanusProSamplingSection
    assert sampling.attention_backend is None
    assert sampling.model_dump(exclude_unset=True) == {}


def test_shared_attention_family_accepts_explicit_backend() -> None:
    cfg = _minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {"attention_backend": "torch_native"}

    assert parse_config(cfg).sampling.attention_backend == "torch_native"


def test_unknown_attention_backend_raises() -> None:
    """An out-of-set attention_backend is rejected at parse with the dotted path."""
    cfg = _minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {"attention_backend": "bogus"}
    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


@pytest.mark.parametrize("family", ["glm_image", "llamagen"])
def test_native_cache_family_rejects_typed_attention_backend(family: str) -> None:
    cfg = _minimal_grpo_cfg(model={"family": family})
    cfg.sampling = {"attention_backend": "torch_native"}

    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_token_num", 256),
        ("image_size", 256),
        ("max_text_length", 120),
    ],
)
def test_llamagen_sampling_rejects_model_derived_topology(
    field: str,
    value: object,
) -> None:
    cfg = _minimal_grpo_cfg(
        model={"family": "llamagen"},
        sampling={field: value},
    )

    with pytest.raises(ValueError, match=rf"unknown sampling\.{field}"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "family",
    ["janus_pro", "nextstep_1", "emu3", "glm_image"],
)
def test_text_encoded_ar_families_keep_request_text_length(family: str) -> None:
    cfg = _minimal_grpo_cfg(
        model={"family": family},
        sampling={"max_text_length": 128},
    )

    sampling = parse_config(cfg).sampling

    assert sampling is not None
    assert sampling.max_text_length == 128


def test_sampling_schema_is_selected_from_model_family() -> None:
    cfg = _minimal_grpo_cfg(model={"family": "sana"})
    cfg.sampling = {"max_sequence_length": 300}

    sampling = parse_config(cfg).sampling

    assert type(sampling) is TextEncodedImageSamplingSection
    assert sampling.max_sequence_length == 300


def test_sampling_section_requires_model_family_for_schema_selection() -> None:
    cfg = _minimal_grpo_cfg(sampling={"num_steps": 10})

    with pytest.raises(ValueError, match=r"sampling requires model\.family"):
        parse_config(cfg)


@pytest.mark.parametrize(
    ("family", "expected_type"),
    [
        ("hunyuan_image", DenoiseImageSamplingSection),
        ("cosmos-predict2", VideoSamplingSection),
        ("cosmos3", VideoSamplingSection),
        ("echo", EchoSamplingSection),
    ],
)
def test_family_without_text_length_rejects_max_sequence_length(
    family: str,
    expected_type: type,
) -> None:
    cfg = _minimal_grpo_cfg(model={"family": family})
    cfg.sampling = {"max_sequence_length": 512}

    with pytest.raises(ValueError, match=r"unknown sampling\.max_sequence_length"):
        parse_config(cfg)

    parsed = parse_config(
        _minimal_grpo_cfg(model={"family": family}, sampling={}),
    )
    assert type(parsed.sampling) is expected_type


def test_echo_accepts_only_baked_guidance_value() -> None:
    valid = _minimal_grpo_cfg(model={"family": "echo"})
    valid.sampling = {"guidance_scale": 1.0}
    assert parse_config(valid).sampling.guidance_scale == 1.0

    invalid = _minimal_grpo_cfg(model={"family": "echo"})
    invalid.sampling = {"guidance_scale": 4.5}
    with pytest.raises(ValueError, match=r"unknown sampling\.guidance_scale=4\.5"):
        parse_config(invalid)


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("janus_pro", "max_reflect_len", 80),
        ("nextstep_1", "temperature", 0.9),
        ("emu3", "image_token_num", 256),
        ("glm_image", "guidance_scale", 4.5),
        ("magi_1", "guidance_scale", 4.5),
    ],
)
def test_sampling_fields_are_rejected_outside_their_behavior_owner(
    family: str,
    field: str,
    value: object,
) -> None:
    cfg = _minimal_grpo_cfg(model={"family": family}, sampling={field: value})

    with pytest.raises(ValueError, match=rf"unknown sampling\.{field}"):
        parse_config(cfg)


def test_reflection_length_is_owned_only_by_janus_r1() -> None:
    cfg = _minimal_grpo_cfg(
        model={"family": "janus_pro_r1"},
        sampling={"max_reflect_len": 80},
    )
    cfg.algorithm.kind = "token_grpo_multisegment"
    cfg.rollout.final_image_policy = "always_generate"

    assert parse_config(cfg).sampling.max_reflect_len == 80


def test_unknown_final_image_policy_raises() -> None:
    """final_image_policy is Literal-typed regardless of algorithm kind."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.final_image_policy = "bogus"
    with pytest.raises(ValueError, match=r"unknown rollout\.final_image_policy"):
        parse_config(cfg)


# ── distributed.training strategy ─────────────────────────────────────────────


def test_unknown_training_strategy_raises() -> None:
    """An unimplemented/typo strategy is rejected at parse time, not silently run."""
    cfg = _minimal_grpo_cfg(distributed={"training": {"strategy": "deepspeed"}})
    with pytest.raises(ValueError, match=r"unknown distributed\.training\.strategy"):
        parse_config(cfg)


def test_training_keys_are_registered_not_unknown() -> None:
    """distributed.training.* keys are known to the unknown-key walker."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "training": {
                "strategy": "fsdp",
                "num_nodes": 1,
                "gpus_per_node": 2,
            }
        }
    )
    unknown = find_unknown_keys(cfg)
    assert not [k for k in unknown if k.startswith("distributed.training")]


# ── model family scoped keys ──────────────────────────────────────────────────


def test_wan_model_keys_are_scoped_to_wan_family() -> None:
    """Wan trainable-topology/offload keys are accepted only for Wan families."""
    from vrl.config.unknown_keys import find_unknown_keys

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
    assert find_unknown_keys(wan_cfg) == []

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
    assert find_unknown_keys(alias_cfg) == []

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
    assert find_unknown_keys(sd3_cfg) == [
        "model.boundary_ratio",
        "model.offload_mode",
        "model.trainable_transformers",
    ]


@pytest.mark.parametrize("family", ["wan_2_1", "wan_2_1_i2v", "wan", "wan_i2v"])
def test_wan_boundary_ratio_is_source_derived_not_public_config(family: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {
            "model": {
                "family": family,
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "boundary_ratio": 0.9,
            },
        },
    )

    assert find_unknown_keys(cfg) == ["model.boundary_ratio"]
    with pytest.raises(ValueError, match=r"unknown model\.boundary_ratio"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "family",
    ["janus_pro", "janus", "janus_pro_r1", "janus_r1"],
)
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_janus_keys_and_aliases_select_shared_family_section(
    family: str,
    trust_remote_code: bool,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg_data: dict[str, object] = {
        "model": {
            "family": family,
            "trust_remote_code": trust_remote_code,
            "vq_latent_channels": 8,
        },
    }
    if family in {"janus_pro_r1", "janus_r1"}:
        cfg_data.update(
            {
                "algorithm": {"kind": "token_grpo_multisegment"},
                "rollout": {"final_image_policy": "always_generate"},
            },
        )
    cfg = OmegaConf.create(cfg_data)

    assert find_unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, JanusProModelSection)
    assert parsed.model.trust_remote_code is trust_remote_code
    assert parsed.model.vq_latent_channels == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [("trust_remote_code", False), ("vq_latent_channels", 8)],
)
def test_janus_keys_are_unknown_for_shared_token_sibling(
    field: str,
    value: object,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"model": {"family": "emu3", field: value}})

    assert find_unknown_keys(cfg) == [f"model.{field}"]


@pytest.mark.parametrize("family", ["nextstep_1", "nextstep"])
@pytest.mark.parametrize("freeze_vae", [False, True])
def test_nextstep_keys_and_alias_select_family_section(
    family: str,
    freeze_vae: bool,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    payload = {
        "freeze_vae": freeze_vae,
        "vae_path": "stepfun-ai/NextStep-1-f8ch16-Tokenizer",
        "vae_revision": "immutable",
    }
    cfg = OmegaConf.create({"model": {"family": family, **payload}})

    assert find_unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, NextStep1ModelSection)
    assert parsed.model.freeze_vae is freeze_vae
    assert parsed.model.vae_path == payload["vae_path"]
    assert parsed.model.vae_revision == payload["vae_revision"]


@pytest.mark.parametrize("field", ["freeze_vae", "vae_path", "vae_revision"])
def test_nextstep_keys_are_unknown_for_shared_token_sibling(field: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"model": {"family": "emu3", field: "unexpected"}})

    assert find_unknown_keys(cfg) == [f"model.{field}"]


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


@pytest.mark.parametrize("expert_lifecycle_profiling", [False, True])
def test_root_retains_selected_family_model_section_and_serializes_its_fields(
    expert_lifecycle_profiling: bool,
) -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "expert_lifecycle_profiling": expert_lifecycle_profiling,
                "offload_mode": "sequential",
            },
        },
    )

    parsed = parse_config(cfg)

    assert isinstance(parsed.model, WanModelSection)
    assert parsed.model.expert_lifecycle_profiling is expert_lifecycle_profiling
    assert parsed.model.offload_mode == "sequential"
    assert parsed.model_dump()["model"]["expert_lifecycle_profiling"] is expert_lifecycle_profiling
    assert parsed.model_dump()["model"]["offload_mode"] == "sequential"


@pytest.mark.parametrize("family", ["cosmos-predict2.5", "cosmos_predict2_5"])
@pytest.mark.parametrize("skip_text_encoder", [False, True])
def test_cosmos_predict25_keys_and_alias_select_family_section(
    family: str,
    skip_text_encoder: bool,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {
            "model": {
                "family": family,
                "skip_text_encoder": skip_text_encoder,
            },
        },
    )

    assert find_unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, CosmosPredict25ModelSection)
    assert parsed.model.skip_text_encoder is skip_text_encoder
    assert parsed.model_dump()["model"]["skip_text_encoder"] is skip_text_encoder


def test_cosmos_predict25_key_is_unknown_for_shared_predict2() -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2",
                "skip_text_encoder": True,
            },
        },
    )

    assert find_unknown_keys(cfg) == ["model.skip_text_encoder"]


@pytest.mark.parametrize(
    "family",
    ["cosmos-predict2-anima", "anima", "cosmos_anima"],
)
def test_cosmos_anima_keys_and_aliases_select_family_section(family: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {
            "model": {
                "family": family,
                "qwen_tokenizer_path": "Qwen/Qwen2.5-0.5B",
                "scheduler_shift": 3.0,
                "transformer_file": "split_files/diffusion_models/anima.safetensors",
            },
        },
    )

    assert find_unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, CosmosAnimaModelSection)
    assert parsed.model.scheduler_shift == 3.0
    assert parsed.model.transformer_file == "split_files/diffusion_models/anima.safetensors"


def test_cosmos_anima_key_is_unknown_for_shared_predict2() -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2",
                "scheduler_shift": 3.0,
            },
        },
    )

    assert find_unknown_keys(cfg) == ["model.scheduler_shift"]


@pytest.mark.parametrize("family", ["llamagen", "llamagen_t2i"])
def test_llamagen_keys_and_alias_select_family_section(family: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    payload = {
        "gpt_ckpt": "custom-gpt.pt",
        "gpt_model": "GPT-XL",
        "image_token_num": 256,
        "t5_path": "org/t5",
        "t5_revision": "immutable",
        "vq_ckpt": "custom-vq.pt",
    }
    cfg = OmegaConf.create({"model": {"family": family, **payload}})

    assert find_unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, LlamaGenModelSection)
    assert parsed.model is not None
    parsed_payload = parsed.model.model_dump()
    assert {key: parsed_payload[key] for key in payload} == payload


@pytest.mark.parametrize(
    "field",
    [
        "gpt_ckpt",
        "gpt_model",
        "image_token_num",
        "t5_path",
        "t5_revision",
        "vq_ckpt",
    ],
)
def test_llamagen_keys_are_unknown_for_shared_token_sibling(field: str) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"model": {"family": "emu3", field: "unexpected"}})

    assert find_unknown_keys(cfg) == [f"model.{field}"]


@pytest.mark.parametrize(
    ("family", "section_cls", "payload"),
    [
        ("flux", FluxModelSection, {"nft_previous_adapter": True}),
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
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"model": {"family": family, **payload}})

    assert find_unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert type(parsed.model) is section_cls
    assert parsed.model is not None
    parsed_payload = parsed.model.model_dump()
    assert {key: parsed_payload[key] for key in payload} == payload


@pytest.mark.parametrize("accepted", [False, True])
def test_causvid_license_acknowledgement_preserves_both_public_states(
    accepted: bool,
) -> None:
    parsed = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "causvid",
                    "accept_noncommercial_license": accepted,
                },
            },
        ),
    )

    assert isinstance(parsed.model, CausVidModelSection)
    assert parsed.model.accept_noncommercial_license is accepted


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nft_previous_adapter", True),
        ("gemma_path", "google/gemma-3-12b-it"),
        ("accept_noncommercial_license", True),
        ("source_path", "third_party/MAGI-1"),
    ],
)
def test_family_owned_denoise_keys_are_unknown_for_shared_sibling(
    field: str,
    value: object,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create({"model": {"family": "sd3_5", field: value}})

    assert find_unknown_keys(cfg) == [f"model.{field}"]


def test_model_family_aliases_select_their_canonical_section_classes() -> None:
    for alias, family in _FAMILY_BY_ALIAS.items():
        canonical = parse_config(
            OmegaConf.create({"model": {"family": family}}),
        )
        parsed_alias = parse_config(
            OmegaConf.create({"model": {"family": alias}}),
        )

        assert type(parsed_alias.model) is type(canonical.model)
        assert parsed_alias.model.family == alias


def test_model_runtime_capability_matrix_covers_every_registered_family() -> None:
    assert set(_MODEL_RUNTIME_CAPABILITY_MATRIX) == set(FAMILY_REGISTRY)

    for family, (supports_executor, supports_memory) in _MODEL_RUNTIME_CAPABILITY_MATRIX.items():
        entry = get_model_family_entry(family)

        assert (entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR) is supports_executor
        assert entry.runtime_capabilities.supported_model_memory_sections == (
            frozenset({"vae_decode"}) if supports_memory else frozenset()
        )


def test_shared_nested_model_sections_preserve_explicit_falsy_presence() -> None:
    raw_model = {
        "family": "flux",
        "lora": {
            "rank": 0,
            "alpha": 0,
            "path": None,
            "target_modules": [],
            "init_lora_weights": False,
            "dropout": 0.0,
            "init": None,
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
            "num_frames": 0,
            "max_sequence_length": 0,
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

    # Re-selecting a family subclass from an already-typed shared section must
    # not materialize omitted defaults into the runtime projection.
    reparsed = RootConfig.model_validate(
        {"model": ModelSection.model_validate(raw_model)},
    )
    assert reparsed.model is not None
    assert reparsed.model.model_dump(exclude_unset=True) == raw_model


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
        ({"memory": {"transformer": {}}}, "model.memory.transformer"),
        ({"torch_compile": {"enabled": True}}, "model.torch_compile.enabled"),
        (
            {"executor": {"max_seq_length": 300}},
            "model.executor.max_seq_length",
        ),
        ({"pth": "org/model"}, "model.pth"),
    ],
)
def test_model_subtree_typos_fail_closed_with_complete_paths(
    model_fragment: dict[str, object],
    path: str,
) -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = OmegaConf.create(
        {
            "model": {
                "family": "flux",
                **model_fragment,
            },
        },
    )

    assert find_unknown_keys(cfg) == [path]
    with pytest.raises(ValueError) as error:
        parse_config(cfg)
    assert str(error.value) == f"unknown {path}"


@pytest.mark.parametrize(
    ("family", "supports_executor", "supports_memory"),
    [
        (family, supports_executor, supports_memory)
        for family, (supports_executor, supports_memory) in sorted(
            _MODEL_RUNTIME_CAPABILITY_MATRIX.items(),
        )
    ],
)
def test_model_runtime_sections_follow_family_capabilities(
    family: str,
    supports_executor: bool,
    supports_memory: bool,
) -> None:
    executor_cfg = OmegaConf.create(
        {
            "model": {
                "family": family,
                "executor": {"max_sequence_length": 123},
            },
        },
    )
    if supports_executor:
        parsed = parse_config(executor_cfg)
        assert parsed.model is not None
        assert parsed.model.executor is not None
        assert parsed.model.executor.model_dump(exclude_unset=True) == {
            "max_sequence_length": 123,
        }
    else:
        with pytest.raises(
            ValueError,
            match=rf"model family {family!r} does not support model\.executor",
        ):
            parse_config(executor_cfg)

    for tiling in (False, True):
        memory_cfg = OmegaConf.create(
            {
                "model": {
                    "family": family,
                    "memory": {"vae_decode": {"tiling": tiling}},
                },
            },
        )
        if supports_memory:
            parsed = parse_config(memory_cfg)
            assert parsed.model is not None
            assert parsed.model.memory is not None
            assert parsed.model.memory.model_dump(exclude_unset=True) == {
                "vae_decode": {"tiling": tiling},
            }
        else:
            with pytest.raises(
                ValueError,
                match=(
                    rf"model family {family!r} does not support "
                    r"model\.memory section\(s\): vae_decode"
                ),
            ):
                parse_config(memory_cfg)


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
                        "executor": empty_value,
                        "memory": empty_value,
                    },
                },
            ),
        )

        assert parsed.model is not None


def test_model_family_aliases_preserve_runtime_section_capabilities() -> None:
    def parse_error(family: str, section: str, value: object) -> str | None:
        try:
            parse_config(
                OmegaConf.create(
                    {"model": {"family": family, section: value}},
                ),
            )
        except ValueError as error:
            return str(error)
        return None

    runtime_sections = {
        "executor": {"max_sequence_length": 123},
        "memory": {"vae_decode": {"tiling": True}},
    }
    for alias, family in _FAMILY_BY_ALIAS.items():
        for section, value in runtime_sections.items():
            assert parse_error(alias, section, value) == parse_error(
                family,
                section,
                value,
            )


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


# ── distributed.rollout knobs ─────────────────────────────────────────────────


def test_unknown_batch_placement_strategy_raises() -> None:
    """A typo batch placement strategy is rejected at parse time, not at launch."""
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"batch_placement_strategy": "work_stealing"}},
    )
    with pytest.raises(
        ValueError,
        match=r"unknown distributed\.rollout\.batch_placement_strategy",
    ):
        parse_config(cfg)


def test_rollout_keys_are_registered_not_unknown() -> None:
    """distributed.rollout.* per-worker runtime keys are known to the walker.

    Colocation lives at distributed.resources.rollout.gpu_pool=trainer, so the
    removed distributed.rollout.colocate block is correctly an unknown key."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "cpus_per_worker": 2.0,
                "health_check_interval_s": 45.0,
                "health_check_timeout_s": 30.0,
                "health_check_first_wait_s": 5.0,
                "worker_rpc_timeout_s": 3600.0,
                "generation_stall_timeout_s": 1200.0,
                "batch_placement_strategy": "dynamic",
                "sync_trainable_state": False,
                "pipelined": True,
            }
        }
    )
    unknown = find_unknown_keys(cfg)
    assert not [k for k in unknown if k.startswith("distributed.rollout")]


def test_retired_sync_actor_inflight_knob_is_unknown() -> None:
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "max_inflight_chunks_per_worker": 2,
            },
        },
    )

    assert find_unknown_keys(cfg) == [
        "distributed.rollout.max_inflight_chunks_per_worker",
    ]


def test_rollout_health_check_defaults_and_accepts_override() -> None:
    default = parse_config(_minimal_grpo_cfg(distributed={"rollout": {}}))
    assert default.distributed.rollout.health_check_interval_s == 30.0
    assert default.distributed.rollout.health_check_timeout_s == 30.0
    assert default.distributed.rollout.health_check_first_wait_s == 0.0
    assert default.distributed.rollout.worker_rpc_timeout_s == 600.0
    assert default.distributed.rollout.generation_stall_timeout_s == 3600.0

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 12.5,
                "health_check_timeout_s": 7.5,
                "health_check_first_wait_s": 2.5,
                "worker_rpc_timeout_s": 3600.0,
                "generation_stall_timeout_s": 1200.0,
            }
        },
    )
    rollout = parse_config(cfg).distributed.rollout
    assert rollout.health_check_interval_s == 12.5
    assert rollout.health_check_timeout_s == 7.5
    assert rollout.health_check_first_wait_s == 2.5
    assert rollout.worker_rpc_timeout_s == 3600.0
    assert rollout.generation_stall_timeout_s == 1200.0


def test_rollout_worker_section_mirrors_worker_runtime_config() -> None:
    """RolloutRuntimeSection (pydantic lint boundary) and RolloutWorkerConfig (the
    frozen runtime projection composed into RayGenerationConfig) must stay
    field-for-field identical. ``from_public_section`` builds the dataclass via
    ``cls(**section.model_dump())``, so a field on one but not the other silently
    breaks at runtime (TypeError / unfilled required field) instead of at parse.

    The two types are deliberately NOT merged (the pydantic schema is a lint-only
    boundary; the dataclass is the real runtime consumer), so this parity test is
    the guard against drift. Public defaults remain only on the section; the
    runtime dataclass has no fallback literals.
    """
    import dataclasses

    from vrl.generation.ray.config import RolloutWorkerConfig

    section_fields = set(RolloutRuntimeSection.model_fields)
    config_fields = {f.name for f in dataclasses.fields(RolloutWorkerConfig)}
    assert section_fields == config_fields
    assert "health_check_first_wait_s" in section_fields
    assert "worker_rpc_timeout_s" in section_fields
    assert "generation_stall_timeout_s" in section_fields

    # Per-field default parity: the section's declared defaults must survive the
    # projection unchanged (from_public_section adds no fallbacks or overrides), so
    # the section stays the single home of the default literals.
    projected = RolloutWorkerConfig.from_public_section(RolloutRuntimeSection())
    for name in section_fields:
        assert getattr(projected, name) == RolloutRuntimeSection.model_fields[name].default


@pytest.mark.parametrize("interval_s", [0.0, -1.0])
def test_rollout_health_check_interval_le_zero_disables_probe(interval_s: float) -> None:
    """A non-positive interval turns the probe off; the timeout is then unchecked."""

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": interval_s,
                "health_check_timeout_s": 0.0,
            }
        },
    )

    assert parse_config(cfg).distributed.rollout.health_check_interval_s == interval_s


@pytest.mark.parametrize("interval_s", [float("inf"), float("-inf"), float("nan")])
def test_rollout_health_check_interval_must_be_finite(interval_s: float) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"health_check_interval_s": interval_s}},
    )

    with pytest.raises(ValueError, match=r"health_check_interval_s must be finite"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_rollout_health_check_timeout_must_be_finite_and_positive_when_enabled(
    timeout_s: float,
) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 30.0,
                "health_check_timeout_s": timeout_s,
            }
        },
    )

    with pytest.raises(ValueError, match=r"health_check_timeout_s must be finite and > 0"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "first_wait_s",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_rollout_health_check_first_wait_must_be_finite_and_non_negative(
    first_wait_s: float,
) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"health_check_first_wait_s": first_wait_s}},
    )

    with pytest.raises(ValueError, match=r"health_check_first_wait_s must be finite and >= 0"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_rollout_worker_rpc_timeout_must_be_finite_and_positive(
    timeout_s: float,
) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"worker_rpc_timeout_s": timeout_s}},
    )

    with pytest.raises(ValueError, match=r"worker_rpc_timeout_s must be finite and > 0"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "timeout_s",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_rollout_generation_stall_timeout_must_be_finite_and_positive(
    timeout_s: float,
) -> None:
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"generation_stall_timeout_s": timeout_s}},
    )

    with pytest.raises(ValueError, match=r"generation_stall_timeout_s must be finite and > 0"):
        parse_config(cfg)


def test_reward_resident_overlap_is_not_a_resource_key() -> None:
    """Same-GPU reward/rollout residency is not a public resource topology knob."""
    from vrl.config.unknown_keys import find_unknown_keys

    cfg = _minimal_grpo_cfg(
        distributed={
            "resources": {
                "reward": {
                    "resident_overlap": True,
                },
            },
        },
    )

    assert "distributed.resources.reward.resident_overlap" in find_unknown_keys(cfg)


# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize("loader", _literal_args(DataConfig.model_fields["loader"].annotation))
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    """Every loader in the DataConfig.loader Literal allow-list is accepted; the
    per-loader construction branches below stay as real behavior coverage."""
    if loader == "prompt_manifest":
        data = DataConfig(
            loader=loader,
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": "text"},
            sampler={"type": "random_without_replacement"},
        )
    elif loader == "prompt_image_manifest":
        data = DataConfig(
            loader=loader,
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            loader=loader,
            dataset_name="org/dataset",
            split="train",
            cache_dir="/tmp/cache",
            preprocessing={"resolution": 512, "random_crop": False, "horizontal_flip": True},
            sampler={"shuffle": True, "drop_last": True, "dataloader_num_workers": 4},
        )
    assert data.loader == loader


def test_unknown_data_loader_raises() -> None:
    """Checks unknown data loader raises."""
    cfg = _minimal_grpo_cfg()
    cfg.data.loader = "s3_loader"
    with pytest.raises(ValueError, match=r"unknown data\.loader"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("image_caption_jsonl", "prompt_image_manifest"),
        ("jsonl", "prompt_manifest"),
        ("text", "prompt_manifest"),
    ],
)
def test_omitted_loader_derives_from_preprocessing_format(fmt: str, expected: str) -> None:
    """An omitted data.loader is derived from preprocessing.format for the prompt-* family."""
    if expected == "prompt_image_manifest":
        data = DataConfig(
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": fmt},
            sampler={"type": "random_without_replacement"},
        )
    assert data.loader == expected


@pytest.mark.parametrize(
    ("loader", "fmt", "message"),
    [
        (
            "prompt_manifest",
            "image_caption_jsonl",
            r"requires.*prompt_image_manifest",
        ),
        (
            "prompt_image_manifest",
            "text",
            r"requires.*image_caption_jsonl",
        ),
    ],
)
def test_explicit_data_loader_rejects_preprocessing_format_conflict(
    loader: str,
    fmt: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(
            loader=loader,
            manifest="train.jsonl",
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_image_manifest_requires_image_caption_fields() -> None:
    """Checks prompt image manifest requires image caption fields."""
    with pytest.raises(ValueError, match=r"data\.preprocessing\.caption_field"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest="x",
            eval_manifest="y",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_manifest_accepts_mixture_counts() -> None:
    """A {path: count} manifest is the recipe-level way to declare a prompt mix."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest={"anatomy.jsonl": 6800, "safety.jsonl": 1200},
        mix_seed=20260818,
        preprocessing={"format": "jsonl"},
        sampler={"type": "random_without_replacement"},
    )
    assert data.manifest == {"anatomy.jsonl": 6800, "safety.jsonl": 1200}


def test_prompt_manifest_mixture_requires_a_seed() -> None:
    """A mixture without a seed would draw a different prompt set on every rank."""
    with pytest.raises(ValueError, match=r"data\.mix_seed"):
        DataConfig(
            loader="prompt_manifest",
            manifest={"anatomy.jsonl": 6800, "safety.jsonl": 1200},
            preprocessing={"format": "jsonl"},
            sampler={"type": "random_without_replacement"},
        )


@pytest.mark.parametrize("count", [0, -5, 1.5, "many"])
def test_prompt_manifest_rejects_non_positive_mixture_count(count: object) -> None:
    """A mixture count that cannot select prompts fails at config time."""
    with pytest.raises(ValueError, match=r"positive prompt count"):
        DataConfig(
            loader="prompt_manifest",
            manifest={"anatomy.jsonl": count},
            preprocessing={"format": "jsonl"},
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_image_manifest_rejects_mixture() -> None:
    """Image-conditioned runs pair one manifest with its reference tree."""
    with pytest.raises(ValueError, match=r"single data\.manifest path"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest={"a.jsonl": 10, "b.jsonl": 10},
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


# ── Sampler type literal ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sampler_type",
    ["random_without_replacement", "sequential_window"],
)
def test_valid_sampler_types_are_accepted(sampler_type: str) -> None:
    """Checks valid sampler types are accepted."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest="x",
        preprocessing={"format": "text"},
        sampler={"type": sampler_type},
    )
    assert data.sampler["type"] == sampler_type


def test_unknown_sampler_type_raises() -> None:
    """Checks unknown sampler type raises."""
    with pytest.raises(ValueError, match=r"unknown data\.sampler\.type"):
        DataConfig(
            loader="prompt_manifest",
            manifest="x",
            preprocessing={},
            sampler={"type": "round_robin"},
        )


# ── Reward weight validation ──────────────────────────────────────────────────


def test_zero_weight_observation_component_is_valid() -> None:
    """Checks zero weight keeps a component valid for observation-only scoring."""
    cfg = RewardConfig.model_validate({"components": {"kling_video_reward": 0.0}, "kwargs": {}})
    assert cfg.components["kling_video_reward"] == 0.0


def test_non_numeric_reward_weight_raises() -> None:
    """Checks non numeric reward weight raises."""
    with pytest.raises(ValueError, match="must be numeric"):
        RewardConfig.model_validate({"components": {"aesthetic": "heavy"}, "kwargs": {}})


def test_reward_http_inference_config_is_typed_beside_open_component_kwargs() -> None:
    """Transport config is typed even though reward-specific kwargs are open."""

    cfg = RewardConfig.model_validate(
        {
            "components": {"videoscore2": 1.0},
            "kwargs": {"videoscore2": {"artifact_dir": "/shared/artifacts"}},
            "inference": {
                "videoscore2": {
                    "kind": "http",
                    "endpoint": "http://reward:8300",
                    "expected_model": "videoscore2-v1",
                },
            },
        },
    )

    assert cfg.inference["videoscore2"].kind == "http"


def test_reward_inference_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match=r"unsupported reward\.inference\..* keys"):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "inference": {
                    "videoscore2": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                        "service_url": "http://legacy",
                    },
                },
            },
        )


def test_reward_inference_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "inference": {
                    "typo_component": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                },
            },
        )


def test_grpo_requires_valid_sde_type() -> None:
    """Checks GRPO requires valid SDE type."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "euler"
    with pytest.raises(ValueError, match=r"unknown rollout\.sde\.type"):
        parse_config(cfg)


def test_grpo_accepts_cps_sde_type() -> None:
    """Checks GRPO accepts cps SDE type."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "cps"
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


def test_diffusion_nft_requires_valid_sde_type() -> None:
    """Checks diffusion NFT requires valid SDE type."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_nft"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "rollout": {"sde": {"type": "invalid"}},
        }
    )
    with pytest.raises(ValueError, match=r"unknown rollout\.sde\.type"):
        parse_config(cfg)


def test_token_grpo_multisegment_requires_explicit_janus_r1_family() -> None:
    """The algorithm cannot silently turn base Janus into the R1 protocol."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo_multisegment"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_pro"},
            "rollout": {"final_image_policy": "always_generate"},
        }
    )
    with pytest.raises(ValueError, match="janus_pro_r1"):
        parse_config(cfg)


def test_token_grpo_multisegment_final_image_policy_single_source() -> None:
    """final_image_policy is owned by the rollout section."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo_multisegment"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_pro_r1"},
            "rollout": {"final_image_policy": "always_generate"},
        }
    )
    assert parse_config(cfg).rollout.final_image_policy == "always_generate"


def test_janus_r1_family_requires_multisegment_algorithm() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_r1"},
            "rollout": {},
        }
    )

    with pytest.raises(ValueError, match="token_grpo_multisegment"):
        parse_config(cfg)


def test_production_video_reward_structural_rules() -> None:
    """Checks production video reward structural rules."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
                "task_type": "text_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "sleep_offload": True,
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )
    from vrl.config.validation import validate_production_reward_contract

    parse_config(cfg)  # schema parse stays clean
    validate_production_reward_contract(cfg)


def test_production_video_reward_accepts_image_to_video_task_type() -> None:
    """Checks production video reward accepts image to video task type."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_image_manifest",
                "manifest": "x",
                "eval_manifest": "y",
                "preprocessing": {
                    "format": "image_caption_jsonl",
                    "image_field": "image",
                    "caption_field": "caption",
                    "conditioning": "reference_image",
                },
                "sampler": {"type": "random_without_replacement"},
                "task_type": "image_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "sleep_offload": True,
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        },
    )

    parsed = parse_config(cfg)

    assert parsed.data.task_type == "image_to_video"


def test_production_schema_defaults_known_gate_to_disabled() -> None:
    cfg = OmegaConf.create({"production": {}})

    parsed = parse_config(cfg)

    assert parsed.production is not None
    assert parsed.production.kling_video_reward.enabled is False


def test_production_schema_accepts_enabled_gate() -> None:
    cfg = OmegaConf.create(
        {"production": {"kling_video_reward": {"enabled": True}}},
    )

    parsed = parse_config(cfg)

    assert parsed.production is not None
    assert parsed.production.kling_video_reward.enabled is True


# ── Missing field mapping (??? → ValueError) ──────────────────────────────────


def test_missing_mandatory_value_produces_repo_standard_message() -> None:
    """Checks missing mandatory value produces repo standard message."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "flow_grpo"
    # Inject an OmegaConf mandatory-missing marker
    OmegaConf.update(cfg, "algorithm.kind", "???")
    with pytest.raises(ValueError, match="config missing required field"):
        parse_config(cfg)


# ── extra="ignore" migration policy ──────────────────────────────────────────


def test_unknown_top_level_sections_are_rejected() -> None:
    """parse_config is the one gate: a section no consumer reads fails loud."""
    cfg = _minimal_grpo_cfg()
    OmegaConf.update(cfg, "some_future_section.foo", "bar")
    with pytest.raises(ValueError, match=r"unknown some_future_section"):
        parse_config(cfg)


def test_unknown_reward_component_with_unknown_kwargs_is_accepted() -> None:
    """Checks unknown reward component with unknown kwargs is accepted."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "rollout": {"sde": {"type": "flow_grpo"}},
            "reward": {
                "components": {"custom_reward": 1.0},
                "kwargs": {"custom_reward": {"model_repo": "org/model"}},
            },
        }
    )
    parsed = parse_config(cfg)
    assert parsed.reward.components["custom_reward"] == 1.0


# ── algorithm.sft_weight x data.sft_latents (regularizer data channel) ───────


def test_sft_weight_without_latents_shard_raises() -> None:
    """A weight without its data channel would be a silent no-op knob."""
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    with pytest.raises(ValueError, match=r"data\.sft_latents"):
        parse_config(cfg)


def test_sft_weight_with_latents_shard_parses() -> None:
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


def test_diffusion_dpo_sft_weight_does_not_require_online_latents_shard() -> None:
    cfg = _minimal_grpo_cfg(
        algorithm={"kind": "diffusion_dpo", "sft_weight": 0.1},
    )
    del cfg.rollout
    parsed = parse_config(cfg)
    assert parsed.algorithm.sft_weight == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("section", "payload", "field"),
    [
        ("actor", {"ema": {"enable": True}}, "actor.ema"),
        ("actor", {"ppo_epochs": 2}, "actor.ppo_epochs"),
        ("trainer", {"total_epochs": 10}, "trainer.total_epochs"),
        ("trainer", {"precision_drift_guard": {"mode": "warn"}}, "trainer.precision_drift_guard"),
        ("rollout", {"prompts_per_batch": 1}, "rollout.prompts_per_batch"),
    ],
)
def test_diffusion_dpo_rejects_online_only_config_fields(
    section: str,
    payload: dict,
    field: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_dpo"},
            section: payload,
        },
    )

    with pytest.raises(ValueError, match=rf"{field}"):
        parse_config(cfg)


def test_diffusion_dpo_accepts_its_resume_and_optimizer_surface() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_dpo"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "bf16"},
            },
            "actor": {
                "optim": {"lr": 1e-8},
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "max_norm": 1.0,
                "prediction_type": "flow_matching",
                "scale_lr": False,
                "train_batch_size": 1,
                "use_adafactor": False,
            },
            "trainer": {
                "checkpointing_steps": 10,
                "entrypoint": "pkg.module:train",
                "log_interval": 1,
                "max_train_steps": 20,
                "output_dir": "outputs/dpo",
                "resume_from": "",
                "resume_strict": True,
            },
        },
    )

    parsed = parse_config(cfg)

    assert parsed.algorithm.kind == "diffusion_dpo"


def test_latents_shard_without_weight_is_inert_and_allowed() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_sft_weight_must_be_finite_and_nonnegative(value: float) -> None:
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": value})
    with pytest.raises(ValueError, match="finite number >= 0"):
        parse_config(cfg)


def test_sft_weight_rejects_non_diffusion_grpo_kind() -> None:
    cfg = _minimal_grpo_cfg(
        algorithm={"kind": "flash_grpo", "sft_weight": 0.1},
    )
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    with pytest.raises(ValueError, match="only for diffusion"):
        parse_config(cfg)
