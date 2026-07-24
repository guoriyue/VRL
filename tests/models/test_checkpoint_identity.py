"""CPU-only tests for immutable model checkpoint identity."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vrl.config.model_schema import LoraSection, ModelSection
from vrl.config.precision import RolePrecision
from vrl.families.registry import FAMILY_REGISTRY, TokenFamilyBuild, get_model_family_entry
from vrl.models.checkpoint_identity import (
    MODEL_IDENTITY_SCHEMA,
    LocalCheckpointContent,
    local_checkpoint_content,
    resolve_checkpoint_model_identity,
    validate_checkpoint_identity_schema,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.utils.config import import_from_path

_COMMIT = "a" * 40
_OTHER_COMMIT = "b" * 40


def _build(
    *,
    family: str = "sana",
    path: str = "example/model",
    revision: str | None = _COMMIT,
    **model_config: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        family=family,
        model_name_or_path=path,
        revision=revision,
        model_config=model_config,
    )


def _token_build(
    family: str,
    *,
    sampling_config: dict[str, object] | None = None,
    **model_config: object,
) -> ModelBuild:
    recipe = get_model_family_entry(family).family_build
    assert isinstance(recipe, TokenFamilyBuild)
    return ModelBuild(
        model_name_or_path=recipe.default_model_path,
        revision=_COMMIT,
        device="cpu",
        parameter_dtype="fp32",
        family=family,
        precision=RolePrecision(dtype="fp32", float32_precision="tf32"),
        model_config=dict(model_config),
        sampling_config=dict(sampling_config or {}),
    )


def test_every_registered_model_and_nested_lora_field_is_classified() -> None:
    validate_checkpoint_identity_schema(LoraSection)
    for entry in FAMILY_REGISTRY.values():
        validate_checkpoint_identity_schema(import_from_path(entry.model_section_cls))


def test_new_model_or_lora_field_without_metadata_fails_closed() -> None:
    class IncompleteModelSection(ModelSection):
        new_source_selector: str | None = None

    class IncompleteLoraSection(LoraSection):
        new_adapter_shape: int | None = None

    with pytest.raises(TypeError, match="new_source_selector"):
        validate_checkpoint_identity_schema(IncompleteModelSection)
    with pytest.raises(TypeError, match="new_adapter_shape"):
        validate_checkpoint_identity_schema(IncompleteLoraSection)


def test_local_file_identity_is_path_independent_and_counts_content(tmp_path: Path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "elsewhere" / "right.bin"
    right.parent.mkdir()
    left.write_bytes(b"checkpoint-bytes")
    right.write_bytes(b"checkpoint-bytes")

    left_identity = local_checkpoint_content(left)
    right_identity = local_checkpoint_content(right)

    assert left_identity == right_identity
    assert left_identity == LocalCheckpointContent(
        kind="file",
        sha256=left_identity.sha256,
        bytes=16,
        files=1,
    )

    right.write_bytes(b"different")
    assert local_checkpoint_content(right) != left_identity


def test_local_tree_identity_is_path_independent_and_follows_symlinks(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    linked = tmp_path / "linked"
    external = tmp_path / "external"
    (plain / "weights").mkdir(parents=True)
    linked.mkdir()
    external.mkdir()
    (plain / "weights" / "model.bin").write_bytes(b"model")
    (external / "model.bin").write_bytes(b"model")
    (linked / "weights").symlink_to(external, target_is_directory=True)

    plain_identity = local_checkpoint_content(plain)
    linked_identity = local_checkpoint_content(linked)

    assert plain_identity == linked_identity
    assert plain_identity.kind == "tree"
    assert plain_identity.bytes == 5
    assert plain_identity.files == 1


def test_local_source_rejects_broken_link_cycle_and_special_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken-root"
    broken.mkdir()
    (broken / "weights").symlink_to(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="cannot resolve"):
        local_checkpoint_content(broken)

    cycle = tmp_path / "cycle-root"
    cycle.mkdir()
    (cycle / "loop").symlink_to(cycle, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink cycle"):
        local_checkpoint_content(cycle)

    special = tmp_path / "special-root"
    special.mkdir()
    os.mkfifo(special / "weights.pipe")
    with pytest.raises(RuntimeError, match="special file"):
        local_checkpoint_content(special)


def test_local_source_rejects_file_mutation_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.models.checkpoint_identity as identity_module

    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"weights")
    real_fstat = identity_module.os.fstat
    calls = 0

    class _ChangedStat:
        def __init__(self, original: os.stat_result) -> None:
            self._original = original

        def __getattr__(self, name: str) -> object:
            if name == "st_mtime_ns":
                return self._original.st_mtime_ns + 1
            return getattr(self._original, name)

    def changing_fstat(fd: int) -> os.stat_result | _ChangedStat:
        nonlocal calls
        calls += 1
        result = real_fstat(fd)
        return result if calls == 1 else _ChangedStat(result)

    monkeypatch.setattr(identity_module.os, "fstat", changing_fstat)

    with pytest.raises(RuntimeError, match="changed while hashing"):
        local_checkpoint_content(checkpoint)


def test_local_source_rejects_root_symlink_retarget_during_resolution(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "model.bin").write_bytes(b"first")
    (second / "model.bin").write_bytes(b"second")
    alias = tmp_path / "model"
    alias.symlink_to(first, target_is_directory=True)

    def retargeting_resolver(path: Path) -> LocalCheckpointContent:
        content = local_checkpoint_content(path)
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)
        return content

    with pytest.raises(RuntimeError, match="changed while hashing"):
        resolve_checkpoint_model_identity(
            _build(path=str(alias), revision=None),
            local_resolver=retargeting_resolver,
        )


@pytest.mark.parametrize("revision", [None, "", "main", "a" * 39, "A" * 40])
def test_remote_source_requires_full_lowercase_commit(revision: str | None) -> None:
    with pytest.raises(ValueError, match="40-character commit"):
        resolve_checkpoint_model_identity(_build(revision=revision))


def test_remote_source_requires_valid_huggingface_repo_id() -> None:
    with pytest.raises(ValueError, match="valid Hugging Face repo id"):
        resolve_checkpoint_model_identity(
            _build(path="../missing-local-checkpoint"),
        )


@pytest.mark.parametrize(
    "member",
    [
        "../outside.pt",
        "/outside.pt",
        "weights/../outside.pt",
        "./weights.pt",
        "weights//model.pt",
        r"weights\model.pt",
        "C:/outside.pt",
    ],
)
def test_checkpoint_source_member_cannot_escape_source(member: str) -> None:
    with pytest.raises(ValueError, match=r"checkpoint source|POSIX"):
        resolve_checkpoint_model_identity(
            _token_build(
                "llamagen",
                gpt_ckpt=member,
                t5_revision=_OTHER_COMMIT,
            ),
        )


def test_remote_identity_uses_protocol_source_and_behavior_keys() -> None:
    identity = resolve_checkpoint_model_identity(_build())

    assert identity == {
        "schema": MODEL_IDENTITY_SCHEMA,
        "sources": {
            "main": {
                "kind": "huggingface",
                "repo_id": "example/model",
                "revision": _COMMIT,
            },
        },
        "build": {"use_lora": False},
    }


def test_local_identity_omits_root_path_and_caches_duplicate_source(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    calls: list[Path] = []

    def resolver(path: Path) -> LocalCheckpointContent:
        calls.append(path)
        return local_checkpoint_content(path)

    identity = resolve_checkpoint_model_identity(
        _build(
            family="echo",
            path=str(model),
            revision="ignored-for-local-content",
            gemma_path=str(model),
            gemma_revision=None,
        ),
        local_resolver=resolver,
    )

    assert calls == [model.resolve()]
    assert identity["sources"]["main"] == identity["sources"]["gemma"]
    assert identity["sources"]["main"]["kind"] == "local-tree"
    assert identity["sources"]["main"]["bytes"] == 7
    assert identity["sources"]["main"]["files"] == 1
    assert str(tmp_path) not in repr(identity)


def test_disabled_lora_ignores_topology_initializer_and_warm_start_path() -> None:
    left = resolve_checkpoint_model_identity(
        _build(
            use_lora=False,
            lora={
                "rank": 8,
                "alpha": 16,
                "target_modules": ["q"],
                "dropout": 0.1,
                "init": "gaussian",
                "path": "/first",
            },
        ),
    )
    right = resolve_checkpoint_model_identity(
        _build(
            use_lora=False,
            lora={
                "rank": 64,
                "alpha": 128,
                "target_modules": ["v"],
                "dropout": 0.9,
                "init": False,
                "path": "/second",
            },
        ),
    )

    assert left == right
    assert left["build"] == {"use_lora": False}


def test_enabled_lora_includes_topology_but_not_initializer_or_warm_start_path() -> None:
    first = resolve_checkpoint_model_identity(
        _build(
            use_lora=True,
            lora={
                "rank": 8,
                "alpha": 16,
                "target_modules": ["q", "v"],
                "dropout": 0.25,
                "init_lora_weights": "gaussian",
                "path": "/first",
            },
        ),
    )
    second = resolve_checkpoint_model_identity(
        _build(
            use_lora=True,
            lora={
                "rank": 8,
                "alpha": 16,
                "target_modules": ["v", "q", "q"],
                "dropout": 0.25,
                "init_lora_weights": False,
                "path": "/second",
            },
        ),
    )

    assert first == second
    assert first["build"]["lora"] == {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q", "v"],
        "dropout": 0.25,
    }

    changed = resolve_checkpoint_model_identity(
        _build(
            use_lora=True,
            lora={"rank": 16, "alpha": 16, "target_modules": ["q", "v"]},
        ),
    )
    assert changed != first


def test_enabled_lora_requires_shape_fields() -> None:
    with pytest.raises(ValueError, match=r"model\.lora\.target_modules"):
        resolve_checkpoint_model_identity(
            _build(use_lora=True, lora={"rank": 8, "alpha": 16}),
        )


def test_anima_uses_main_members_and_independent_tokenizer_sources() -> None:
    identity = resolve_checkpoint_model_identity(
        _build(
            family="cosmos-predict2-anima",
            transformer_file="transformer.safetensors",
            text_encoder_file="text.safetensors",
            vae_file="vae.safetensors",
            qwen_tokenizer_path="example/qwen",
            qwen_tokenizer_revision=_OTHER_COMMIT,
            t5_tokenizer_path="example/t5",
            t5_tokenizer_revision=_OTHER_COMMIT,
            scheduler_shift=99.0,
        ),
    )

    assert set(identity["sources"]) == {"main", "qwen_tokenizer", "t5_tokenizer"}
    assert identity["build"] == {
        "scheduler_shift": 99.0,
        "text_encoder_file": "text.safetensors",
        "transformer_file": "transformer.safetensors",
        "use_lora": False,
        "vae_file": "vae.safetensors",
    }


def test_anima_scheduler_shift_default_is_canonical_and_override_changes_identity() -> None:
    omitted = resolve_checkpoint_model_identity(
        _build(
            family="cosmos-predict2-anima",
            transformer_file="transformer.safetensors",
        ),
    )
    explicit_default = resolve_checkpoint_model_identity(
        _build(
            family="cosmos-predict2-anima",
            transformer_file="transformer.safetensors",
            scheduler_shift=3.0,
        ),
    )
    changed = resolve_checkpoint_model_identity(
        _build(
            family="cosmos-predict2-anima",
            transformer_file="transformer.safetensors",
            scheduler_shift=4.0,
        ),
    )

    assert omitted == explicit_default
    assert omitted["build"]["scheduler_shift"] == 3.0
    assert changed["build"]["scheduler_shift"] == 4.0
    assert changed != omitted


def test_anima_explicit_artifact_source_overrides_main_member(
    tmp_path: Path,
) -> None:
    transformer = tmp_path / "transformer.safetensors"
    transformer.write_bytes(b"transformer")
    identity = resolve_checkpoint_model_identity(
        _build(
            family="cosmos-predict2-anima",
            transformer_file="ignored.safetensors",
            transformer_path=str(transformer),
        ),
    )

    assert set(identity["sources"]) == {"transformer"}
    assert "transformer_file" not in identity["build"]
    assert identity["sources"]["transformer"]["kind"] == "local-file"


def test_causvid_local_checkpoint_file_ignores_unused_member(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    identity = resolve_checkpoint_model_identity(
        _build(
            family="causvid",
            path=str(checkpoint),
            revision=None,
            checkpoint_file="ignored/model.pt",
            base_model_path="example/base",
            base_model_revision=_OTHER_COMMIT,
            causvid_source_path="/location-only",
            causvid_source_revision="c" * 40,
            checkpoint_sha256="d" * 64,
            accept_noncommercial_license=True,
        ),
    )

    assert set(identity["sources"]) == {"base_model", "main"}
    assert identity["build"] == {
        "causvid_source_revision": "c" * 40,
        "use_lora": False,
    }
    assert "ignored/model.pt" not in repr(identity)
    assert "location-only" not in repr(identity)


def test_llamagen_members_model_choice_and_t5_source_are_identity() -> None:
    identity = resolve_checkpoint_model_identity(
        _token_build(
            "llamagen",
            gpt_ckpt="gpt.pt",
            vq_ckpt="vq.pt",
            gpt_model="GPT-XL",
            t5_path="example/t5",
            t5_revision=_OTHER_COMMIT,
        ),
    )

    assert set(identity["sources"]) == {"main", "t5"}
    assert identity["build"] == {
        "gpt_ckpt": "gpt.pt",
        "gpt_model": "GPT-XL",
        "image_token_num": 256,
        "use_lora": False,
        "vq_ckpt": "vq.pt",
    }


def test_llamagen_construction_grid_is_model_owned_identity() -> None:
    shared = {"t5_revision": _OTHER_COMMIT}
    omitted_request_value = resolve_checkpoint_model_identity(
        _token_build("llamagen", **shared),
    )
    matching_request_value = resolve_checkpoint_model_identity(
        _token_build(
            "llamagen",
            **shared,
            image_token_num=256,
            sampling_config={"image_token_num": 256},
        ),
    )
    changed_construction = resolve_checkpoint_model_identity(
        _token_build(
            "llamagen",
            **shared,
            image_token_num=1024,
            sampling_config={"image_token_num": 1024},
        ),
    )

    assert omitted_request_value == matching_request_value
    assert omitted_request_value["build"]["image_token_num"] == 256
    assert changed_construction["build"]["image_token_num"] == 1024
    assert changed_construction != omitted_request_value

    with pytest.raises(
        ValueError,
        match=r"sampling\.image_token_num=1024.*model\.image_token_num=256",
    ):
        resolve_checkpoint_model_identity(
            _token_build(
                "llamagen",
                **shared,
                image_token_num=256,
                sampling_config={"image_token_num": 1024},
            ),
        )


def test_echo_construction_dimensions_change_identity() -> None:
    shared = {
        "family": "echo",
        "gemma_path": "example/gemma",
        "gemma_revision": _OTHER_COMMIT,
    }
    omitted = resolve_checkpoint_model_identity(_build(**shared))
    explicit_defaults = resolve_checkpoint_model_identity(
        _build(**shared, video_height=512, video_width=768),
    )
    changed = resolve_checkpoint_model_identity(
        _build(**shared, video_height=256, video_width=256),
    )

    assert omitted == explicit_defaults
    assert omitted["build"]["video_height"] == 512
    assert omitted["build"]["video_width"] == 768
    assert changed["build"]["video_height"] == 256
    assert changed["build"]["video_width"] == 256
    assert changed != omitted


@pytest.mark.parametrize(
    ("family", "included", "first", "second"),
    [
        ("flux", "nft_previous_adapter", False, True),
        ("cosmos-predict2.5", "skip_text_encoder", False, True),
        ("janus_pro", "vq_latent_channels", 8, 16),
    ],
)
def test_family_behavior_value_changes_identity(
    family: str,
    included: str,
    first: object,
    second: object,
) -> None:
    if isinstance(get_model_family_entry(family).family_build, TokenFamilyBuild):
        left_build = _token_build(family, **{included: first})
        right_build = _token_build(family, **{included: second})
    else:
        extra = (
            {"trainable_transformers": ["transformer_2"]} if family.startswith("wan_2_1") else {}
        )
        left_build = _build(family=family, **extra, **{included: first})
        right_build = _build(family=family, **extra, **{included: second})
    left = resolve_checkpoint_model_identity(left_build)
    right = resolve_checkpoint_model_identity(right_build)

    assert left["build"][included] == first
    assert right["build"][included] == second
    assert left != right


def test_runtime_only_fields_do_not_change_wan_identity() -> None:
    baseline = resolve_checkpoint_model_identity(
        _build(
            family="wan_2_1_i2v",
            offload_mode="none",
            expert_lifecycle_profiling=False,
            trainable_transformers=["transformer_2"],
            memory={"vae_decode": {"tiling": False}},
            torch_compile={"enable": False},
        ),
    )
    runtime_tuned = resolve_checkpoint_model_identity(
        _build(
            family="wan_2_1_i2v",
            offload_mode="sequential",
            expert_lifecycle_profiling=True,
            trainable_transformers=["transformer_2"],
            memory={"vae_decode": {"tiling": True}},
            torch_compile={"enable": True, "mode": "reduce-overhead"},
        ),
    )

    assert baseline == runtime_tuned


def test_source_derived_wan_boundary_is_not_duplicated_in_identity() -> None:
    first_source = resolve_checkpoint_model_identity(
        _build(
            family="wan_2_1_i2v",
            boundary_ratio=0.9,
            trainable_transformers=["transformer_2"],
        ),
    )
    changed_source = resolve_checkpoint_model_identity(
        _build(
            family="wan_2_1_i2v",
            revision=_OTHER_COMMIT,
            boundary_ratio=0.8,
            trainable_transformers=["transformer_2"],
        ),
    )

    assert "boundary_ratio" not in first_source["build"]
    assert "boundary_ratio" not in changed_source["build"]
    assert first_source["sources"]["main"]["revision"] == _COMMIT
    assert changed_source["sources"]["main"]["revision"] == _OTHER_COMMIT
    assert first_source != changed_source


def test_wan_trainable_transformer_topology_changes_identity() -> None:
    low_noise_only = resolve_checkpoint_model_identity(
        _build(
            family="wan_2_1_i2v",
            trainable_transformers=["transformer_2"],
        ),
    )
    both_experts = resolve_checkpoint_model_identity(
        _build(
            family="wan_2_1_i2v",
            trainable_transformers=["transformer_2", "transformer", "transformer_2"],
        ),
    )

    assert low_noise_only["build"]["trainable_transformers"] == ["transformer_2"]
    assert both_experts["build"]["trainable_transformers"] == [
        "transformer",
        "transformer_2",
    ]
    assert low_noise_only != both_experts


@pytest.mark.parametrize(
    ("family", "expected_targets", "source_pins"),
    [
        ("janus_pro", ["q_proj", "k_proj", "v_proj", "o_proj"], {}),
        ("emu3", ["q_proj", "k_proj", "v_proj", "o_proj"], {}),
        ("glm_image", ["q_proj", "k_proj", "v_proj", "o_proj"], {}),
        (
            "nextstep_1",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            {"vae_revision": _OTHER_COMMIT},
        ),
        (
            "llamagen",
            ["wqkv", "wo"],
            {"t5_revision": _OTHER_COMMIT},
        ),
    ],
)
def test_token_partial_lora_override_resolves_family_defaults(
    family: str,
    expected_targets: list[str],
    source_pins: dict[str, object],
) -> None:
    partial = resolve_checkpoint_model_identity(
        _token_build(
            family,
            **source_pins,
            use_lora=True,
            lora={"rank": 8},
        ),
    )
    explicit = resolve_checkpoint_model_identity(
        _token_build(
            family,
            **source_pins,
            use_lora=True,
            lora={
                "rank": 8,
                "alpha": 64,
                "target_modules": expected_targets,
                "dropout": 0.0,
            },
        ),
    )

    assert partial == explicit
    assert partial["build"]["lora"] == {
        "rank": 8,
        "alpha": 64,
        "target_modules": sorted(set(expected_targets)),
        "dropout": 0.0,
    }


def test_token_family_source_and_member_defaults_match_explicit_values() -> None:
    from vrl.models.families.llamagen.config import LlamaGenConfig
    from vrl.models.families.nextstep_1.config import NextStep1Config

    llamagen_defaults = LlamaGenConfig()
    llamagen_omitted = resolve_checkpoint_model_identity(
        _token_build("llamagen", t5_revision=_OTHER_COMMIT),
    )
    llamagen_explicit = resolve_checkpoint_model_identity(
        _token_build(
            "llamagen",
            gpt_ckpt=llamagen_defaults.gpt_ckpt,
            gpt_model=llamagen_defaults.gpt_model,
            image_token_num=llamagen_defaults.image_token_num,
            t5_path=llamagen_defaults.t5_path,
            t5_revision=_OTHER_COMMIT,
            vq_ckpt=llamagen_defaults.vq_ckpt,
        ),
    )
    assert llamagen_omitted == llamagen_explicit

    nextstep_defaults = NextStep1Config()
    nextstep_omitted = resolve_checkpoint_model_identity(
        _token_build("nextstep_1", vae_revision=_OTHER_COMMIT),
    )
    nextstep_explicit = resolve_checkpoint_model_identity(
        _token_build(
            "nextstep_1",
            vae_path=nextstep_defaults.vae_path,
            vae_revision=_OTHER_COMMIT,
        ),
    )
    assert nextstep_omitted == nextstep_explicit


def test_causvid_protocol_defaults_match_explicit_values() -> None:
    from vrl.models.families.causvid.config import (
        CAUSVID_CHECKPOINT_FILE,
        CAUSVID_SOURCE_REVISION,
    )

    shared = {
        "family": "causvid",
        "base_model_path": "example/base",
        "base_model_revision": _OTHER_COMMIT,
    }
    omitted = resolve_checkpoint_model_identity(_build(**shared))
    explicit = resolve_checkpoint_model_identity(
        _build(
            **shared,
            checkpoint_file=CAUSVID_CHECKPOINT_FILE,
            causvid_source_revision=CAUSVID_SOURCE_REVISION,
        ),
    )

    assert omitted == explicit
    assert omitted["build"]["checkpoint_file"] == CAUSVID_CHECKPOINT_FILE
    assert omitted["build"]["causvid_source_revision"] == CAUSVID_SOURCE_REVISION
