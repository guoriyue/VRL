"""CPU-only tests for immutable model checkpoint identity."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vrl.config.model_schema import LoraSection, ModelSection
from vrl.models.checkpoint_identity import (
    MODEL_IDENTITY_SCHEMA,
    LocalCheckpointContent,
    resolve_checkpoint_model_identity,
    validate_checkpoint_identity_schema,
)
from vrl.models.families.registry import FAMILY_REGISTRY
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

    left_identity = LocalCheckpointContent.from_path(left)
    right_identity = LocalCheckpointContent.from_path(right)

    assert left_identity == right_identity
    assert left_identity == LocalCheckpointContent(
        kind="file",
        sha256=left_identity.sha256,
        bytes=16,
        files=1,
    )

    right.write_bytes(b"different")
    assert LocalCheckpointContent.from_path(right) != left_identity


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

    plain_identity = LocalCheckpointContent.from_path(plain)
    linked_identity = LocalCheckpointContent.from_path(linked)

    assert plain_identity == linked_identity
    assert plain_identity.kind == "tree"
    assert plain_identity.bytes == 5
    assert plain_identity.files == 1


def test_local_source_rejects_broken_link_cycle_and_special_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken-root"
    broken.mkdir()
    (broken / "weights").symlink_to(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="cannot resolve"):
        LocalCheckpointContent.from_path(broken)

    cycle = tmp_path / "cycle-root"
    cycle.mkdir()
    (cycle / "loop").symlink_to(cycle, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink cycle"):
        LocalCheckpointContent.from_path(cycle)

    special = tmp_path / "special-root"
    special.mkdir()
    os.mkfifo(special / "weights.pipe")
    with pytest.raises(RuntimeError, match="special file"):
        LocalCheckpointContent.from_path(special)


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
        LocalCheckpointContent.from_path(checkpoint)


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
        content = LocalCheckpointContent.from_path(path)
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)
        return content

    with pytest.raises(RuntimeError, match="changed while hashing"):
        resolve_checkpoint_model_identity(
            _build(path=str(alias), revision=None),
            local_resolver=retargeting_resolver,
        )


@pytest.mark.parametrize("revision", ["main", "A" * 40])
def test_remote_source_requires_full_lowercase_commit(revision: str | None) -> None:
    with pytest.raises(ValueError, match="40-character commit"):
        resolve_checkpoint_model_identity(_build(revision=revision))


def test_remote_source_requires_valid_huggingface_repo_id() -> None:
    with pytest.raises(ValueError, match="valid Hugging Face repo id"):
        resolve_checkpoint_model_identity(
            _build(path="../missing-local-checkpoint"),
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
        return LocalCheckpointContent.from_path(path)

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
        ("cosmos-predict2.5", "skip_text_encoder", False, True),
    ],
)
def test_family_behavior_value_changes_identity(
    family: str,
    included: str,
    first: object,
    second: object,
) -> None:
    left_build = _build(family=family, **{included: first})
    right_build = _build(family=family, **{included: second})
    left = resolve_checkpoint_model_identity(left_build)
    right = resolve_checkpoint_model_identity(right_build)

    assert left["build"][included] == first
    assert right["build"][included] == second
    assert left != right


@pytest.mark.parametrize("family", ["wan_2_1", "wan_2_1_i2v"])
def test_wan_adapter_storage_identity_is_opt_in(family: str) -> None:
    values = {"family": family, "trainable_transformers": ["transformer"], "use_lora": True}
    lora = {"rank": 8, "alpha": 16, "target_modules": ["to_q"]}
    baseline = resolve_checkpoint_model_identity(_build(**values, lora=lora))
    explicit_default = resolve_checkpoint_model_identity(
        _build(**values, lora={**lora, "parameter_dtype": None}),
    )
    fp32 = resolve_checkpoint_model_identity(
        _build(**values, lora={**lora, "parameter_dtype": "float32"}),
    )
    assert baseline == explicit_default
    assert "lora_parameter_dtype" not in baseline["build"]
    assert fp32["build"]["lora_parameter_dtype"] == "float32"
    assert fp32 != baseline


def test_lora_storage_autocast_changes_identity() -> None:
    field = "autocast_adapter_dtype"
    lora = {"rank": 8, "alpha": 16, "target_modules": ["to_q"]}
    identities = [
        resolve_checkpoint_model_identity(
            _build(family="flux", use_lora=True, lora={**lora, field: value}),
        )
        for value in (False, True)
    ]
    assert identities[0]["build"]["lora"][field] is False
    assert field not in identities[1]["build"]["lora"]
    assert identities[0] != identities[1]


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
