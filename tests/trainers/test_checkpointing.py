from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from vrl.models.interfaces.runtime import register_checkpoint_owned_state
from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    LORA_WEIGHTS_NAME,
    TRAINING_CHECKPOINT_NAME,
    AdapterExport,
    TrainingCheckpoint,
    export_checkpoint_state,
    infer_next_epoch,
    load_checkpoint_state,
    load_training_checkpoint,
    prepare_model_config_for_training_resume,
    resolve_training_resume_config,
    resolve_training_resume_strict,
    restore_model_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_compatibility,
    validate_checkpoint_meta_compatibility,
)

UNIT_IDENTITY = {"schema": "unit-model/v1"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_resume_strict_uses_checkpoint_policy(
    value: bool | None,
    expected: bool,
) -> None:
    trainer = {} if value is None else {"resume_strict": value}
    cfg = OmegaConf.create({"trainer": trainer})

    assert resolve_training_resume_strict(cfg) is expected


def test_resume_strict_rejects_string_truthiness() -> None:
    cfg = OmegaConf.create({"trainer": {"resume_strict": "false"}})

    with pytest.raises(TypeError, match="must be a boolean"):
        resolve_training_resume_strict(cfg)


@pytest.mark.parametrize(
    ("resume_from", "expected_path"),
    [("", None), (" /tmp/checkpoint-7 ", "/tmp/checkpoint-7")],
)
def test_resume_config_resolves_fresh_and_checkpoint_paths(
    resume_from: str,
    expected_path: str | None,
) -> None:
    cfg = OmegaConf.create(
        {"trainer": {"resume_from": resume_from, "resume_strict": False}},
    )

    resolved = resolve_training_resume_config(cfg)

    assert resolved.checkpoint_path == expected_path
    assert resolved.strict is False


def test_loaded_checkpoint_resume_facade_still_clears_warm_start_path(tmp_path) -> None:
    cfg = OmegaConf.create({"model": {"lora": {"path": "/tmp/warm-start"}}})
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload={},
        meta={},
    )

    prepare_model_config_for_training_resume(cfg, checkpoint, strict=False)

    assert cfg.model.lora.path == ""


def test_training_checkpoint_round_trips_trainer_and_owned_state(tmp_path) -> None:
    """Checks training checkpoint round trips trainer and checkpoint-owned state."""
    trainer = _Trainer()
    source = _Bundle()
    with torch.no_grad():
        source.module.weight.fill_(3.0)
    save_training_checkpoint(
        tmp_path / "checkpoint-2",
        trainer=trainer,
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 2, "global_step": 5},
        rng_state={},
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-2")
    restored = _Bundle()
    with torch.no_grad():
        restored.module.weight.fill_(0.0)
    restore_training_checkpoint(
        checkpoint,
        trainer=trainer,
        bundle=restored,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
        strict=True,
    )

    assert (tmp_path / "checkpoint-2" / TRAINING_CHECKPOINT_NAME).exists()
    assert checkpoint.next_epoch == 2
    assert trainer.loaded == {"step": 2, "global_step": 5}
    assert restored.module.weight.item() == pytest.approx(3.0)


def test_restore_model_checkpoint_restores_without_trainer_state(tmp_path) -> None:
    source = _Bundle()
    with torch.no_grad():
        source.module.weight.fill_(4.0)
    path = tmp_path / "checkpoint-model-only"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    restored = _Bundle()
    with torch.no_grad():
        restored.module.weight.zero_()

    restore_model_checkpoint(
        load_training_checkpoint(path),
        bundle=restored,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
        strict=True,
    )

    assert restored.module.weight.item() == pytest.approx(4.0)


def test_restore_model_checkpoint_rejects_identity_before_loading(tmp_path) -> None:
    source = _Bundle()
    path = tmp_path / "checkpoint-model-only"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    restored = _Bundle()
    before = restored.module.weight.detach().clone()

    with pytest.raises(ValueError, match="model identity mismatch"):
        restore_model_checkpoint(
            load_training_checkpoint(path),
            bundle=restored,
            family="unit",
            expected_model_identity={"schema": "wrong/v1"},
            strict=True,
        )

    assert torch.equal(restored.module.weight, before)


def test_restore_training_checkpoint_rejects_family_mismatch(tmp_path) -> None:
    """The explicit runtime family rejects a checkpoint from another family."""
    trainer = _Trainer()
    source = _Bundle()
    save_training_checkpoint(
        tmp_path / "checkpoint-janus",
        trainer=trainer,
        bundle=source,
        family="janus_pro",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1, "global_step": 1},
        rng_state={},
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-janus")
    with pytest.raises(ValueError, match="family mismatch"):
        restore_training_checkpoint(
            checkpoint,
            trainer=trainer,
            bundle=_Bundle(),
            family="wan_2_1",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )


def test_restore_training_checkpoint_strictly_checks_model_identity(tmp_path) -> None:
    identity = {
        "model_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "revision": "pinned",
        "boundary_ratio": 0.9,
        "trainable_transformers": ["transformer", "transformer_2"],
    }
    path = tmp_path / "checkpoint-dual-expert"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="wan_2_1_i2v",
        progress={"next_epoch": 1},
        rng_state={},
        model_identity=identity,
    )
    checkpoint = load_training_checkpoint(path)

    restore_training_checkpoint(
        checkpoint,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="wan_2_1_i2v",
        expected_model_identity=identity,
        strict=True,
    )
    wrong = {**identity, "boundary_ratio": 0.8}
    with pytest.raises(ValueError, match="model identity mismatch"):
        restore_training_checkpoint(
            checkpoint,
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="wan_2_1_i2v",
            expected_model_identity=wrong,
            strict=True,
        )


def test_checkpoint_compatibility_preflight_accepts_matching_identity(tmp_path) -> None:
    identity = {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}}
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "sana",
            "model": {"identity": identity, "owned_state": {}},
        },
        meta={},
    )

    validate_checkpoint_compatibility(
        checkpoint,
        family="sana",
        expected_model_identity=identity,
        strict=True,
    )


def test_checkpoint_meta_preflight_accepts_matching_identity() -> None:
    validate_checkpoint_meta_compatibility(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "sana",
            "model_identity": UNIT_IDENTITY,
        },
        family="sana",
        expected_model_identity=UNIT_IDENTITY,
        strict=True,
    )


def test_checkpoint_meta_preflight_rejects_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="metadata model identity mismatch"):
        validate_checkpoint_meta_compatibility(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "family": "sana",
                "model_identity": UNIT_IDENTITY,
            },
            family="sana",
            expected_model_identity={"schema": "wrong/v1"},
            strict=True,
        )


@pytest.mark.parametrize(
    "meta",
    [
        {"family": "sana", "model_identity": UNIT_IDENTITY},
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_identity": UNIT_IDENTITY,
        },
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "",
            "model_identity": UNIT_IDENTITY,
        },
    ],
)
def test_checkpoint_meta_preflight_rejects_missing_protocol_fields(meta) -> None:
    with pytest.raises((TypeError, ValueError), match=r"schema_version|family"):
        validate_checkpoint_meta_compatibility(
            meta,
            family="sana",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )


def test_checkpoint_compatibility_rejects_schema_v2_without_saved_family(tmp_path) -> None:
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "",
            "model": {"identity": UNIT_IDENTITY, "owned_state": {}},
        },
        meta={},
    )

    with pytest.raises(ValueError, match="family"):
        validate_checkpoint_compatibility(
            checkpoint,
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )


@pytest.mark.parametrize(
    ("family", "identity"),
    [
        ("flux", {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}}),
        ("sana", {"schema": "vrl.model-identity/v1", "sources": {"base": {}}, "build": {}}),
    ],
)
def test_checkpoint_compatibility_preflight_rejects_strict_mismatch(
    tmp_path,
    family,
    identity,
) -> None:
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload={
            "family": "sana",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": {
                "identity": {
                    "schema": "vrl.model-identity/v1",
                    "sources": {},
                    "build": {},
                },
                "owned_state": {},
            },
        },
        meta={},
    )

    with pytest.raises(ValueError, match=r"family mismatch|model identity mismatch"):
        validate_checkpoint_compatibility(
            checkpoint,
            family=family,
            expected_model_identity=identity,
            strict=True,
        )


def test_checkpoint_compatibility_preflight_allows_non_strict_mismatch(tmp_path) -> None:
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload={
            "family": "sana",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": {"identity": {"schema": "saved"}, "owned_state": {}},
        },
        meta={},
    )

    validate_checkpoint_compatibility(
        checkpoint,
        family="flux",
        expected_model_identity={"schema": "other"},
        strict=False,
    )


def test_restore_training_checkpoint_routes_model_load_through_strategy(tmp_path) -> None:
    """Distributed model restore must use the same strategy seam as export."""

    class _SpyStrategy:
        def __init__(self) -> None:
            self.calls = []

        def load_checkpoint_state(self, bundle, state, *, strict=True):
            self.calls.append((bundle, state, strict))
            load_checkpoint_state(bundle, state, strict=strict)

        def export_checkpoint_state(self, bundle):
            return export_checkpoint_state(bundle)

    source = _Bundle()
    save_training_checkpoint(
        tmp_path / "checkpoint-strategy-restore",
        trainer=_Trainer(),
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-strategy-restore")
    trainer = _Trainer()
    trainer._strategy = _SpyStrategy()
    restored = _Bundle()

    restore_training_checkpoint(
        checkpoint,
        trainer=trainer,
        bundle=restored,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
        strict=True,
    )

    assert trainer._strategy.calls == [(restored, checkpoint.checkpoint_state, True)]


def test_save_training_checkpoint_routes_export_through_strategy(tmp_path) -> None:
    """When a strategy is wired, checkpoint-owned state comes from its export.

    Locks sprint P3 ownership: the checkpoint reads the strategy seam, not the
    bundle's raw modules directly, so the future FSDP full-state export controls
    what lands on disk. The spy returns weights the bundle does not hold, so a
    match proves the strategy -- not ``export_checkpoint_state(bundle)`` -- was used.
    """

    class _SpyStrategy:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.context = SimpleNamespace(is_primary=True, world_size=1)

        def export_checkpoint_state(self, bundle):
            self.calls.append(bundle)
            return {"module": {"weight": torch.full((1, 1), 9.0)}}

    strategy = _SpyStrategy()
    bundle = _Bundle()  # module weight is the Linear default, never 9.0
    save_training_checkpoint(
        tmp_path / "checkpoint-strategy",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        strategy=strategy,
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-strategy")
    assert strategy.calls == [bundle]
    assert checkpoint.checkpoint_state["module"]["weight"].item() == pytest.approx(9.0)
    assert checkpoint.trainable_state is checkpoint.checkpoint_state


def test_save_training_checkpoint_non_primary_gathers_but_writes_nothing(tmp_path) -> None:
    """Multi-rank contract: the checkpoint-state export (a COLLECTIVE under FSDP2)
    runs on every rank, but only the primary rank writes files.

    Gating the whole save to rank0 would deadlock FSDP — rank0 waits at the
    all-gather for peers that never call it. A non-primary strategy context MUST
    still invoke the strategy export (joining the collective) yet create no
    checkpoint directory/files. The spy proves the export ran; the empty tmp_path
    proves nothing was written."""

    class _SpyStrategy:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.context = SimpleNamespace(is_primary=False, world_size=1)

        def export_checkpoint_state(self, bundle):
            self.calls.append(bundle)
            return {"module": {"weight": torch.full((1, 1), 9.0)}}

    strategy = _SpyStrategy()
    bundle = _Bundle()
    ckpt_dir = tmp_path / "checkpoint-nonprimary"
    out = save_training_checkpoint(
        ckpt_dir,
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        strategy=strategy,
    )

    assert strategy.calls == [bundle]  # joined the collective gather
    assert out == {}  # no meta returned
    assert not ckpt_dir.exists()  # rank0-only IO: nothing written on a non-primary rank


def test_primary_checkpoint_publication_failure_reraises_after_rank_agreement(
    tmp_path,
) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise RuntimeError("artifact write failed")

    class _Strategy:
        def __init__(self) -> None:
            self.context = SimpleNamespace(is_primary=True, world_size=2)
            self.agreements = []

        def export_checkpoint_state(self, bundle):
            return export_checkpoint_state(bundle)

        def all_ranks_succeeded(self, succeeded):
            self.agreements.append(succeeded)
            return succeeded

    module = _ExportModule(1, 1, bias=False)
    strategy = _Strategy()

    with pytest.raises(RuntimeError, match="artifact write failed"):
        save_training_checkpoint(
            tmp_path / "checkpoint-primary-write-failure",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            strategy=strategy,
        )

    assert strategy.agreements == [True, False, True, True, True, False]
    assert not (tmp_path / "checkpoint-primary-write-failure").exists()
    assert not list(tmp_path.glob("checkpoint-primary-write-failure.tmp-*"))


def test_non_primary_receives_primary_checkpoint_publication_failure(tmp_path) -> None:
    class _Strategy:
        def __init__(self) -> None:
            self.context = SimpleNamespace(is_primary=False, world_size=2)
            self.agreements = []

        def export_checkpoint_state(self, bundle):
            return export_checkpoint_state(bundle)

        def all_ranks_succeeded(self, succeeded):
            self.agreements.append(succeeded)
            if len(self.agreements) == 6:
                return False
            return succeeded

    strategy = _Strategy()

    with pytest.raises(RuntimeError, match="publication failed on the primary"):
        save_training_checkpoint(
            tmp_path / "checkpoint-peer-write-failure",
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            strategy=strategy,
        )

    assert strategy.agreements == [True, False, True, True, True, True]
    assert not (tmp_path / "checkpoint-peer-write-failure").exists()


def test_training_checkpoint_writes_optional_lora_export(tmp_path) -> None:
    """Checks training checkpoint writes optional LoRA export."""

    class _ExportModule(nn.Linear):
        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert state_dict.keys() == {"weight"}
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    export_module = _ExportModule(1, 1, bias=False)
    save_training_checkpoint(
        tmp_path / "checkpoint-1",
        trainer=_Trainer(),
        bundle=_Bundle(export_module),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(export_module)},
    )

    assert (tmp_path / "checkpoint-1" / TRAINING_CHECKPOINT_NAME).exists()
    assert (tmp_path / "checkpoint-1" / LORA_WEIGHTS_NAME / "adapter_model.safetensors").exists()
    assert (tmp_path / "checkpoint-1" / CHECKPOINT_META_NAME).exists()


def test_adapter_export_accepts_safe_namespaced_path(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, path, *, state_dict, selected_adapters):
            del state_dict, selected_adapters
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    module = _ExportModule(1, 1, bias=False)
    artifact_name = f"{LORA_WEIGHTS_NAME}/transformer"

    save_training_checkpoint(
        tmp_path / "checkpoint-namespaced",
        trainer=_Trainer(),
        bundle=_Bundle(module),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={artifact_name: AdapterExport(module)},
    )

    assert (
        tmp_path / "checkpoint-namespaced" / artifact_name / "adapter_model.safetensors"
    ).is_file()


def test_adapter_export_accepts_safe_named_adapter(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.peft_config = {"publish": object()}

        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert state_dict.keys() == {"weight"}
            assert selected_adapters == ["publish"]
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    module = _ExportModule()
    save_training_checkpoint(
        tmp_path / "checkpoint-named-adapter",
        trainer=_Trainer(),
        bundle=_Bundle(module),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={
            LORA_WEIGHTS_NAME: AdapterExport(module, adapter_name="publish"),
        },
    )

    assert (
        tmp_path / "checkpoint-named-adapter" / LORA_WEIGHTS_NAME / "adapter_model.safetensors"
    ).is_file()


@pytest.mark.parametrize(
    "adapter_name",
    [
        "",
        " ",
        "/absolute",
        "C:/windows",
        "a\\windows",
        "a/b",
        ".",
        "..",
    ],
)
def test_adapter_export_rejects_unsafe_adapter_name(adapter_name) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.peft_config = {adapter_name: object()}

        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("unsafe adapter name must fail at construction")

    with pytest.raises(ValueError, match=r"trimmed|string|single path segment"):
        AdapterExport(_ExportModule(), adapter_name=adapter_name)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "/absolute",
        "C:/windows",
        "a\\windows",
        "a//b",
        "a/",
        "a/./b",
        "a/../b",
    ],
)
def test_adapter_export_rejects_unsafe_output_path(tmp_path, artifact_name) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("unsafe artifact path must fail before IO")

    module = _ExportModule(1, 1, bias=False)

    with pytest.raises(ValueError, match="safe relative path"):
        save_training_checkpoint(
            tmp_path / "checkpoint-unsafe",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={artifact_name: AdapterExport(module)},
        )

    assert not (tmp_path / "checkpoint-unsafe").exists()


def test_adapter_export_rejects_duplicate_normalized_output_path(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("duplicate artifact path must fail before IO")

    class _DuplicateItems(dict):
        def items(self):
            export = next(iter(self.values()))
            return iter([("lora_weights/model", export), ("lora_weights/model", export)])

    module = _ExportModule(1, 1, bias=False)
    exports = _DuplicateItems({"lora_weights/model": AdapterExport(module)})

    with pytest.raises(ValueError, match="same PEFT output path"):
        save_training_checkpoint(
            tmp_path / "checkpoint-duplicate",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports=exports,
        )


def test_adapter_export_rejects_ancestor_output_paths(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("overlapping artifact paths must fail before IO")

    module = _ExportModule(1, 1, bias=False)

    with pytest.raises(ValueError, match=r"paths .* overlap"):
        save_training_checkpoint(
            tmp_path / "checkpoint-overlap",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={
                "lora_weights": AdapterExport(module),
                "lora_weights/expert": AdapterExport(module),
            },
        )

    assert not (tmp_path / "checkpoint-overlap").exists()


def test_adapter_export_rejects_custom_adapter_effective_path_collision(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.peft_config = {"default": object(), "expert": object()}

        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("colliding PEFT outputs must fail before IO")

    module = _ExportModule()

    with pytest.raises(ValueError, match="same PEFT output path"):
        save_training_checkpoint(
            tmp_path / "checkpoint-effective-collision",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={
                "lora_weights": AdapterExport(module, adapter_name="expert"),
                "lora_weights/expert": AdapterExport(module),
            },
        )

    assert not (tmp_path / "checkpoint-effective-collision").exists()


def test_training_checkpoint_exports_lora_with_ema_without_mutating_resume_state(
    tmp_path,
) -> None:
    """Checks training checkpoint exports LoRA with EMA without mutating resume state."""

    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.saved_weight = None

        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            self.saved_weight = float(state_dict["weight"].item())
            (path / "adapter_model.safetensors").write_text(str(self.saved_weight))

    class _EMA:
        has_updates = True

        def copy_ema_to(self, parameters, *, store_temp=True):
            self.parameters = list(parameters)
            assert store_temp is True
            self.temp = [p.detach().clone() for p in self.parameters]
            with torch.no_grad():
                for p in self.parameters:
                    p.fill_(7.0)

        def copy_temp_to(self, parameters):
            with torch.no_grad():
                for p, temp in zip(parameters, self.temp, strict=True):
                    p.copy_(temp)

    export_module = _ExportModule()
    bundle = _Bundle(export_module)
    with torch.no_grad():
        bundle.module.weight.fill_(3.0)

    save_training_checkpoint(
        tmp_path / "checkpoint-ema",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(export_module)},
        export_ema=_EMA(),
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-ema")
    saved_trainable = checkpoint.checkpoint_state["module"]["weight"].item()

    assert saved_trainable == pytest.approx(3.0)
    assert export_module.saved_weight == pytest.approx(7.0)
    assert bundle.module.weight.item() == pytest.approx(3.0)


def test_training_checkpoint_skips_lora_ema_export_before_first_ema_update(
    tmp_path,
) -> None:
    """Checks training checkpoint skips LoRA EMA export before first EMA update."""

    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.saved_weight = None

        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            self.saved_weight = float(state_dict["weight"].item())
            (path / "adapter_model.safetensors").write_text(str(self.saved_weight))

    class _UnupdatedEMA:
        has_updates = False

        def __init__(self) -> None:
            self.copy_called = False

        def copy_ema_to(self, parameters, *, store_temp=True):
            del parameters, store_temp
            self.copy_called = True

        def copy_temp_to(self, parameters):
            del parameters
            raise AssertionError("raw export must not restore an unused EMA swap")

    export_module = _ExportModule()
    bundle = _Bundle(export_module)
    with torch.no_grad():
        bundle.module.weight.fill_(3.0)
    ema = _UnupdatedEMA()

    save_training_checkpoint(
        tmp_path / "checkpoint-raw-export",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(export_module)},
        export_ema=ema,
    )

    assert export_module.saved_weight == pytest.approx(3.0)
    assert ema.copy_called is False
    assert bundle.module.weight.item() == pytest.approx(3.0)


@pytest.mark.parametrize(
    "ema",
    [
        object(),
        type(
            "_CallableHasUpdates",
            (),
            {
                "has_updates": lambda self: True,
                "copy_ema_to": lambda self, *_args, **_kwargs: None,
                "copy_temp_to": lambda self, *_args, **_kwargs: None,
            },
        )(),
    ],
)
def test_training_checkpoint_requires_bool_ema_update_protocol(tmp_path, ema) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("invalid EMA protocol must fail before artifact IO")

    module = _ExportModule(1, 1, bias=False)

    with pytest.raises(TypeError, match=r"(has_updates.*bool|bool has_updates)"):
        save_training_checkpoint(
            tmp_path / "checkpoint-invalid-ema",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=ema,
        )


def test_training_checkpoint_restores_raw_weights_when_ema_gather_fails(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("failed gather must not reach artifact IO")

    class _FailSecondGather:
        def __init__(self) -> None:
            self.calls = 0
            self.context = SimpleNamespace(is_primary=True, world_size=1)

        def export_checkpoint_state(self, bundle):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("EMA gather failed")
            return export_checkpoint_state(bundle)

    class _EMA:
        has_updates = True

        def copy_ema_to(self, parameters, *, store_temp=True):
            assert store_temp is True
            self.temp = [parameter.detach().clone() for parameter in parameters]
            with torch.no_grad():
                for parameter in parameters:
                    parameter.fill_(7.0)

        def copy_temp_to(self, parameters):
            with torch.no_grad():
                for parameter, temp in zip(parameters, self.temp, strict=True):
                    parameter.copy_(temp)

    module = _ExportModule(1, 1, bias=False)
    bundle = _Bundle(module)
    with torch.no_grad():
        module.weight.fill_(3.0)
    strategy = _FailSecondGather()

    with pytest.raises(RuntimeError, match="EMA gather failed"):
        save_training_checkpoint(
            tmp_path / "checkpoint-failed-ema-gather",
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=_EMA(),
            strategy=strategy,
        )

    assert strategy.calls == 2
    assert module.weight.item() == pytest.approx(3.0)
    assert not (tmp_path / "checkpoint-failed-ema-gather").exists()


def test_training_checkpoint_preserves_swap_failure_without_unset_restore(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("failed swap must not reach artifact IO")

    class _FailingEMA:
        has_updates = True

        def copy_ema_to(self, parameters, *, store_temp=True):
            del parameters, store_temp
            raise RuntimeError("EMA swap failed")

        def copy_temp_to(self, parameters):
            del parameters
            raise AssertionError("no EMA snapshot was established")

    module = _ExportModule(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(3.0)

    with pytest.raises(RuntimeError, match="EMA swap failed"):
        save_training_checkpoint(
            tmp_path / "checkpoint-failed-ema-swap",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=_FailingEMA(),
        )

    assert module.weight.item() == pytest.approx(3.0)
    assert not (tmp_path / "checkpoint-failed-ema-swap").exists()


def test_peer_ema_swap_failure_rolls_back_before_second_export(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("peer failure must not reach artifact IO")

    class _AgreementStrategy:
        def __init__(self) -> None:
            self.context = SimpleNamespace(is_primary=True, world_size=2)
            self.exported_weights = []
            self.agreements = []

        def export_checkpoint_state(self, bundle):
            state = export_checkpoint_state(bundle)
            self.exported_weights.append(float(state["module"]["weight"].item()))
            return state

        def all_ranks_succeeded(self, succeeded):
            self.agreements.append(succeeded)
            if len(self.agreements) == 9:
                return False
            return succeeded

    class _EMA:
        has_updates = True

        def copy_ema_to(self, parameters, *, store_temp=True):
            assert store_temp is True
            self.temp = [parameter.detach().clone() for parameter in parameters]
            with torch.no_grad():
                for parameter in parameters:
                    parameter.fill_(7.0)

        def copy_temp_to(self, parameters):
            with torch.no_grad():
                for parameter, temp in zip(parameters, self.temp, strict=True):
                    parameter.copy_(temp)

    module = _ExportModule(1, 1, bias=False)
    bundle = _Bundle(module)
    with torch.no_grad():
        module.weight.fill_(3.0)
    strategy = _AgreementStrategy()

    with pytest.raises(RuntimeError, match="peer rank failed before"):
        save_training_checkpoint(
            tmp_path / "checkpoint-peer-swap-failure",
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=_EMA(),
            strategy=strategy,
        )

    assert strategy.exported_weights == pytest.approx([3.0])
    assert strategy.agreements == [
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert module.weight.item() == pytest.approx(3.0)
    assert not (tmp_path / "checkpoint-peer-swap-failure").exists()


def test_peer_ema_swap_failure_propagates_local_rollback_failure(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("failed rollback must not reach artifact IO")

    class _AgreementStrategy:
        def __init__(self) -> None:
            self.context = SimpleNamespace(is_primary=True, world_size=2)
            self.agreements = []

        def export_checkpoint_state(self, bundle):
            return export_checkpoint_state(bundle)

        def all_ranks_succeeded(self, succeeded):
            self.agreements.append(succeeded)
            if len(self.agreements) == 9:
                return False
            return succeeded

    class _EMA:
        has_updates = True

        def copy_ema_to(self, parameters, *, store_temp=True):
            assert store_temp is True
            with torch.no_grad():
                for parameter in parameters:
                    parameter.fill_(7.0)

        def copy_temp_to(self, parameters):
            del parameters
            raise RuntimeError("rank-local EMA rollback failed")

    module = _ExportModule(1, 1, bias=False)
    strategy = _AgreementStrategy()

    with pytest.raises(
        RuntimeError,
        match="rollback could not restore raw training weights",
    ) as error:
        save_training_checkpoint(
            tmp_path / "checkpoint-failed-ema-rollback",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=_EMA(),
            strategy=strategy,
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "rank-local EMA rollback failed"
    assert strategy.agreements[-2:] == [True, False]
    assert not (tmp_path / "checkpoint-failed-ema-rollback").exists()


def test_mixed_rank_ema_update_state_fails_before_swap_or_second_export(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("mixed EMA state must fail before artifact IO")

    class _Strategy:
        def __init__(self) -> None:
            self.context = SimpleNamespace(is_primary=True, world_size=2)
            self.export_calls = 0
            self.agreements = []

        def export_checkpoint_state(self, bundle):
            self.export_calls += 1
            return export_checkpoint_state(bundle)

        def all_ranks_succeeded(self, succeeded):
            self.agreements.append(succeeded)
            if len(self.agreements) in {6, 7}:
                return False
            return succeeded

    class _EMA:
        has_updates = True

        def copy_ema_to(self, *_args, **_kwargs):
            raise AssertionError("mixed update state must fail before EMA swap")

        def copy_temp_to(self, *_args, **_kwargs):
            raise AssertionError("mixed update state established no snapshot")

    module = _ExportModule(1, 1, bias=False)
    strategy = _Strategy()

    with pytest.raises(RuntimeError, match="disagree on EMA update state"):
        save_training_checkpoint(
            tmp_path / "checkpoint-mixed-ema-state",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=_EMA(),
            strategy=strategy,
        )

    assert strategy.export_calls == 1
    assert not (tmp_path / "checkpoint-mixed-ema-state").exists()


def test_non_primary_joins_ema_artifact_gather_and_restores_raw_weights(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("non-primary rank must not write artifacts")

    class _GatherStrategy:
        def __init__(self) -> None:
            self.states = []
            self.context = SimpleNamespace(is_primary=False, world_size=1)

        def export_checkpoint_state(self, bundle):
            state = export_checkpoint_state(bundle)
            self.states.append(float(state["module"]["weight"].item()))
            return state

    class _EMA:
        has_updates = True

        def copy_ema_to(self, parameters, *, store_temp=True):
            assert store_temp is True
            self.temp = [parameter.detach().clone() for parameter in parameters]
            with torch.no_grad():
                for parameter in parameters:
                    parameter.fill_(7.0)

        def copy_temp_to(self, parameters):
            with torch.no_grad():
                for parameter, temp in zip(parameters, self.temp, strict=True):
                    parameter.copy_(temp)

    module = _ExportModule(1, 1, bias=False)
    bundle = _Bundle(module)
    with torch.no_grad():
        module.weight.fill_(3.0)
    strategy = _GatherStrategy()

    result = save_training_checkpoint(
        tmp_path / "checkpoint-nonprimary-ema",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
        export_ema=_EMA(),
        strategy=strategy,
    )

    assert result == {}
    assert strategy.states == pytest.approx([3.0, 7.0])
    assert module.weight.item() == pytest.approx(3.0)
    assert not (tmp_path / "checkpoint-nonprimary-ema").exists()


def test_adapter_export_derives_nested_module_state_prefix(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.saved_keys = None

        def save_pretrained(self, path, *, state_dict, selected_adapters):
            self.saved_keys = set(state_dict)
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _ExportModule()

    root = _Root()
    bundle = _Bundle(root)

    save_training_checkpoint(
        tmp_path / "checkpoint-nested-adapter",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={
            LORA_WEIGHTS_NAME: AdapterExport(root.language_model),
        },
    )

    assert root.language_model.saved_keys == {"weight"}


def test_adapter_export_selects_default_and_excludes_frozen_previous(tmp_path) -> None:
    peft = pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")

    from vrl.models.steps.denoise.common.lora import (
        freeze_checkpoint_owned_adapter_params,
    )

    class _Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2, bias=False)

        def forward(self, inputs):
            return self.lin(inputs)

    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["lin"])
    module = peft.get_peft_model(_Base(), config)
    module.add_adapter("previous", config)
    freeze_checkpoint_owned_adapter_params(module, "previous")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if ".default." in name:
                parameter.fill_(3.0)
            elif ".previous." in name:
                parameter.fill_(9.0)
    bundle = _Bundle(module)

    save_training_checkpoint(
        tmp_path / "checkpoint-default-only",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-default-only")
    root_state = checkpoint.checkpoint_state["module"]
    assert any(".previous." in name for name in root_state)
    artifact_dir = tmp_path / "checkpoint-default-only" / LORA_WEIGHTS_NAME
    artifact = safetensors.load_file(str(artifact_dir / "adapter_model.safetensors"))
    assert artifact
    assert all(torch.equal(tensor, torch.full_like(tensor, 3.0)) for tensor in artifact.values())
    assert not (artifact_dir / "previous").exists()


def test_load_training_checkpoint_requires_checkpoint_pt(tmp_path) -> None:
    """Checks load training checkpoint requires checkpoint pt."""
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()

    with pytest.raises(FileNotFoundError, match=TRAINING_CHECKPOINT_NAME):
        load_training_checkpoint(ckpt)


def test_load_training_checkpoint_rejects_bad_schema(tmp_path) -> None:
    """Checks load training checkpoint rejects bad schema."""
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save({"schema_version": 999}, ckpt / TRAINING_CHECKPOINT_NAME)

    with pytest.raises(ValueError, match="schema_version"):
        load_training_checkpoint(ckpt)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing"),
        pytest.param({"schema_version": None}, id="none"),
        pytest.param({"schema_version": True}, id="bool"),
        pytest.param({"schema_version": "2"}, id="string"),
        pytest.param({"schema_version": 2.0}, id="float"),
    ],
)
def test_load_training_checkpoint_rejects_non_integer_schema_version(
    tmp_path,
    payload,
) -> None:
    ckpt = tmp_path / "checkpoint-invalid-schema"
    ckpt.mkdir()
    torch.save(payload, ckpt / TRAINING_CHECKPOINT_NAME)

    with pytest.raises((TypeError, ValueError), match="schema_version"):
        load_training_checkpoint(ckpt)


@pytest.mark.parametrize(
    ("schema_version", "model"),
    [
        (1, {"trainable_modules": {}}),
        (
            CHECKPOINT_SCHEMA_VERSION,
            {"identity": UNIT_IDENTITY, "owned_state": {}},
        ),
    ],
)
def test_load_training_checkpoint_accepts_exact_integer_schema_versions(
    tmp_path,
    schema_version,
    model,
) -> None:
    ckpt = tmp_path / f"checkpoint-schema-{schema_version}"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": schema_version,
            "family": "unit",
            "trainer": {},
            "model": model,
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )

    assert load_training_checkpoint(ckpt).schema_version == schema_version


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing"),
        pytest.param({"schema_version": None}, id="none"),
        pytest.param({"schema_version": True}, id="bool"),
        pytest.param({"schema_version": "2"}, id="string"),
        pytest.param({"schema_version": 2.0}, id="float"),
    ],
)
def test_training_checkpoint_schema_version_rejects_manual_invalid_payload(
    tmp_path,
    payload,
) -> None:
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload=payload,
        meta={},
    )

    with pytest.raises((TypeError, ValueError), match="schema_version"):
        _ = checkpoint.schema_version


def test_load_training_checkpoint_rejects_schema_v2_without_identity(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-no-identity"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "unit",
            "trainer": {},
            "model": {"owned_state": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )

    with pytest.raises(ValueError, match="model keys mismatch"):
        load_training_checkpoint(ckpt)


@pytest.mark.parametrize("family", [None, "", "   "])
def test_load_training_checkpoint_rejects_schema_v2_without_family(
    tmp_path,
    family,
) -> None:
    ckpt = tmp_path / "checkpoint-no-family"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": family,
            "trainer": {},
            "model": {"identity": UNIT_IDENTITY, "owned_state": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )

    with pytest.raises(ValueError, match="family"):
        load_training_checkpoint(ckpt)


def test_save_training_checkpoint_requires_non_empty_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="model_identity"):
        save_training_checkpoint(
            tmp_path / "checkpoint-no-identity",
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            progress={"next_epoch": 1},
            model_identity={},
        )


@pytest.mark.parametrize("family", [None, "", "   ", " unit "])
def test_save_training_checkpoint_requires_canonical_family(tmp_path, family) -> None:
    with pytest.raises(ValueError, match="family"):
        save_training_checkpoint(
            tmp_path / "checkpoint-no-family",
            trainer=_Trainer(),
            bundle=_Bundle(),
            family=family,
            progress={"next_epoch": 1},
            model_identity=UNIT_IDENTITY,
        )
    assert not (tmp_path / "checkpoint-no-family").exists()


def test_load_checkpoint_state_strict_rejects_key_mismatch() -> None:
    """Checks load checkpoint state strict rejects key mismatch."""
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint_state(_Bundle(), {}, strict=True)


def test_infer_next_epoch_falls_back_to_trainer_step_for_checkpoint_final(tmp_path) -> None:
    """Checks infer next epoch falls back to trainer step for checkpoint final."""
    ckpt = tmp_path / "checkpoint-final"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {"step": 12}, None) == 12


def test_infer_next_epoch_falls_back_to_numeric_checkpoint_suffix(tmp_path) -> None:
    """Checks infer next epoch falls back to numeric checkpoint suffix."""
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {}, {}) == 42


def test_load_training_checkpoint_rejects_non_object_meta(tmp_path) -> None:
    """Checks load training checkpoint rejects non object meta."""
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "trainer": {},
            "model": {"trainable_modules": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )
    (ckpt / CHECKPOINT_META_NAME).write_text(json.dumps([{"next_epoch": 1}]))

    with pytest.raises(TypeError, match="JSON object"):
        load_training_checkpoint(ckpt)


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"step": 2, "global_step": 5}

    def load_state_dict(self, state, *, strict=True):
        del strict
        self.loaded = dict(state)


class _Bundle:
    def __init__(self, module=None) -> None:
        import torch.nn as nn

        self.module = module or nn.Linear(1, 1, bias=False)
        self.model = self.module
        self.trainable_modules = {"module": self.module}


class _OwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.frozen_base = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.register_buffer("previous", torch.tensor([3.0]))
        register_checkpoint_owned_state(self, ["previous"])


class _OwnedBundle:
    def __init__(self) -> None:
        self.module = _OwnedModule()
        self.trainable_modules = {"module": self.module}


def _training_checkpoint(tmp_path, payload) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload=payload,
        meta={},
    )


def _v1_payload(
    state: dict[str, torch.Tensor],
    *,
    identity: dict | None = None,
) -> dict:
    model = {"trainable_modules": {"module": state}}
    if identity is not None:
        model["identity"] = identity
    return {
        "schema_version": 1,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": model,
        "progress": {"next_epoch": 2},
        "rng": {},
    }


def test_checkpoint_export_is_exact_owned_clone_and_unwraps_compile() -> None:
    bundle = _OwnedBundle()
    compiled = torch.compile(bundle.module)
    bundle.trainable_modules["module"] = compiled

    snapshot = export_checkpoint_state(bundle)["module"]

    assert set(snapshot) == {"weight", "previous"}
    assert snapshot["weight"].data_ptr() != bundle.module.weight.data_ptr()
    assert snapshot["previous"].data_ptr() != bundle.module.previous.data_ptr()
    assert all("_orig_mod" not in name for name in snapshot)
    before = {name: value.clone() for name, value in snapshot.items()}
    with torch.no_grad():
        bundle.module.weight.add_(10)
        bundle.module.previous.add_(10)
    assert all(torch.equal(snapshot[name], before[name]) for name in snapshot)


def test_checkpoint_export_rejects_module_without_owned_state() -> None:
    module = nn.Linear(1, 1)
    module.requires_grad_(False)
    bundle = type("_FrozenBundle", (), {"trainable_modules": {"module": module}})()

    with pytest.raises(ValueError, match="no checkpoint-owned state"):
        export_checkpoint_state(bundle)


def test_strict_schema_v2_restore_requires_exact_owned_keys(tmp_path) -> None:
    bundle = _OwnedBundle()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0], dtype=torch.float64),
                    "previous": torch.tensor([8.0], dtype=torch.float64),
                },
            },
        },
        "progress": {"next_epoch": 2},
        "rng": {},
    }

    restore_training_checkpoint(
        _training_checkpoint(tmp_path, payload),
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
        strict=True,
    )

    assert bundle.module.weight.item() == pytest.approx(7.0)
    assert bundle.module.previous.item() == pytest.approx(8.0)
    assert bundle.module.frozen_base.item() == pytest.approx(2.0)
    assert bundle.module.weight.dtype == torch.float32
    assert bundle.module.previous.dtype == torch.float32


@pytest.mark.parametrize("strict", [True, False])
def test_schema_v2_wrong_shape_rejects_all_roots_before_mutation(
    tmp_path,
    strict,
) -> None:
    bundle = _OwnedBundle()
    bundle.second = _OwnedModule()
    bundle.trainable_modules["second"] = bundle.second
    before = {
        root_name: {name: value.clone() for name, value in module.state_dict().items()}
        for root_name, module in bundle.trainable_modules.items()
    }
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "previous": torch.tensor([8.0]),
                },
                "second": {
                    "weight": torch.tensor([9.0]),
                    "previous": torch.tensor([10.0, 11.0]),
                },
            },
        },
        "progress": {},
        "rng": {},
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, payload),
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
            strict=strict,
        )

    assert all(
        torch.equal(value, before[root_name][name])
        for root_name, module in bundle.trainable_modules.items()
        for name, value in module.state_dict().items()
    )


def test_schema_v2_non_tensor_owned_value_rejects_before_mutation(tmp_path) -> None:
    bundle = _OwnedBundle()
    before = {name: value.clone() for name, value in bundle.module.state_dict().items()}
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "previous": 8.0,
                },
            },
        },
        "progress": {},
        "rng": {},
    }

    with pytest.raises(TypeError, match="must be a tensor"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, payload),
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )

    assert all(
        torch.equal(value, before[name]) for name, value in bundle.module.state_dict().items()
    )


@pytest.mark.parametrize(
    "owned_state",
    [
        {"module": {"weight": torch.tensor([7.0])}},
        {
            "module": {
                "weight": torch.tensor([7.0]),
                "previous": torch.tensor([8.0]),
                "frozen_base": torch.tensor([9.0]),
            },
        },
        {
            "module": {
                "weight": torch.tensor([7.0]),
                "previous": torch.tensor([8.0]),
            },
            "extra": {},
        },
    ],
)
def test_strict_schema_v2_restore_rejects_missing_extra_keys_and_roots(
    tmp_path,
    owned_state,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": {"identity": UNIT_IDENTITY, "owned_state": owned_state},
        "progress": {},
        "rng": {},
    }

    with pytest.raises(ValueError, match=r"keys mismatch|roots mismatch"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, payload),
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )


def test_strict_schema_v1_full_state_restores_without_identity(tmp_path) -> None:
    source = _OwnedBundle()
    with torch.no_grad():
        source.module.weight.fill_(7.0)
        source.module.frozen_base.fill_(9.0)
        source.module.previous.fill_(8.0)
    restored = _OwnedBundle()

    restore_training_checkpoint(
        _training_checkpoint(tmp_path, _v1_payload(source.module.state_dict())),
        trainer=_Trainer(),
        bundle=restored,
        family="unit",
        strict=True,
    )

    assert restored.module.weight.item() == pytest.approx(7.0)
    assert restored.module.previous.item() == pytest.approx(8.0)
    assert restored.module.frozen_base.item() == pytest.approx(9.0)


def test_strict_schema_v1_full_wrong_shape_rejects_all_roots_before_mutation(
    tmp_path,
) -> None:
    restored = _OwnedBundle()
    restored.second = _OwnedModule()
    restored.trainable_modules["second"] = restored.second
    before = {
        root_name: {name: value.clone() for name, value in module.state_dict().items()}
        for root_name, module in restored.trainable_modules.items()
    }
    state = {
        "module": {
            "weight": torch.tensor([7.0]),
            "frozen_base": torch.tensor([9.0]),
            "previous": torch.tensor([8.0]),
        },
        "second": {
            "weight": torch.tensor([10.0]),
            "frozen_base": torch.tensor([11.0]),
            "previous": torch.tensor([12.0, 13.0]),
        },
    }
    payload = {
        "schema_version": 1,
        "family": "unit",
        "trainer": {},
        "model": {"trainable_modules": state},
        "progress": {},
        "rng": {},
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, payload),
            trainer=_Trainer(),
            bundle=restored,
            family="unit",
            strict=True,
        )

    assert all(
        torch.equal(value, before[root_name][name])
        for root_name, module in restored.trainable_modules.items()
        for name, value in module.state_dict().items()
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_strict_schema_v1_malformed_full_state_rejects_before_mutation(
    tmp_path,
    mutation,
) -> None:
    source = _OwnedBundle()
    with torch.no_grad():
        source.module.weight.fill_(7.0)
        source.module.frozen_base.fill_(9.0)
        source.module.previous.fill_(8.0)
    state = dict(source.module.state_dict())
    if mutation == "missing":
        state.pop("frozen_base")
    else:
        state["unknown"] = torch.tensor([10.0])
    restored = _OwnedBundle()
    before = {name: value.clone() for name, value in restored.module.state_dict().items()}

    with pytest.raises(ValueError, match=r"verified model identity|keys mismatch"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, _v1_payload(state)),
            trainer=_Trainer(),
            bundle=restored,
            family="unit",
            strict=True,
        )

    assert all(
        torch.equal(value, before[name]) for name, value in restored.module.state_dict().items()
    )


def test_strict_schema_v1_compiled_full_state_normalizes_legacy_prefix(tmp_path) -> None:
    source = _OwnedBundle()
    with torch.no_grad():
        source.module.frozen_base.fill_(9.0)
    legacy_compiled = dict(torch.compile(source.module).state_dict())
    restored = _OwnedBundle()

    restore_training_checkpoint(
        _training_checkpoint(tmp_path, _v1_payload(legacy_compiled)),
        trainer=_Trainer(),
        bundle=restored,
        family="unit",
        strict=True,
    )

    assert torch.equal(restored.module.weight, source.module.weight)
    assert torch.equal(restored.module.previous, source.module.previous)
    assert torch.equal(restored.module.frozen_base, source.module.frozen_base)


def test_strict_schema_v1_rejects_mixed_compile_prefixes(tmp_path) -> None:
    mixed = {
        "_orig_mod.weight": torch.tensor([7.0]),
        "frozen_base": torch.tensor([2.0]),
        "_orig_mod.previous": torch.tensor([8.0]),
    }

    with pytest.raises(ValueError, match="mixes compiled and uncompiled"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, _v1_payload(mixed)),
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            strict=True,
        )


def test_strict_schema_v2_never_normalizes_compile_prefix(tmp_path) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "_orig_mod.weight": torch.tensor([7.0]),
                    "_orig_mod.previous": torch.tensor([8.0]),
                },
            },
        },
        "progress": {},
        "rng": {},
    }

    with pytest.raises(ValueError, match="keys mismatch"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, payload),
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )


def test_strict_schema_v1_selective_state_requires_verified_identity(tmp_path) -> None:
    selective = {
        "weight": torch.tensor([7.0]),
        "previous": torch.tensor([8.0]),
    }

    with pytest.raises(ValueError, match="verified model identity"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, _v1_payload(selective)),
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            strict=True,
        )


def test_strict_schema_v1_selective_state_rejects_missing_registered_state(tmp_path) -> None:
    payload = _v1_payload(
        {"weight": torch.tensor([7.0])},
        identity=UNIT_IDENTITY,
    )

    with pytest.raises(ValueError, match=r"missing=.*previous"):
        restore_training_checkpoint(
            _training_checkpoint(tmp_path, payload),
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
            strict=True,
        )


def test_non_strict_restore_warns_and_loads_matching_owned_state(tmp_path, caplog) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "other",
        "trainer": {"step": 2, "global_step": 5},
        "model": {
            "identity": {"schema": "other"},
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "unknown": torch.tensor([9.0]),
                },
            },
        },
        "progress": {},
        "rng": {},
    }
    bundle = _OwnedBundle()

    restore_training_checkpoint(
        _training_checkpoint(tmp_path, payload),
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
        strict=False,
    )

    assert bundle.module.weight.item() == pytest.approx(7.0)
    assert bundle.module.previous.item() == pytest.approx(3.0)
    assert "Non-strict checkpoint restore" in caplog.text


def test_checkpoint_publish_is_atomic(tmp_path) -> None:
    """A save that dies mid-write leaves no directory a resume would trust."""
    from vrl.trainers.checkpointing import (
        find_latest_complete_checkpoint,
        is_complete_checkpoint,
    )

    class _ExplodingExport(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise RuntimeError("disk full")

    export_module = _ExplodingExport(1, 1, bias=False)
    target = tmp_path / "checkpoint-3"
    with pytest.raises(RuntimeError, match="disk full"):
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=_Bundle(export_module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 3, "global_step": 3},
            rng_state={},
            adapter_exports={"lora_weights": AdapterExport(export_module)},
        )

    # Neither the final directory nor a trusted leftover exists.
    assert not target.exists()
    assert find_latest_complete_checkpoint(tmp_path) is None
    for leftover in tmp_path.iterdir():
        assert not is_complete_checkpoint(leftover)


def test_published_checkpoint_is_complete_and_size_stamped(tmp_path) -> None:
    from vrl.trainers.checkpointing import (
        TRAINING_CHECKPOINT_NAME,
        is_complete_checkpoint,
        read_checkpoint_meta,
    )

    target = tmp_path / "checkpoint-7"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 7, "global_step": 7},
        rng_state={},
    )

    assert is_complete_checkpoint(target)
    meta = read_checkpoint_meta(target)
    assert meta["checkpoint_file_bytes"] == (target / TRAINING_CHECKPOINT_NAME).stat().st_size
    assert meta["model_identity"] == UNIT_IDENTITY
    # Truncation is detected without loading the payload.
    (target / TRAINING_CHECKPOINT_NAME).write_bytes(b"torn")
    assert not is_complete_checkpoint(target)


def test_checkpoint_load_rejects_meta_identity_drift(tmp_path) -> None:
    target = tmp_path / "checkpoint-identity-drift"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    meta_path = target / CHECKPOINT_META_NAME
    meta = json.loads(meta_path.read_text())
    meta["model_identity"] = {"schema": "stale/v1"}
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="model_identity disagrees"):
        load_training_checkpoint(target)


def test_checkpoint_load_rejects_meta_without_schema_version(tmp_path) -> None:
    target = tmp_path / "checkpoint-meta-no-schema"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    meta_path = target / CHECKPOINT_META_NAME
    meta = json.loads(meta_path.read_text())
    meta.pop("schema_version")
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="missing schema_version"):
        load_training_checkpoint(target)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda meta: meta.pop("family"), "family"),
        (lambda meta: meta.__setitem__("family", ""), "family"),
        (lambda meta: meta.__setitem__("family", "other"), "family disagrees"),
    ],
)
def test_checkpoint_load_requires_meta_family_to_match_payload(
    tmp_path,
    mutate,
    message,
) -> None:
    target = tmp_path / "checkpoint-meta-family"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    meta_path = target / CHECKPOINT_META_NAME
    meta = json.loads(meta_path.read_text())
    mutate(meta)
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match=message):
        load_training_checkpoint(target)


def test_find_latest_complete_checkpoint_skips_staging_and_orders_by_step(tmp_path) -> None:
    from vrl.trainers.checkpointing import find_latest_complete_checkpoint

    class _StepTrainer(_Trainer):
        def __init__(self, step: int) -> None:
            super().__init__()
            self._step = step

        def state_dict(self):
            return {"step": self._step, "global_step": self._step}

    for step in (2, 10):
        save_training_checkpoint(
            tmp_path / f"checkpoint-{step}",
            trainer=_StepTrainer(step),
            bundle=_Bundle(),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": step, "global_step": step},
            rng_state={},
        )
    # A staging leftover (crash before publish) and a torn dir are both skipped.
    (tmp_path / "checkpoint-99.tmp-abc").mkdir()
    torn = tmp_path / "checkpoint-50"
    torn.mkdir()
    (torn / "checkpoint.pt").write_bytes(b"torn")

    latest = find_latest_complete_checkpoint(tmp_path)
    assert latest is not None and latest.name == "checkpoint-10"


def test_resave_to_existing_checkpoint_dir_replaces_it(tmp_path) -> None:
    """Crash-loop overwriting the same checkpoint-N republishes cleanly."""
    from vrl.trainers.checkpointing import is_complete_checkpoint

    target = tmp_path / "checkpoint-1"
    for _ in range(2):
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1, "global_step": 1},
            rng_state={},
        )
    assert is_complete_checkpoint(target)
    assert load_training_checkpoint(target).payload["family"] == "unit"


def test_prepare_metrics_csv_rejects_resume_across_schema_change(tmp_path) -> None:
    """Appending new-schema rows under an old header silently shifts columns."""
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    columns = ("epoch", "loss", "approx_kl")

    prepare_metrics_csv(path, columns, resume_at=None)
    prepare_metrics_csv(path, columns, resume_at=("epoch", 0))

    with pytest.raises(ValueError, match="different metrics schema"):
        prepare_metrics_csv(
            path,
            ("epoch", "loss", "active_clip_fraction", "approx_kl"),
            resume_at=("epoch", 0),
        )


@pytest.mark.parametrize(
    "columns",
    [
        "epoch",
        b"epoch",
        (),
        ("epoch", "epoch"),
        ("", "loss"),
        ("bad,column", "loss"),
        ("bad\ncolumn", "loss"),
        (1, "loss"),
    ],
)
def test_prepare_metrics_csv_rejects_invalid_columns(tmp_path, columns) -> None:
    from vrl.trainers.checkpointing import prepare_metrics_csv

    with pytest.raises(ValueError, match="columns"):
        prepare_metrics_csv(tmp_path / "metrics.csv", columns, resume_at=None)


def test_prepare_metrics_csv_discards_rows_not_covered_by_checkpoint(tmp_path) -> None:
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    columns = ("epoch", "loss")
    header = "epoch,loss\n"
    path.write_text(header + "38,1.0\n39,0.9\n40,0.8\n41,0.7\n42,0")

    prepare_metrics_csv(path, columns, resume_at=("epoch", 40))

    assert path.read_text() == header + "38,1.0\n39,0.9\n"


def test_prepare_metrics_csv_rejects_wrong_row_width(tmp_path) -> None:
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    path.write_text("epoch,loss\n0\n")

    with pytest.raises(ValueError, match="has 1 columns; expected 2"):
        prepare_metrics_csv(path, ("epoch", "loss"), resume_at=("epoch", 1))


@pytest.mark.parametrize(
    "rows",
    [
        "0,1.0\n0,0.9\n",
        "1,1.0\n0,0.9\n",
        "0.5,1.0\n",
        "-1,1.0\n",
    ],
)
def test_prepare_metrics_csv_rejects_invalid_resume_positions(tmp_path, rows) -> None:
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    columns = ("epoch", "loss")
    header = "epoch,loss\n"
    path.write_text(header + rows)

    with pytest.raises(ValueError, match=r"integer|strictly increasing"):
        prepare_metrics_csv(path, columns, resume_at=("epoch", 2))
