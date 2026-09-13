from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.trainers._checkpoint_helpers import (
    UNIT_IDENTITY,
    _OwnedBundle,
    _OwnedModule,
    _Trainer,
    _training_checkpoint,
    _v1_payload,
)
from vrl.trainers.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
    export_checkpoint_state,
    restore_training_checkpoint,
    validate_checkpoint_compatibility,
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
        TrainingCheckpoint.load(ckpt)


def test_load_training_checkpoint_rejects_schema_v2_without_family(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-no-family"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "",
            "trainer": {},
            "model": {"identity": UNIT_IDENTITY, "owned_state": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )

    with pytest.raises(ValueError, match="family"):
        TrainingCheckpoint.load(ckpt)


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


def test_schema_v2_wrong_shape_rejects_all_roots_before_mutation(tmp_path) -> None:
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
            strict=True,
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
