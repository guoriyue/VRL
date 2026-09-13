"""The Wan DPO entrypoint trains the registry-built model under the config it was
given: family aliases resolve, non-T2V families are refused before any side
effect, and ``actor.gradient_checkpointing`` reaches the real transformer."""

from __future__ import annotations

import csv
import math

import pytest

from tests.scripts._wan_dpo_helpers import install_local_pickapic, spy_wan_from_build, tiny_dpo_run
from vrl.scripts.families.wan_2_1.train_dpo import train_wan_2_1_dpo
from vrl.trainers.activation_checkpointing import selective_checkpoint_func
from vrl.trainers.checkpointing import read_checkpoint_meta


@pytest.mark.parametrize("family", ["wan_2_1", "wan"])
def test_offline_dpo_trains_the_registry_model_end_to_end(monkeypatch, tmp_path, family) -> None:
    """One real DPO step on the tiny snapshot: the alias resolves to ``wan_2_1``,
    the loss is finite, and both checkpoints carry the canonical family."""

    install_local_pickapic(monkeypatch)
    cfg = tiny_dpo_run(tmp_path, family=family)

    train_wan_2_1_dpo(cfg)

    run = tmp_path / "run"
    with (run / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0"]
    assert math.isfinite(float(rows[0]["loss"]))
    for name in ("checkpoint-1", "checkpoint-final"):
        meta = read_checkpoint_meta(run / name)
        assert meta["family"] == "wan_2_1"
        assert meta["uses_lora"] is False
        assert meta["next_step"] == 1


@pytest.mark.parametrize("family", ["wan_2_1_i2v", "sd3_5"])
def test_offline_dpo_rejects_non_t2v_wan_family_before_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    family: str,
) -> None:
    cfg = tiny_dpo_run(tmp_path, family=family)
    if family == "sd3_5":
        del cfg.sampling.num_frames
    dataset_calls = install_local_pickapic(monkeypatch)
    built = spy_wan_from_build(monkeypatch)

    with pytest.raises(
        ValueError,
        match=rf"Wan Diffusion-DPO requires model\.family='wan_2_1'.*got '{family}'",
    ):
        train_wan_2_1_dpo(cfg)

    assert built == []
    assert dataset_calls == []


@pytest.mark.parametrize(
    ("checkpointing", "expected_mode"),
    [
        ('"off"', "off"),
        ("true", "full"),
        ('"selective"', "selective"),
    ],
)
def test_offline_dpo_uses_shared_gradient_checkpointing_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    checkpointing: str,
    expected_mode: str,
) -> None:
    """The policy lands on the real transformer: off leaves it untouched, full
    enables diffusers' default recompute, selective installs the SAC function."""

    install_local_pickapic(monkeypatch)
    cfg = tiny_dpo_run(tmp_path, overrides=[f"actor.gradient_checkpointing={checkpointing}"])
    built = spy_wan_from_build(monkeypatch)

    train_wan_2_1_dpo(cfg)

    (model,) = built
    transformer = model.transformer
    assert transformer.is_gradient_checkpointing is (expected_mode != "off")
    funcs = {
        block._gradient_checkpointing_func
        for block in transformer.modules()
        if getattr(block, "gradient_checkpointing", False)
    }
    if expected_mode == "selective":
        assert funcs == {selective_checkpoint_func}
    elif expected_mode == "full":
        assert funcs and selective_checkpoint_func not in funcs
