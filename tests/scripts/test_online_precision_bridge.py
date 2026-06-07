"""Integration: the unified precision policy drives the online trainer (P1)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.scripts.common.online import (
    _apply_precision_policy,
    _prepare_metrics_csv,
    _write_metric_row,
)
from vrl.trainers.precision import torch_dtype_for_trainer_precision

_RECIPES = [
    "diffusion/sd3_5/online_grpo_ocr",          # fp32
    "diffusion/sd3_5/online_grpo_geneval",      # fp32
    "diffusion/wan_2_1/online_grpo_ocr",        # bf16
    "ar/janus_pro/online_grpo_ocr",             # bf16
]


@pytest.mark.parametrize("experiment", _RECIPES)
def test_bridge_preserves_legacy_dtype(experiment):
    """Checks bridge preserves legacy dtype."""
    cfg = load_config(f"experiment/{experiment}")
    trainer_config = build_configs(cfg)["trainer"]
    legacy = torch_dtype_for_trainer_precision(trainer_config, torch)
    _apply_precision_policy(cfg, trainer_config)
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is legacy


def test_precision_block_drives_trainer():
    # An fp32 recipe + a top-level `precision: bf16` must flip the trainer to bf16.
    """Checks precision block drives trainer."""
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    assert torch_dtype_for_trainer_precision(build_configs(cfg)["trainer"], torch) is torch.float32

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": "bf16"}))
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)
    assert trainer_config.mixed_precision == "bf16"
    assert trainer_config.bf16 is True
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is torch.bfloat16


def _with_precision(experiment, block):
    cfg = load_config(f"experiment/{experiment}")
    return OmegaConf.merge(cfg, OmegaConf.create({"precision": block}))


def test_decoupled_rollout_compute():
    # P3: bf16 replay / fp32 rollout — compute drives the trainer, rollout is
    # a separate generation dtype (no longer refused).
    """Checks decoupled rollout compute."""
    cfg = _with_precision(
        "diffusion/sd3_5/online_grpo_ocr", {"compute": "bf16", "rollout": "fp32"},
    )
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)  # must not raise
    assert trainer_config.mixed_precision == "bf16"
    assert resolve_precision_policy(cfg).rollout == "fp32"
    # The bridge exposes the rollout dtype so the precision drift guard can compare
    # it against compute (here bf16 compute / fp32 rollout = a mismatch it watches).
    assert trainer_config.rollout_precision == "fp32"


@pytest.mark.parametrize("math,expected", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_math_axis_resolves_to_dtype(math, expected):
    # P2: the `math` axis resolves to the evaluator's log-prob math dtype.
    """Checks math axis resolves to dtype."""
    from vrl.config.precision import resolve_precision_policy
    from vrl.models.dtypes import resolve_torch_dtype

    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"compute": "fp32", "math": math})
    assert resolve_torch_dtype(resolve_precision_policy(cfg).math) is expected
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)
    assert trainer_config.math_precision == math


def test_precision_drift_guard_config_is_bridged_from_yaml():
    """Checks precision drift guard YAML config reaches TrainerConfig."""
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "trainer": {
                    "precision_drift_guard": {
                        "mode": "fail",
                        "max_abs_log_ratio": 0.02,
                    },
                },
            },
        ),
    )

    trainer_config = build_configs(cfg)["trainer"]

    assert trainer_config.precision_drift_guard.mode == "fail"
    assert trainer_config.precision_drift_guard.max_abs_log_ratio == pytest.approx(0.02)


def test_online_metrics_csv_includes_logprob_mismatch_metrics(tmp_path):
    """Checks rollout-vs-replay mismatch metrics are written as regular CSV columns."""
    from types import SimpleNamespace

    from vrl.algorithms.types import TrainStepMetrics

    csv_path = tmp_path / "metrics.csv"
    _prepare_metrics_csv(
        csv_path,
        (),
        resume=False,
        prepare_metrics_csv=lambda path, header, *, resume: path.write_text(header),
    )
    _write_metric_row(
        csv_path,
        0,
        TrainStepMetrics(
            loss=1.0,
            policy_loss=2.0,
            logprob_abs_diff_mean=0.1,
            logprob_abs_diff_max=0.2,
            ratio_abs_dev_mean=0.3,
            ratio_abs_dev_max=0.4,
            mismatch_kl=-0.5,
            mismatch_k3_kl=0.6,
        ),
        reward_fn=SimpleNamespace(last_components={}),
        component_names=(),
        metric_row_hook=None,
    )

    header, row = csv_path.read_text().splitlines()
    assert "logprob_abs_diff_mean" in header
    assert "ratio_abs_dev_max" in header
    assert "mismatch_k3_kl" in header
    values = dict(zip(header.split(","), row.split(","), strict=True))
    assert values["logprob_abs_diff_mean"] == "0.100000"
    assert values["ratio_abs_dev_max"] == "0.400000"
    assert values["mismatch_kl"] == "-0.500000"


@pytest.mark.parametrize(
    "frozen,expected", [("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_frozen_axis_in_runtime_spec(frozen, expected):
    # P1b: the frozen axis rides the spec as a real torch.dtype (like `dtype`).
    """Checks frozen axis in runtime spec."""
    from vrl.models.diffusion.sd3_5.runtime import extract_sd3_5_runtime_spec

    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"compute": "fp32", "frozen": frozen})
    spec = extract_sd3_5_runtime_spec(cfg, torch.device("cpu"), torch.float32)
    assert spec.frozen_dtype is expected
