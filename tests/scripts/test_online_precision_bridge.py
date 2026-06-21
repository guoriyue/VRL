"""Integration: the unified precision policy drives the online trainer (P1)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.config.test_load_all_experiments import _experiment_names
from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.models.dtypes import resolve_torch_dtype
from vrl.scripts.common.online import _apply_precision_policy
from vrl.trainers.core.types import PrecisionDriftGuardConfig
from vrl.trainers.precision import torch_dtype_for_trainer_precision

# Derive every online recipe from the experiment glob (the single source of
# truth in test_load_all_experiments) rather than hand-maintaining a subset —
# the old hand list had drifted and only exercised four recipes. "online" ==
# every experiment whose final path component is not an `offline_*` recipe.
_RECIPES = [
    name
    for name in _experiment_names()
    if not Path(name).name.startswith("offline_")
]


@pytest.mark.parametrize("experiment", _RECIPES)
def test_bridge_uses_aligned_public_precision(experiment):
    """Checks bridge derives trainer precision from public precision."""
    cfg = load_config(f"experiment/{experiment}")
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)
    policy = resolve_precision_policy(cfg)
    assert trainer_config.train_precision == policy.train
    assert trainer_config.rollout_precision == policy.rollout
    assert policy.train == policy.rollout
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is resolve_torch_dtype(
        policy.train,
    )


def test_precision_block_drives_trainer():
    """Checks precision block drives trainer."""
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_geneval")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": "fp32"}))
    assert torch_dtype_for_trainer_precision(build_configs(cfg)["trainer"], torch) is torch.float32

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": "bf16"}))
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)
    assert trainer_config.train_precision == "bf16"
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is torch.bfloat16


def test_fp16_precision_block_drives_trainer_and_rollout():
    """Checks scalar fp16 drives both replay compute and rollout precision."""
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": "fp16"}))

    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)

    assert trainer_config.train_precision == "fp16"
    assert trainer_config.rollout_precision == "fp16"
    assert trainer_config.math_precision == "fp32"
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is torch.float16


def test_rollout_precision_split_auto_derives_correction_policy():
    """A low-precision rollout split is a user intent; correction is derived."""
    cfg = _with_precision(
        "diffusion/sd3_5/online_grpo_ocr",
        {"train": "bf16", "rollout": "fp8"},
    )

    trainer_config = build_configs(cfg)["trainer"]

    assert trainer_config.train_precision == "bf16"
    assert trainer_config.rollout_precision == "fp8"
    # The auto split-precision policy is whatever the builder helper installs;
    # assert the whole struct equals that single source, not a per-field copy of
    # its constants (which would falsely fail on any retune of the policy).
    assert trainer_config.precision_correction == PrecisionCorrectionConfig(
        tis_mode="truncate",
        rs_mode="seq_mean_k1",
    )
    assert trainer_config.precision_drift_guard == PrecisionDriftGuardConfig(
        mode="fail",
        max_abs_log_ratio=math.log(10.0),
        max_ratio_abs_dev=9.0,
        fail_on_nonfinite=True,
    )


def test_no_split_means_no_auto_correction_policy() -> None:
    """rollout == train: the builder early-returns and installs no auto policy."""
    cfg = _with_precision(
        "diffusion/sd3_5/online_grpo_ocr",
        {"train": "bf16", "rollout": "bf16"},
    )

    trainer_config = build_configs(cfg)["trainer"]

    # No split -> the correction/guard fields keep their dataclass defaults; the
    # split-only policy (TIS truncate / drift_guard mode="fail") is NOT installed.
    assert trainer_config.precision_correction == PrecisionCorrectionConfig()
    assert trainer_config.precision_drift_guard == PrecisionDriftGuardConfig()


def test_explicit_precision_correction_is_respected_on_rollout_split():
    """Expert correction blocks override the auto split-precision defaults."""
    cfg = _with_precision(
        "diffusion/sd3_5/online_grpo_ocr",
        {"train": "bf16", "rollout": "fp8"},
    )
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "trainer": {
                    "precision_correction": {"tis_mode": "off", "rs_mode": "off"},
                    "precision_drift_guard": {"mode": "warn", "max_abs_log_ratio": 0.25},
                },
            },
        ),
    )

    trainer_config = build_configs(cfg)["trainer"]

    assert trainer_config.precision_correction.tis_mode == "off"
    assert trainer_config.precision_correction.rs_mode == "off"
    assert trainer_config.precision_drift_guard.mode == "warn"
    assert trainer_config.precision_drift_guard.max_abs_log_ratio == pytest.approx(0.25)


def _with_precision(experiment, block):
    cfg = load_config(f"experiment/{experiment}")
    return OmegaConf.merge(cfg, OmegaConf.create({"precision": block}))


@pytest.mark.parametrize("math,expected", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_math_axis_resolves_to_dtype(math, expected):
    # P2: the `math` axis resolves to the evaluator's log-prob math dtype.
    """Checks math axis resolves to dtype."""
    from vrl.config.precision import resolve_precision_policy
    from vrl.models.dtypes import resolve_torch_dtype

    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"train": "fp32", "math": math})
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
    """Mismatch + continuous-async diagnostics are written as regular CSV columns."""
    from types import SimpleNamespace

    from vrl.algorithms.types import TrainStepMetrics
    from vrl.scripts.common.online import OnlineRecipeRun

    csv_path = tmp_path / "metrics.csv"
    # The metrics-CSV side effects now live on OnlineRecipeRun; the controller
    # reads component_names / reward_fn / metric_row_hook off its stack, so a
    # minimal SimpleNamespace stack is enough to exercise the row formatting.
    run = OnlineRecipeRun(
        stack=SimpleNamespace(
            component_names=(),
            reward_fn=SimpleNamespace(last_components={}),
            definition=SimpleNamespace(metric_row_hook=None),
        ),
        csv_path=csv_path,
        eval_csv_path=tmp_path / "eval_metrics.csv",
        rng=None,
        resume=False,
    )
    run.prepare_metrics_csv()
    run.write_metric_row(
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
            phase_times={
                "continuous.stale_policy_versions": 1.0,
                "continuous.queue_ready_groups": 3.0,
                "continuous.weight_sync_pause_s": 0.25,
                "continuous.predicted_admit_staleness": 2.0,
                "continuous.admit_blocked_on_staleness": 1.0,
            },
        ),
    )

    header, row = csv_path.read_text().splitlines()
    assert "logprob_abs_diff_mean" in header
    assert "ratio_abs_dev_max" in header
    assert "mismatch_k3_kl" in header
    assert "continuous_stale_versions" in header
    assert "continuous_predicted_admit_staleness" in header
    assert "continuous_admit_blocked_on_staleness" in header
    values = dict(zip(header.split(","), row.split(","), strict=True))
    assert values["logprob_abs_diff_mean"] == "0.100000"
    assert values["ratio_abs_dev_max"] == "0.400000"
    assert values["mismatch_kl"] == "-0.500000"
    # Continuous-async diagnostics sourced from TrainStepMetrics.phase_times.
    assert values["continuous_stale_versions"] == "1.0"
    assert values["continuous_ready_groups"] == "3.0"
    assert values["continuous_weight_sync_pause_s"] == "0.2500"
    # Admit-time predicted-version throttle observability.
    assert values["continuous_predicted_admit_staleness"] == "2.0"
    assert values["continuous_admit_blocked_on_staleness"] == "1.0"


@pytest.mark.parametrize(
    "frozen,expected", [("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_frozen_axis_in_runtime_spec(frozen, expected):
    # P1b: the frozen axis rides the spec as a real torch.dtype (like `dtype`).
    """Checks frozen axis in runtime spec."""
    from vrl.models.diffusion.sd3_5.runtime import extract_sd3_5_runtime_spec

    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"train": "fp32", "frozen": frozen})
    spec = extract_sd3_5_runtime_spec(cfg, torch.device("cpu"), torch.float32)
    assert spec.frozen_dtype is expected
