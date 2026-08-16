"""A rollout approximation must not run with every drift correction off.

Quantization is already covered without a new check: it changes the rollout
precision label, so ``PrecisionPolicy.stages_match`` goes False and the trainer
installs TIS correction plus a drift guard whose default ``mode="auto"``
resolves to ``"fail"``.

TeaCache is the uncovered case, and the reason this check exists. It reuses a
cached ``noise_pred`` on skipped denoise steps -- so the collection-time
log-prob no longer matches the trainer's exact replay forward -- while leaving
both roles' precision labels identical. Every automatic correction stays off and
the run trains on uncorrected off-policy gradients while still reporting
convergence.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.precision import resolve_precision_policy
from vrl.config.validation import require_guarded_rollout_drift


def _cfg(**overrides) -> OmegaConf:
    """A minimal config carrying only what the drift check reads."""

    base = {
        "precision": {
            "float32_precision": "tf32",
            "training": {"dtype": "bf16"},
            "rollout": {"dtype": "bf16"},
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _precision(cfg) -> object:
    return resolve_precision_policy(cfg)


def test_teacache_without_any_correction_is_refused() -> None:
    cfg = _cfg(sampling={"teacache": True})

    with pytest.raises(ValueError, match=r"teacache.*no drift guard"):
        require_guarded_rollout_drift(cfg, _precision(cfg))


def test_teacache_mapping_form_is_refused_too() -> None:
    """``teacache: {threshold: ...}`` enables it just as ``teacache: true`` does."""

    cfg = _cfg(sampling={"teacache": {"threshold": 0.2}})

    with pytest.raises(ValueError, match="teacache"):
        require_guarded_rollout_drift(cfg, _precision(cfg))


@pytest.mark.parametrize(
    "sampling",
    [
        {},
        {"teacache": False},
        {"teacache": {"enabled": False}},
    ],
)
def test_teacache_off_passes(sampling: dict) -> None:
    cfg = _cfg(sampling=sampling)

    require_guarded_rollout_drift(cfg, _precision(cfg))


def test_quantized_rollout_needs_no_extra_check() -> None:
    """A precision split already arms the guard + TIS, so this must not fire.

    Firing here would reject every fp8 rollout that also uses TeaCache, even
    though such a run is exactly the covered case.
    """

    cfg = _cfg(
        sampling={"teacache": True},
        precision={
            "float32_precision": "tf32",
            "training": {"dtype": "bf16"},
            "rollout": {"dtype": "bf16", "quantization": {"format": "fp8"}},
        },
    )
    assert not _precision(cfg).stages_match

    require_guarded_rollout_drift(cfg, _precision(cfg))


@pytest.mark.parametrize(
    "expert_block",
    ["precision_drift_guard", "precision_correction"],
)
def test_explicit_expert_block_is_honored(expert_block: str) -> None:
    """The same escape hatch the precision-split default path respects.

    ``OnlineTrainerConfig.from_cfg`` only fills these in when the user has not,
    so an explicit block means the correction policy was chosen deliberately.
    """

    cfg = _cfg(
        sampling={"teacache": True},
        trainer={expert_block: {"mode": "warn"}},
    )

    require_guarded_rollout_drift(cfg, _precision(cfg))


def test_check_runs_inside_validate_training_config() -> None:
    """Wired into the real entry point, not only callable on its own."""

    import inspect

    from vrl.config.validation import validate_training_config

    assert "require_guarded_rollout_drift" in inspect.getsource(validate_training_config)
