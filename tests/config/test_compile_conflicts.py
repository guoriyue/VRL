"""The torch.compile compatibility matrix.

One home for every torch.compile incompatibility. Each was discovered
separately and used to live somewhere different -- grad-checkpointing in the
trainer, sequence parallelism nowhere at all -- so adding the next one meant
first finding the others.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config


def _compile_cfg(**overrides) -> OmegaConf:
    """A config with compile on and nothing else set, plus ``overrides``."""

    base = {"model": {"family": "sd3_5", "torch_compile": {"enable": True}}}
    base.update(overrides)
    return OmegaConf.create(base)


def test_compile_alone_has_no_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    assert compile_conflicts(parse_config(_compile_cfg())) == ()


def test_compile_off_never_conflicts() -> None:
    """Every other feature is free as long as compile is not requested."""
    from vrl.config.validation import compile_conflicts

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
            "actor": {"gradient_checkpointing": True},
            "distributed": {"training": {"strategy": "fsdp"}},
        },
    )

    assert compile_conflicts(parse_config(cfg)) == ()


def test_incompatible_checkpointing_is_reported() -> None:
    from vrl.config.validation import compile_conflicts

    conflicts = compile_conflicts(
        parse_config(_compile_cfg(actor={"gradient_checkpointing": "full_cpu"}))
    )

    assert len(conflicts) == 1
    assert "gradient_checkpointing" in conflicts[0].message


def test_selective_checkpointing_and_fsdp_compile_with_the_replay_policy() -> None:
    # Full and selective checkpointing ran replay forward+backward under
    # compile (tests/trainers/test_activation_checkpointing.py traces the
    # selective policy at the root), and a compiled FSDP2 replay trains like
    # the unsharded compiled module (tests/trainers/test_fsdp_compile_distributed.py).
    from vrl.config.validation import compile_conflicts

    cfg = _compile_cfg(
        actor={"gradient_checkpointing": "selective"},
        distributed={"training": {"strategy": "fsdp"}},
    )

    assert compile_conflicts(parse_config(cfg)) == ()


def test_several_conflicts_are_reported_together() -> None:
    """One pass over the matrix, not fail-on-first: the user sees all of them."""
    from vrl.config.validation import compile_conflicts

    cfg = _compile_cfg(actor={"gradient_checkpointing": "full_cpu"})
    cfg.model.fused_kernels = True
    conflicts = compile_conflicts(parse_config(cfg))

    assert len(conflicts) == 2


def test_compile_gate_raises_with_every_reason() -> None:
    from vrl.config.validation import require_training_config

    cfg = _compile_cfg(
        precision={"float32_precision": "tf32", "training": {"dtype": "bf16"}},
        actor={"gradient_checkpointing": "full_cpu"},
    )
    cfg.model.fused_kernels = True
    with pytest.raises(ValueError) as excinfo:
        require_training_config(cfg)
    message = str(excinfo.value)
    assert "gradient_checkpointing" in message
    assert "fused_kernels" in message


# Every conflict in the matrix binds one build role, so ``scope`` releases the
# conflicts of the role it excludes. The rollout-scope case is the production
# unlock: full_cpu-checkpointing trainers with a compiled rollout.


def _scoped_compile_cfg(scope: str, **overrides) -> OmegaConf:
    cfg = _compile_cfg(**overrides)
    cfg.model.torch_compile.scope = scope
    return cfg


def test_rollout_scope_releases_trainer_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    cfg = _scoped_compile_cfg("rollout", actor={"gradient_checkpointing": "full_cpu"})

    assert compile_conflicts(parse_config(cfg)) == ()


def test_replay_scope_keeps_trainer_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    cfg = _scoped_compile_cfg("replay", actor={"gradient_checkpointing": "full_cpu"})

    conflicts = compile_conflicts(parse_config(cfg))
    assert len(conflicts) == 1
    assert conflicts[0].feature == "gradient_checkpointing"
    assert "gradient_checkpointing" in conflicts[0].message


def test_unknown_compile_scope_is_refused_at_config_load() -> None:
    from vrl.config.validation import compile_conflicts

    with pytest.raises(ValueError, match=r"torch_compile\.scope must be one of"):
        compile_conflicts(parse_config(_scoped_compile_cfg("trainer")))
