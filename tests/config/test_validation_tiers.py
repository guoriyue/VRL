"""The three validation tiers stay where they are declared.

Tier 1 (section shapes) is ``schema.py``; tier 2 (cross-section rules) is
``rules.check_cross_section_rules``, run by ``RootConfig``'s own validator;
tier 3 (launch gates) runs in ``require_training_config`` in ``validation.py``.
These pin the seams a new check must go
through, so a rule cannot quietly grow back into the pydantic model or a gate
into ``parse_config``.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config import validation
from vrl.config.schema import parse_config


def test_launch_gates_do_not_run_inside_parse_config() -> None:
    """A gate that needs the precision policy or a runtime module must not tax
    ``parse_config`` callers: the compile matrix reads the build-role resolver,
    so a config it rejects still parses and fails only through
    ``require_training_config``."""

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5", "torch_compile": {"enable": True}},
            "precision": {"float32_precision": "tf32", "training": {"dtype": "bf16"}},
            "actor": {"gradient_checkpointing": "full_cpu"},
        }
    )

    root = parse_config(cfg)

    assert root.model.torch_compile.enable is True
    with pytest.raises(ValueError, match=r"torch_compile\.enable=true cannot combine"):
        validation.require_training_config(cfg)


def test_quantized_replay_is_refused_without_a_frozen_base() -> None:
    block = {
        "model": {"family": "sd3_5"},
        "precision": {
            "float32_precision": "tf32",
            "training": {"dtype": "bf16", "quantization": {"format": "nvfp4"}},
        },
    }

    with pytest.raises(ValueError, match=r"training\.quantization requires model\.use_lora"):
        parse_config(OmegaConf.create(block))

    block["model"]["use_lora"] = True
    assert parse_config(OmegaConf.create(block)).precision.training.quantization.format == "nvfp4"
