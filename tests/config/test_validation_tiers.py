"""The three validation tiers stay where they are declared.

Tier 1 (section shapes) is ``schema.py``; tier 2 (cross-section rules) is
``rules.check_cross_section_rules``, run by ``RootConfig``'s own validator;
tier 3 (launch gates) is the ``TRAINING_GATES`` registry in ``validation.py``,
run by ``require_training_config``. These pin the seams a new check must go
through, so a rule cannot quietly grow back into the pydantic model or a gate
into ``parse_config``.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config import validation
from vrl.config.schema import RootConfig, parse_config


def test_cross_section_rules_fire_on_direct_root_construction() -> None:
    """A caller that bypasses ``parse_config`` still gets tier 2."""

    with pytest.raises(ValueError, match=r"data\.sft_latents"):
        RootConfig.model_validate(
            {"model": {"family": "sd3_5"}, "algorithm": {"kind": "grpo", "sft_weight": 0.1}}
        )


def test_launch_gates_do_not_run_inside_parse_config() -> None:
    """A gate that needs the precision policy or the filesystem must not tax
    ``parse_config`` callers: the production gate reads manifests, so a config
    that enables it parses but fails only through ``require_training_config``."""

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5"},
            "precision": {"float32_precision": "tf32", "training": {"dtype": "bf16"}},
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )

    root = parse_config(cfg)

    assert root.production is not None
    with pytest.raises(ValueError, match=r"production\.kling_video_reward requires"):
        validation.require_training_config(cfg)


def test_dataset_provenance_requires_existing_paths(tmp_path) -> None:
    from vrl.trainers.data.provenance import DatasetProvenance

    manifest = tmp_path / "train.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    root = parse_config(
        OmegaConf.create(
            {
                "data": {
                    "loader": "prompt_manifest",
                    "manifest": str(manifest),
                    "eval_manifest": str(tmp_path / "missing.jsonl"),
                    "source_report": str(tmp_path / "report.json"),
                    "task_type": "text_to_video",
                    "preprocessing": {},
                    "sampler": {"type": "random_without_replacement"},
                }
            }
        )
    )

    with pytest.raises(ValueError, match=r"data\.eval_manifest does not exist"):
        DatasetProvenance.from_config(root.data)
