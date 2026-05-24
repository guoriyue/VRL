from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.validation import validate_training_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"


def test_prompt_dataset_configs_are_named_by_schema_not_source() -> None:
    expected = {
        "ocr": (
            "datasets/ocr/train.txt",
            "datasets/ocr/test.txt",
        ),
    }
    for name, (manifest, eval_manifest) in expected.items():
        cfg = OmegaConf.load(CONFIGS_ROOT / "dataset" / f"{name}.yaml")
        assert cfg.data.manifest == manifest
        assert cfg.data.eval_manifest == eval_manifest


def test_prompt_task_configs_keep_data_and_reward_together() -> None:
    expected = {
        "geneval": (
            "datasets/geneval/train.jsonl",
            "datasets/geneval/test.jsonl",
            "geneval",
        ),
        "pickscore_sfw": (
            "datasets/pickscore_sfw/train.txt",
            "datasets/pickscore_sfw/test.txt",
            "pickscore",
        ),
    }
    for name, (manifest, eval_manifest, reward_name) in expected.items():
        cfg = OmegaConf.load(CONFIGS_ROOT / "task" / f"{name}.yaml")
        assert cfg.data.manifest == manifest
        assert cfg.data.eval_manifest == eval_manifest
        assert list(cfg.reward.components.keys()) == [reward_name]


def test_sd35_prompt_dataset_experiments_load_and_validate() -> None:
    for name in (
        "diffusion/sd3_5/online_grpo_ocr_prompt_alignment",
        "diffusion/sd3_5/online_grpo_geneval",
        "diffusion/sd3_5/online_grpo_pickscore",
    ):
        cfg = load_config(f"experiment/{name}")
        validate_training_config(cfg)
        assert cfg.trainer.entrypoint == "vrl.scripts.diffusion.sd3_5.train:train_sd3_5_grpo"
