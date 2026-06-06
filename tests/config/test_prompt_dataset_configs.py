from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.validation import validate_training_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"


def test_prompt_dataset_configs_declare_loader_preprocessing_and_sampler() -> None:
    """Checks prompt dataset configs declare loader preprocessing and sampler."""
    expected = {
        "ocr": (
            "datasets/ocr/train.txt",
            "datasets/ocr/test.txt",
            "text",
            "quoted_substring",
        ),
        "geneval": (
            "datasets/geneval/train.jsonl",
            "datasets/geneval/test.jsonl",
            "jsonl",
            "geneval",
        ),
        "pickscore_sfw": (
            "datasets/pickscore_sfw/train.txt",
            "datasets/pickscore_sfw/test.txt",
            "text",
            "none",
        ),
    }
    for name, (manifest, eval_manifest, data_format, preprocessing_value) in expected.items():
        cfg = OmegaConf.load(CONFIGS_ROOT / "dataset" / f"{name}.yaml")
        assert cfg.data.loader == "prompt_manifest"
        assert cfg.data.manifest == manifest
        assert cfg.data.eval_manifest == eval_manifest
        assert cfg.data.preprocessing.format == data_format
        if data_format == "jsonl":
            assert cfg.data.preprocessing.metadata_schema == preprocessing_value
        else:
            assert cfg.data.preprocessing.target_text == preprocessing_value
        assert cfg.data.sampler.type == "random_without_replacement"


def test_videophy_i2v_dataset_config_declares_image_caption_loader() -> None:
    """Checks VideoPhy I2V dataset config declares image caption loader."""
    cfg = OmegaConf.load(CONFIGS_ROOT / "dataset" / "videophy_i2v.yaml")

    assert cfg.data.loader == "prompt_image_manifest"
    assert cfg.data.manifest == "data/external/videophy_i2v/manifests/train.jsonl"
    assert cfg.data.eval_manifest == "data/external/videophy_i2v/manifests/eval.jsonl"
    assert cfg.data.artifact_data_root == "data/external/videophy_i2v"
    assert cfg.data.source_report == "data/external/videophy_i2v/report.json"
    assert cfg.data.task_type == "image_to_video"
    assert cfg.data.preprocessing.format == "image_caption_jsonl"
    assert cfg.data.preprocessing.image_field == "image"
    assert cfg.data.preprocessing.caption_field == "caption"
    assert cfg.data.preprocessing.conditioning == "reference_image"
    assert cfg.data.sampler.type == "random_without_replacement"


def test_pickapic_dataset_config_declares_preprocessing_and_sampler() -> None:
    """Checks Pick-a-Pic dataset config declares preprocessing and sampler."""
    cfg = OmegaConf.load(CONFIGS_ROOT / "dataset" / "pickapic_v2.yaml")

    assert cfg.data.loader == "pickapic_preference"
    assert cfg.data.source == "huggingface"
    assert cfg.data.dataset_name == "yuvalkirstain/pickapic_v2"
    assert cfg.data.preprocessing.resolution == 0
    assert cfg.data.preprocessing.random_crop is False
    assert cfg.data.preprocessing.horizontal_flip is True
    assert cfg.data.sampler.shuffle is True
    assert cfg.data.sampler.drop_last is True
    assert cfg.data.sampler.dataloader_num_workers == 4


def test_prompt_rewards_stay_in_reward_configs() -> None:
    """Checks prompt rewards stay in reward configs."""
    expected = {
        "geneval": "geneval",
        "pickscore": "pickscore",
    }
    for config_name, reward_name in expected.items():
        cfg = OmegaConf.load(CONFIGS_ROOT / "reward" / f"{config_name}.yaml")
        assert list(cfg.reward.components.keys()) == [reward_name]


def test_sd35_prompt_dataset_experiments_load_and_validate() -> None:
    """Checks sd35 prompt dataset experiments load and validate."""
    for name in (
        "diffusion/sd3_5/online_grpo_geneval",
        "diffusion/sd3_5/online_grpo_pickscore",
    ):
        cfg = load_config(f"experiment/{name}")
        validate_training_config(cfg)
        assert cfg.trainer.entrypoint == "vrl.scripts.diffusion.sd3_5.train:train_sd3_5_grpo"
