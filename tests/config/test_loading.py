"""Config discovery behavior shared by source and installed distributions."""

from pathlib import Path

from vrl.config.loading import CONFIGS_ROOT, load_config


def test_default_root_loads_real_layered_experiment() -> None:
    """The public loader resolves the shipped base/recipe/experiment overlays."""

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")

    assert CONFIGS_ROOT.is_dir()
    assert cfg.model.family == "sd3_5"
    assert cfg.trainer.entrypoint == "vrl.scripts.diffusion.train:train_diffusion_grpo"


def test_default_root_is_not_the_callers_working_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Installed CLIs find package configs even when launched elsewhere."""

    monkeypatch.chdir(tmp_path)

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")

    assert cfg.model.family == "sd3_5"
