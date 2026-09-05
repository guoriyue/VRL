"""Runtime preset composition without model/reward experiment cross-products."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.config.loading import compose_config, load_config


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    presets = {
        "neutral.yaml": "trainer:\n  output_dir: ???\n",
        "legacy.yaml": "defaults:\n  - /reward/base\n",
        "reward/base.yaml": "reward:\n  transport: base\n  score: 0.1\n",
        "reward/alternate.yaml": "reward:\n  transport: alternate\n",
        "reward/policy.yaml": (
            "defaults:\n  - /reward/base\n  - _self_\nreward:\n  rubric: quality\n  score: 0.2\n"
        ),
        "reward/identity.yaml": "reward:\n  judge: pinned\n  score: 0.3\n",
        "dataset/train.yaml": "data:\n  manifest: train.jsonl\n",
        "run/local.yaml": "trainer:\n  output_dir: outputs/local\n",
    }
    for name, content in presets.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def test_runtime_presets_supply_absent_groups_and_required_values(config_tree: Path) -> None:
    overrides = ["+reward=policy", "+dataset=train", "+run=local"]

    cfg = load_config("neutral", overrides=overrides, root=config_tree)

    assert cfg.reward.transport == "base"
    assert cfg.reward.rubric == "quality"
    assert cfg.data.manifest == "train.jsonl"
    assert cfg.trainer.output_dir == "outputs/local"


def test_additive_layers_are_ordered_and_dotlist_values_apply_last(config_tree: Path) -> None:
    overrides = ["+reward=policy", "+reward=identity"]
    cfg = compose_config("neutral", overrides=overrides, root=config_tree)
    assert cfg.reward.score == 0.3
    assert cfg.reward.rubric == "quality"
    assert cfg.reward.judge == "pinned"

    reversed_cfg = compose_config("neutral", overrides=list(reversed(overrides)), root=config_tree)
    assert reversed_cfg.reward.score == 0.2

    scalar_cfg = compose_config(
        "neutral", overrides=["reward.score=0.9", *overrides], root=config_tree
    )
    assert scalar_cfg.reward.score == 0.9


def test_additive_preset_inheritance_ignores_legacy_default_replacements(
    config_tree: Path,
) -> None:
    cfg = compose_config(
        "legacy", overrides=["/reward=alternate", "+reward=policy"], root=config_tree
    )

    assert cfg.reward.transport == "base"
    assert cfg.reward.rubric == "quality"


def test_missing_values_remain_mandatory_after_additive_composition(config_tree: Path) -> None:
    cfg = compose_config("neutral", overrides=["+reward=policy"], root=config_tree)
    assert OmegaConf.missing_keys(cfg) == {"trainer.output_dir"}

    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        load_config("neutral", overrides=["+reward=policy"], root=config_tree)


@pytest.mark.parametrize(
    "override",
    [
        "+reward",
        "+=policy",
        "+reward=",
        "++reward=policy",
        "+/reward=policy",
        "+reward=/policy",
        "+reward=../policy",
        "+reward/../dataset=policy",
        "+reward=./policy",
    ],
)
def test_invalid_additive_selection_is_rejected(config_tree: Path, override: str) -> None:
    with pytest.raises(ValueError, match="invalid additive preset override"):
        compose_config("neutral", overrides=[override], root=config_tree)


def test_missing_additive_preset_fails_instead_of_being_ignored(config_tree: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compose_config("neutral", overrides=["+reward=missing"], root=config_tree)
