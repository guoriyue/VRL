"""Per-group loadability of dataset and reward config building blocks.

Dataset and reward group YAMLs under the bundled ``dataset`` / ``reward`` groups
are independently reusable building blocks: an experiment composes one of each
via defaults. The whole-experiment load+validate loop in
``test_load_all_experiments.py::test_all_experiments_load_and_validate`` only
exercises the *merged* config of experiments that happen to consume a group, so
it never proves that each group file on its own parses into a valid section.
These tests pin that per-group invariant — structurally, never by echoing the
literal YAML values (which ``_validate_data`` does not validate anyway).
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.loading import bundled_config_resource, list_bundled_configs
from vrl.config.schema import DataConfig
from vrl.rewards.functions.registry import _register_builtins, get_reward

# Dataset groups whose independent loadability we pin. Each is a reusable
# building block consumed by one or more experiment YAMLs.
DATASET_GROUPS = (
    "ocr",
    "geneval",
    "pickscore_sfw",
    "videophy_i2v",
    "pickapic_v2",
)


def _load_bundled_raw(name: str):
    resource = bundled_config_resource(name)
    with resource.open("r", encoding="utf-8") as stream:
        return OmegaConf.load(stream)


@pytest.mark.parametrize("group", DATASET_GROUPS)
def test_dataset_group_loads_into_valid_data_config(group: str) -> None:
    """Each dataset group YAML parses into a DataConfig that passes _validate_data.

    Asserts structural validity only: the loader is one of the typed
    discriminator values and the after-validator (``_validate_data``) accepts
    the declared shape. No literal-value equality — values are declarations,
    validated structurally by ``test_schema.py`` discriminator tests.
    """
    raw = _load_bundled_raw(f"dataset/{group}")
    payload = OmegaConf.to_container(raw.data, resolve=True)

    # Constructing the model runs the loader discriminator (Literal field) and
    # the @model_validator(mode="after") _validate_data; neither must raise.
    cfg = DataConfig.model_validate(payload)

    valid_loaders = {"prompt_manifest", "prompt_image_manifest", "pickapic_preference"}
    assert cfg.loader in valid_loaders


def test_reward_component_keys_resolve_to_registered_reward_names() -> None:
    """Each reward config's component key must name a registered reward.

    ``RewardConfig.components`` is an OPEN dict (reward names are user-chosen,
    schema.py:52-53) and ``validate_training_config`` does not check key
    registration. The real consumer is ``MultiReward.from_dict`` →
    ``get_reward(name)``, which raises ``KeyError`` at runtime for an
    unregistered key. This guards that every shipped reward group's component
    key actually resolves, value-agnostically (no literal name table), so a new
    prompt reward cannot ship a typo'd component key undetected.
    """
    _register_builtins()  # populate the lazily-filled registry get_reward reads
    for name in list_bundled_configs("reward"):
        raw = _load_bundled_raw(name)
        reward = raw.get("reward", None)
        if reward is None:
            continue  # non-reward asset (e.g. README placeholder)
        for key in reward.components:
            # Raises KeyError if `key` is not a registered reward name.
            get_reward(key)
