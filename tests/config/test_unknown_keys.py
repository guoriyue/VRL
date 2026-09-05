"""Unknown keys fail at parse_config, at every depth, with one message."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tests.config.helpers import unknown_keys
from vrl.config.schema import parse_config


def test_required_blocks_do_not_hide_unknown_keys_during_structural_lint() -> None:
    """A ``???`` value is a launch-time decision; the key beside it is still checked."""
    from vrl.config.lint import experiment_parse_error

    cfg = OmegaConf.create({"reward": "???", "data": "???", "unknown_section": "???"})
    assert experiment_parse_error(cfg) == "unknown unknown_section"
    assert experiment_parse_error(OmegaConf.create({"reward": "???", "data": "???"})) is None
    assert (
        experiment_parse_error(
            OmegaConf.create({"model": {"family": "sd3_5"}, "actor": {"optim": {"lr": "???"}}}),
        )
        is None
    )


def test_typed_online_sections_keep_derived_fields_and_reject_unknown_extras() -> None:
    parsed = parse_config(
        OmegaConf.create(
            {
                "actor": {"ppo_epochs": 2},
                "trainer": {"output_dir": "outputs/test", "profile": True},
            },
        ),
    )
    assert parsed.actor is not None
    assert parsed.actor.ppo_epochs == 2
    assert parsed.trainer is not None
    assert parsed.trainer.output_dir == "outputs/test"
    assert parsed.trainer.profile is True

    with pytest.raises(ValueError, match=r"unknown actor\.future_typo, trainer\.future_typo"):
        parse_config(
            OmegaConf.create(
                {
                    "actor": {"ppo_epochs": 2, "future_typo": "ignored"},
                    "trainer": {"output_dir": "outputs/test", "future_typo": "ignored"},
                },
            ),
        )


def test_unknown_keys_are_found_at_every_depth() -> None:
    """Typos at top level, section level, nested models, and nested runtime
    dataclasses are all named together, sorted."""
    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5"},
            "samplng": {"num_steps": 10},
            "sampling": {"num_stps": 5},
            "rollout": {"sde": {"type": "flow_grpo", "window_sze": 3}},
            "distributed": {
                "resources": {"reward": {"device": "gpu", "share_with_rolout": True}},
            },
            "actor": {
                "mixed_precision": "bf16",
                "optim": {"lr": 1e-4, "lrr": 2, "allow_tf32": True},
            },
        },
    )
    assert unknown_keys(cfg) == [
        "actor.mixed_precision",
        "actor.optim.allow_tf32",
        "actor.optim.lrr",
        "distributed.resources.reward.share_with_rolout",
        "rollout.sde.window_sze",
        "sampling.num_stps",
        "samplng",
    ]


def test_open_blocks_accept_arbitrary_keys() -> None:
    """worker_config and non-modeled reward kwargs are open by design."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"ocr": 1.0},
                "kwargs": {
                    "ocr": {"anything": 1},
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "r",
                        "score_key": "s",
                        "worker_config": {"any_worker_key": True},
                    },
                },
            },
        },
    )
    assert unknown_keys(cfg) == []


@pytest.mark.parametrize("removed_key", ["enabld", "report_path"])
def test_production_gate_is_closed(removed_key: str) -> None:
    cfg = OmegaConf.create(
        {
            "production": {
                "kling_video_reward": {
                    "enabled": False,
                    removed_key: "unused",
                },
            },
        },
    )

    assert unknown_keys(cfg) == [f"production.kling_video_reward.{removed_key}"]


def test_production_enabled_is_a_known_key() -> None:
    cfg = OmegaConf.create(
        {"production": {"kling_video_reward": {"enabled": True}}},
    )

    assert unknown_keys(cfg) == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("microbatch_size", 1),
        ("host_memory_budget_fraction", 0.9),
    ],
)
def test_online_update_memory_keys_are_owned_by_actor(
    field_name: str,
    value: object,
) -> None:
    actor_cfg = OmegaConf.create({"actor": {field_name: value}})
    rollout_cfg = OmegaConf.create({"rollout": {field_name: value}})

    assert unknown_keys(actor_cfg) == []
    assert unknown_keys(rollout_cfg) == [f"rollout.{field_name}"]


@pytest.mark.parametrize(
    ("family", "model_values"),
    [
        (
            "causvid",
            {
                "accept_noncommercial_license": True,
                "base_model_path": "Wan-AI/Wan2.1-T2V-1.3B",
                "base_model_revision": "base-revision",
                "causvid_source_path": "third_party/CausVid",
                "causvid_source_revision": "source-revision",
                "checkpoint_file": "autoregressive_checkpoint/model.pt",
                "checkpoint_sha256": "digest",
            },
        ),
        (
            "magi_1",
            {
                "checkpoint_path": None,
                "config_path": "example/4.5B/4.5B_base_config.json",
                "python_executable": "third_party/MAGI-1/.venv/bin/python",
                "source_path": "third_party/MAGI-1",
                "source_revision": "source-revision",
                "t5_pretrained_path": None,
                "timeout_seconds": 3600,
                "vae_pretrained_path": None,
            },
        ),
    ],
)
def test_chunk_autoregressive_model_keys_are_registered(
    family: str,
    model_values: dict[str, object],
) -> None:
    cfg = OmegaConf.create({"model": {"family": family, **model_values}})

    assert unknown_keys(cfg) == []


def test_all_experiment_configs_parse() -> None:
    """Anti-rot: every shipped experiment must parse through the typed schema."""
    from vrl.config.lint import experiment_parse_failures

    assert experiment_parse_failures() == {}
