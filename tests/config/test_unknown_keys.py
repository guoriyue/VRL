"""Whole-tree unknown-key walker: every depth, one mechanism."""

from __future__ import annotations

import dataclasses
import logging

import pytest
from omegaconf import OmegaConf

from vrl.config.unknown_keys import find_unknown_keys, warn_unknown_keys
from vrl.utils.profiling import TorchProfilerConfig


@dataclasses.dataclass
class _FixtureNestedConfig:
    enabled: bool = False


@dataclasses.dataclass
class _FixtureRuntimeConfig:
    actor_scalar: int = dataclasses.field(
        default=1,
        metadata={"yaml": "actor"},
    )
    actor_nested: _FixtureNestedConfig = dataclasses.field(
        default_factory=_FixtureNestedConfig,
        metadata={"yaml": "actor.actor_nested"},
    )
    bridged_value: int = dataclasses.field(
        default=0,
        metadata={"yaml": "bridged"},
    )
    rollout_value: int = dataclasses.field(
        default=0,
        metadata={"yaml": "rollout"},
    )


def test_config_block_known_keys_derive_from_dataclass_fields() -> None:
    """The mechanism must not maintain a second dataclass field allow-list."""
    from vrl.config.unknown_keys import ConfigBlock

    assert ConfigBlock(TorchProfilerConfig).known == frozenset(
        field.name for field in dataclasses.fields(TorchProfilerConfig)
    )


def test_future_runtime_metadata_fields_enter_their_public_section_automatically() -> None:
    from vrl.config.schema import _online_runtime_section_shape

    known, children = _online_runtime_section_shape(
        "actor",
        (_FixtureRuntimeConfig,),
    )

    assert known == frozenset({"actor_scalar", "actor_nested"})
    assert children["actor_nested"].known == frozenset({"enabled"})


@pytest.mark.parametrize("section", ["actor", "trainer"])
def test_online_section_keys_derive_from_runtime_yaml_owners(section: str) -> None:
    from vrl.config.schema import ActorSection, TrainerSection
    from vrl.config.unknown_keys import ConfigBlock, _root_block
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    public_section = {
        "actor": ActorSection,
        "trainer": TrainerSection,
    }[section]
    runtime_fields = {
        field.name
        for owner in (TrainerConfig, OnlineBatchPlan)
        for field in dataclasses.fields(owner)
        if str(field.metadata.get("yaml", "")).partition(".")[0] == section
    }
    nested_fields = {
        field.name
        for owner in (TrainerConfig, OnlineBatchPlan)
        for field in dataclasses.fields(owner)
        if str(field.metadata.get("yaml", "")).startswith(f"{section}.")
    }

    block = _root_block().children[section]

    assert isinstance(block, ConfigBlock)
    assert runtime_fields.isdisjoint(public_section.model_fields)
    assert block.known == runtime_fields | set(public_section.model_fields)
    assert nested_fields <= set(block.children)


def test_typed_online_sections_keep_derived_fields_and_drop_unknown_extras() -> None:
    from vrl.config.schema import parse_config

    parsed = parse_config(
        OmegaConf.create(
            {
                "actor": {"ppo_epochs": 2, "future_typo": "ignored"},
                "trainer": {
                    "output_dir": "outputs/test",
                    "profile": True,
                    "future_typo": "ignored",
                },
            },
        ),
    )

    assert parsed.actor is not None
    assert parsed.actor.ppo_epochs == 2
    assert "future_typo" not in parsed.actor.model_fields_set
    assert "future_typo" not in parsed.actor.model_dump()
    assert parsed.trainer is not None
    assert parsed.trainer.output_dir == "outputs/test"
    assert parsed.trainer.profile is True
    assert "future_typo" not in parsed.trainer.model_fields_set
    assert "future_typo" not in parsed.trainer.model_dump()


def test_unknown_keys_are_found_at_every_depth() -> None:
    """Typos at top level, section level, and nested blocks are all named."""
    cfg = OmegaConf.create(
        {
            "samplng": {"num_steps": 10},
            "sampling": {"num_stps": 5},
            "rollout": {"sde": {"type": "flow_grpo", "window_sze": 3}},
            "distributed": {
                "resources": {"reward": {"num_gpus": 1, "share_with_rolout": True}},
            },
            "actor": {
                "mixed_precision": "bf16",
                "optim": {"lr": 1e-4, "lrr": 2, "allow_tf32": True},
            },
        },
    )
    unknown = find_unknown_keys(cfg)
    assert unknown == [
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
    assert find_unknown_keys(cfg) == []


@pytest.mark.parametrize(
    ("config", "path"),
    [
        ({"data": {"source": "huggingface"}}, "data.source"),
        (
            {"data": {"preprocessing": {"metadata_schema": "geneval"}}},
            "data.preprocessing.metadata_schema",
        ),
        (
            {"data": {"preprocessing": {"target_text": "quoted_substring"}}},
            "data.preprocessing.target_text",
        ),
        (
            {"data": {"preprocessing": {"media_type": "video"}}},
            "data.preprocessing.media_type",
        ),
    ],
)
def test_removed_data_noop_keys_are_unknown(
    config: dict[str, object],
    path: str,
) -> None:
    assert find_unknown_keys(OmegaConf.create(config)) == [path]


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

    assert find_unknown_keys(cfg) == [
        f"production.kling_video_reward.{removed_key}",
    ]


def test_production_enabled_is_a_known_key() -> None:
    cfg = OmegaConf.create(
        {"production": {"kling_video_reward": {"enabled": True}}},
    )

    assert find_unknown_keys(cfg) == []


def test_removed_rollout_queue_knob_is_unknown() -> None:
    cfg = OmegaConf.create(
        {"trainer": {"rollout_orchestration": {"max_pending_rollouts": 2}}},
    )
    assert find_unknown_keys(cfg) == [
        "trainer.rollout_orchestration.max_pending_rollouts",
    ]


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

    assert find_unknown_keys(actor_cfg) == []
    assert find_unknown_keys(rollout_cfg) == [f"rollout.{field_name}"]


def test_removed_sampling_r1_duplicate_is_unknown() -> None:
    cfg = OmegaConf.create(
        {"sampling": {"r1": {"train_segments": {"initial_image": True}}}},
    )
    assert find_unknown_keys(cfg) == ["sampling.r1"]


def test_removed_sampling_cfg_knob_is_unknown() -> None:
    cfg = OmegaConf.create({"sampling": {"cfg": False, "guidance_scale": 1.0}})
    assert find_unknown_keys(cfg) == ["sampling.cfg"]


def test_removed_use_ar_scheduler_knob_is_unknown() -> None:
    cfg = OmegaConf.create({"sampling": {"use_ar_scheduler": True}})
    assert find_unknown_keys(cfg) == ["sampling.use_ar_scheduler"]


def test_removed_model_dtype_is_unknown() -> None:
    cfg = OmegaConf.create({"model": {"family": "sd3_5", "dtype": "bf16"}})
    assert find_unknown_keys(cfg) == ["model.dtype"]


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

    assert find_unknown_keys(cfg) == []


def test_warn_unknown_keys_logs_one_line(caplog) -> None:
    cfg = OmegaConf.create({"sampling": {"num_stps": 5}})
    with caplog.at_level(logging.WARNING):
        unknown = warn_unknown_keys(cfg)
    assert unknown == ["sampling.num_stps"]
    assert any("sampling.num_stps" in r.getMessage() for r in caplog.records)


# ── Anti-rot sweeps: the registry must track the real code and real configs ──


def test_every_config_path_read_by_code_is_registered() -> None:
    """Anti-rot: cfg paths the code reads must exist in the known-key tree."""
    from vrl.config.lint import unregistered_code_paths

    assert unregistered_code_paths() == []


def test_all_experiment_configs_have_zero_unknown_keys() -> None:
    """Anti-rot: every shipped experiment must load with zero unknown keys."""
    from vrl.config.lint import unknown_yaml_keys

    assert unknown_yaml_keys() == {}
