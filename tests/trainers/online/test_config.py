"""Online trainer configuration ownership tests."""

from __future__ import annotations


def test_trainer_config_public_facades_share_online_owner() -> None:
    from vrl.trainers import TrainerConfig as root_export
    from vrl.trainers.online import TrainerConfig as online_export
    from vrl.trainers.online.config import TrainerConfig

    assert root_export is TrainerConfig
    assert online_export is TrainerConfig


def test_core_package_does_not_claim_online_trainer_config() -> None:
    import vrl.trainers.core as core

    assert "TrainerConfig" not in core.__all__
    assert not hasattr(core, "TrainerConfig")
