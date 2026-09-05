from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vrl.scripts.rewards import install_countgd as installer
from vrl.scripts.rewards.countgd_environment_lock import (
    ENVIRONMENT_LOCK,
    ArtifactKind,
    environment_lock_digest,
    environment_lock_payload,
)


def test_legacy_environment_mode_requires_known_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "countgd"
    target.mkdir()
    manifest_path = target / "install_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"runtime_environment": {"inherits_system_site_packages": True}},
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(installer, "_LEGACY_INSTALL_MANIFEST_SHA256", digest)

    assert installer._manifest_uses_legacy_environment(target) is True

    manifest_path.write_text(
        json.dumps(
            {
                "runtime_environment": {"inherits_system_site_packages": True},
                "untrusted": True,
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy CountGD install manifest SHA-256"):
        installer._manifest_uses_legacy_environment(target)


def test_locked_environment_mode_rejects_manifest_provenance_drift(
    tmp_path: Path,
) -> None:
    target = tmp_path / "countgd"
    target.mkdir()
    manifest_path = target / "install_manifest.json"
    lock = {**environment_lock_payload(), "sha256": environment_lock_digest()}
    manifest = {
        "model_version": installer.COUNTGD_MODEL_VERSION,
        "inference_protocol": installer.countgd_model_protocol(),
        "runtime_environment": {
            "inherits_system_site_packages": False,
            "lock": lock,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert installer._manifest_uses_legacy_environment(target) is False

    lock["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="locked install manifest mismatch"):
        installer._manifest_uses_legacy_environment(target)


def test_environment_lock_covers_model_and_http_service() -> None:
    requirements = {distribution.requirement for distribution in ENVIRONMENT_LOCK}
    locked_names = {distribution.name for distribution in ENVIRONMENT_LOCK}
    sdists = [
        distribution
        for distribution in ENVIRONMENT_LOCK
        if distribution.artifact_kind is ArtifactKind.SDIST
    ]

    assert len(requirements) == len(ENVIRONMENT_LOCK)
    assert {
        "aiohttp",
        "omegaconf",
        "antlr4-python3-runtime",
        "torch",
        "transformers",
    } <= locked_names
    assert [(distribution.name, distribution.version) for distribution in sdists] == [
        ("antlr4-python3-runtime", "4.9.3"),
    ]
    assert all(len(distribution.sha256) == 64 for distribution in ENVIRONMENT_LOCK)
