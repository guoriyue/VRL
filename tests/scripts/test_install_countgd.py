from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from vrl.scripts.rewards import install_countgd as installer
from vrl.scripts.rewards.countgd_environment_lock import (
    ENVIRONMENT_LOCK,
    ArtifactKind,
    environment_lock_digest,
    environment_lock_payload,
    locked_package_versions,
)


def test_patch_replacement_rejects_upstream_drift() -> None:
    replacement = installer._Replacement("old", "new", 2)

    assert (
        installer._replace_text_exact(
            "old\r\nold\r\n",
            (replacement,),
            context="example.cc",
        )
        == "new\nnew\n"
    )
    with pytest.raises(ValueError, match="patch input drift"):
        installer._replace_text_exact(
            "old\n",
            (replacement,),
            context="example.cc",
        )


def test_install_refuses_to_overwrite_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "countgd"
    target.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        installer.install_countgd(
            target_dir=target,
            python_executable=Path(sys.executable),
        )


def test_install_verifies_staging_and_published_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifacts" / "countgd"
    built: list[Path] = []
    verified: list[Path] = []

    def fake_build(staging: Path, **_: object) -> None:
        built.append(staging)
        (staging / "marker").write_text("qualified", encoding="utf-8")

    def fake_verify(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        verified.append(resolved)
        assert (resolved / "marker").read_text(encoding="utf-8") == "qualified"
        return {"status": "verified", "target_dir": str(resolved)}

    monkeypatch.setattr(installer, "_build_staged_install", fake_build)
    monkeypatch.setattr(installer, "verify_countgd_install", fake_verify)

    result = installer.install_countgd(
        target_dir=target,
        python_executable=Path(sys.executable),
    )

    assert result == {"status": "verified", "target_dir": str(target.resolve())}
    assert len(built) == 1
    assert verified == [built[0].resolve(), target.resolve()]
    assert built[0].parent.parent == target.parent.resolve()
    assert built[0].parent.name.startswith(".countgd.install-")
    assert not built[0].parent.exists()
    assert (target / "marker").read_text(encoding="utf-8") == "qualified"


def test_install_removes_new_target_when_final_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "countgd"

    def fake_build(staging: Path, **_: object) -> None:
        (staging / "marker").write_text("qualified", encoding="utf-8")

    def fake_verify(path: Path) -> dict[str, object]:
        if path.resolve() == target.resolve():
            raise ValueError("final-path failure")
        return {"status": "verified"}

    monkeypatch.setattr(installer, "_build_staged_install", fake_build)
    monkeypatch.setattr(installer, "verify_countgd_install", fake_verify)

    with pytest.raises(ValueError, match="final-path failure"):
        installer.install_countgd(
            target_dir=target,
            python_executable=Path(sys.executable),
        )

    assert not target.exists()


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


def test_manifest_records_independent_bert_pin() -> None:
    environment = {
        "python": "3.12.2",
        "platform": "Linux",
        "machine": "x86_64",
        "packages": {"torch": "2.11.0+cu128"},
    }
    manifest = installer._build_manifest(
        runtime_tree_sha256="runtime-sha",
        runtime_tree_file_count=133,
        base_python=Path("/qualified/python"),
        environment=environment,
    )

    assert manifest["bert_revision"] == installer._BERT_REVISION
    assert manifest["bert_repository"] == installer._SPACE_REPOSITORY_URL
    assert len(manifest["bert_files"]) == 6
    assert manifest["model_version"] == installer.COUNTGD_MODEL_VERSION
    assert manifest["inference_protocol"] == installer.countgd_model_protocol()
    assert manifest["runtime_tree"]["file_count"] == 133
    assert manifest["runtime_environment"]["inherits_system_site_packages"] is False
    lock = manifest["runtime_environment"]["lock"]
    assert lock == {**environment_lock_payload(), "sha256": environment_lock_digest()}
    assert lock["package_count"] == len(ENVIRONMENT_LOCK) == 76
    assert {
        package["name"]: package["version"] for package in lock["artifacts"]
    } == locked_package_versions()


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
