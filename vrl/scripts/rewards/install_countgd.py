"""Install or verify the pinned CountGD runtime used by VRL.

The installer reconstructs the qualified ignored artifact tree without copying
from an existing local installation: source comes from the pinned GitHub
archive, application assets come from the pinned Hugging Face Space revision,
and every downloaded file is checked before it enters the runtime tree. Builds
happen in a sibling staging directory and are published only after the same
production verifier used by the reward model accepts them.

Examples::

    python -m vrl.scripts.rewards.install_countgd install \
      --python /path/to/python3.12
    python -m vrl.scripts.rewards.install_countgd verify
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.rewards.models.countgd_person_count import (
    _RUNTIME_TREE_ALGORITHM,
    _RUNTIME_TREE_SCHEMA,
    COUNTGD_CHECKPOINT_SHA256,
    COUNTGD_INSTALL_SCHEMA,
    COUNTGD_MODEL_VERSION,
    COUNTGD_RUNTIME_TREE_SHA256,
    COUNTGD_SOURCE_REVISION,
    COUNTGD_SPACE_REVISION,
    _runtime_tree_digest,
    _verify_install,
    countgd_model_protocol,
)
from vrl.scripts.rewards.countgd_environment_lock import (
    ENVIRONMENT_LOCK,
    QUALIFIED_MACHINE,
    QUALIFIED_PIP_VERSION,
    QUALIFIED_PLATFORM,
    QUALIFIED_PYTHON_VERSION,
    ArtifactKind,
    environment_lock_digest,
    environment_lock_payload,
    expected_environment_package_versions,
)
from vrl.utils.artifacts import default_data_root, sha256_file

_SOURCE_REPOSITORY = "https://github.com/niki-amini-naieni/CountGD"
_SOURCE_ARCHIVE_SHA256 = "dcab136e4c1ce9a567a3f67bef084ec3efbcbe15c6f6155755fea4ac48457378"
_SPACE_REPOSITORY = "nikigoli/countgd"
_SPACE_REPOSITORY_URL = "https://huggingface.co/spaces/nikigoli/countgd"
_LEGACY_INSTALL_MANIFEST_SHA256 = (
    "f030350e3be3640652006530e7000699a7decb638355c171432a20c320b5188d"
)

# The qualified BERT serialization is committed inside the Space. Keeping its
# revision separate in the install contract prevents a future checkpoint-only
# Space update from silently moving the text encoder.
_BERT_REVISION = "6e82e59569a84ee5c6aafa35d396f2d2bee57be2"


@dataclass(frozen=True, slots=True)
class _RemoteArtifact:
    filename: str
    revision: str
    sha256: str
    role: str


_SPACE_ARTIFACTS = (
    _RemoteArtifact(
        filename="cfg_app.py",
        revision=COUNTGD_SPACE_REVISION,
        sha256="b7e7061e8f343f054adff6b9417a6acd4e7c155e767c22cf58f533c8d8fd0587",
        role="space_runtime",
    ),
    _RemoteArtifact(
        filename="checkpoint_best_regular.pth",
        revision=COUNTGD_SPACE_REVISION,
        sha256=COUNTGD_CHECKPOINT_SHA256,
        role="checkpoint",
    ),
    _RemoteArtifact(
        filename="checkpoints/bert-base-uncased/config.json",
        revision=_BERT_REVISION,
        sha256="f2375bb637ff0932102389231414227f0517017a8b8d724d084d9229fd5966e1",
        role="bert",
    ),
    _RemoteArtifact(
        filename="checkpoints/bert-base-uncased/model.safetensors",
        revision=_BERT_REVISION,
        sha256="5875f83030335d194f35b15a32e7f4e654aa302aa83af032a3f36d035dcaf8af",
        role="bert",
    ),
    _RemoteArtifact(
        filename="checkpoints/bert-base-uncased/special_tokens_map.json",
        revision=_BERT_REVISION,
        sha256="b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        role="bert",
    ),
    _RemoteArtifact(
        filename="checkpoints/bert-base-uncased/tokenizer.json",
        revision=_BERT_REVISION,
        sha256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        role="bert",
    ),
    _RemoteArtifact(
        filename="checkpoints/bert-base-uncased/tokenizer_config.json",
        revision=_BERT_REVISION,
        sha256="f62a57a75856b93282501c92a86f62b169997c81e93cf6f75b7cc15d6285968e",
        role="bert",
    ),
    _RemoteArtifact(
        filename="checkpoints/bert-base-uncased/vocab.txt",
        revision=_BERT_REVISION,
        sha256="07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        role="bert",
    ),
)


@dataclass(frozen=True, slots=True)
class _Replacement:
    old: str
    new: str
    expected_count: int


@dataclass(frozen=True, slots=True)
class _PatchSpec:
    path: str
    replacements: tuple[_Replacement, ...]
    sha256: str


_COMPATIBILITY_PATCHES = (
    _PatchSpec(
        path="models/GroundingDINO/ops/src/ms_deform_attn.h",
        replacements=(_Replacement("value.type().is_cuda()", "value.is_cuda()", 2),),
        sha256="3a57af9dd16066f1fa347a070d087caaa87c1416e0b9ac448950b5bc05c01615",
    ),
    _PatchSpec(
        path="models/GroundingDINO/ops/src/cuda/ms_deform_attn_cuda.cu",
        replacements=(
            _Replacement(".type().is_cuda()", ".is_cuda()", 11),
            _Replacement(
                "AT_DISPATCH_FLOATING_TYPES(value.type(),",
                "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),",
                2,
            ),
        ),
        sha256="793012a6c918a0daf1bea492940884d02c45e966d40d4067727082478d163190",
    ),
)

# The qualified tree was produced after an upstream Python 3.10 extension build.
# Only these four generated Python copies enter the production tree digest; the
# ABI-specific extension does not run in the qualified CPU service. Recreating
# the copies from their source files preserves the audited 133-file layout
# without manufacturing or trusting an unusable CPython 3.10 binary.
_QUALIFIED_BUILD_LIB = Path(
    "models/GroundingDINO/ops/build/lib.linux-x86_64-cpython-310",
)
_QUALIFIED_BUILD_PYTHON_FILES = (
    "functions/__init__.py",
    "functions/ms_deform_attn_func.py",
    "modules/__init__.py",
    "modules/ms_deform_attn.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install", help="Build and atomically publish CountGD.")
    install.add_argument(
        "--target-dir",
        type=Path,
        default=default_data_root() / "countgd",
        help="Installation root containing source/, env/, and install_manifest.json.",
    )
    install.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Qualified Python 3.12.2 interpreter used to create the isolated venv.",
    )
    install.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help="Optional predownloaded pinned GitHub archive; its SHA-256 is still checked.",
    )
    install.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    install.add_argument(
        "--artifact-cache-dir",
        type=Path,
        default=None,
        help="Optional reusable directory for the hash-locked Python artifacts.",
    )
    install.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require a supplied source archive and already-cached Space artifacts.",
    )

    verify = subparsers.add_parser("verify", help="Run production and environment checks.")
    verify.add_argument(
        "--target-dir",
        type=Path,
        default=default_data_root() / "countgd",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        result = install_countgd(
            target_dir=args.target_dir,
            python_executable=args.python,
            source_archive=args.source_archive,
            hf_cache_dir=args.hf_cache_dir,
            artifact_cache_dir=args.artifact_cache_dir,
            local_files_only=args.local_files_only,
        )
    else:
        result = verify_countgd_install(args.target_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def install_countgd(
    *,
    target_dir: Path,
    python_executable: Path,
    source_archive: Path | None = None,
    hf_cache_dir: Path | None = None,
    artifact_cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Build a fresh install beside ``target_dir`` and publish it atomically."""

    target = target_dir.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"CountGD installation already exists: {target}")
    if local_files_only and source_archive is None:
        raise ValueError("--local-files-only requires --source-archive")
    if local_files_only and artifact_cache_dir is None:
        raise ValueError("--local-files-only requires --artifact-cache-dir")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.install-", dir=target.parent),
    ).resolve()
    staging = temporary_root / "countgd"
    published = False
    try:
        staging.mkdir()
        _build_staged_install(
            staging,
            python_executable=python_executable,
            source_archive=source_archive,
            hf_cache_dir=hf_cache_dir,
            artifact_cache_dir=artifact_cache_dir,
            local_files_only=local_files_only,
        )
        verify_countgd_install(staging)
        if target.exists():
            raise FileExistsError(f"CountGD installation appeared during build: {target}")
        staging.rename(target)
        published = True
        # Verify again from the stable path. If this path-dependent check fails,
        # remove the installation created by this invocation instead of leaving
        # an unqualified target that blocks a clean retry.
        return verify_countgd_install(target)
    except Exception:
        if published:
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def verify_countgd_install(target_dir: Path) -> dict[str, Any]:
    """Verify assets and smoke-test either locked or explicit legacy runtime."""

    target = target_dir.expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"CountGD installation does not exist: {target}")
    source_dir = target / "source"
    _verify_install(source_dir)
    env_python = target / "env" / "bin" / "python"
    if _manifest_uses_legacy_environment(target):
        environment = _verify_legacy_environment(env_python, source_dir)
    else:
        environment = _verify_environment(env_python, source_dir)
    _run_service_smoke(env_python, source_dir)
    return {
        "status": "verified",
        "target_dir": str(target),
        "source_dir": str(source_dir),
        "runtime_tree_sha256": COUNTGD_RUNTIME_TREE_SHA256,
        "checkpoint_sha256": COUNTGD_CHECKPOINT_SHA256,
        "environment": environment,
    }


def _manifest_uses_legacy_environment(target: Path) -> bool:
    manifest_path = target / "install_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        value = manifest["runtime_environment"]["inherits_system_site_packages"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"CountGD install manifest lacks its environment isolation field: {manifest_path}",
        ) from error
    if not isinstance(value, bool):
        raise TypeError(
            "CountGD install manifest runtime_environment."
            "inherits_system_site_packages must be boolean",
        )
    if value:
        _require_sha256(
            manifest_path,
            _LEGACY_INSTALL_MANIFEST_SHA256,
            what="legacy CountGD install manifest",
        )
    else:
        runtime_environment = manifest.get("runtime_environment")
        expected_lock = {
            **environment_lock_payload(),
            "sha256": environment_lock_digest(),
        }
        mismatches: dict[str, Any] = {}
        if manifest.get("model_version") != COUNTGD_MODEL_VERSION:
            mismatches["model_version"] = {
                "expected": COUNTGD_MODEL_VERSION,
                "actual": manifest.get("model_version"),
            }
        expected_protocol = countgd_model_protocol()
        if manifest.get("inference_protocol") != expected_protocol:
            mismatches["inference_protocol"] = {
                "expected": expected_protocol,
                "actual": manifest.get("inference_protocol"),
            }
        actual_lock = (
            runtime_environment.get("lock") if isinstance(runtime_environment, Mapping) else None
        )
        if actual_lock != expected_lock:
            actual_lock_sha256 = (
                actual_lock.get("sha256") if isinstance(actual_lock, Mapping) else None
            )
            actual_payload = (
                {key: item for key, item in actual_lock.items() if key != "sha256"}
                if isinstance(actual_lock, Mapping)
                else None
            )
            mismatches["runtime_environment.lock"] = {
                "expected_sha256": environment_lock_digest(),
                "actual_sha256": actual_lock_sha256,
                "payload_matches": actual_payload == environment_lock_payload(),
            }
        if mismatches:
            raise ValueError(f"CountGD locked install manifest mismatch: {mismatches}")
    return value


def _build_staged_install(
    staging: Path,
    *,
    python_executable: Path,
    source_archive: Path | None,
    hf_cache_dir: Path | None,
    artifact_cache_dir: Path | None,
    local_files_only: bool,
) -> None:
    source_dir = staging / "source"
    archive_path = staging / "countgd-source.tar.gz"
    if source_archive is None:
        _download_url(
            _source_archive_url(),
            archive_path,
            expected_sha256=_SOURCE_ARCHIVE_SHA256,
        )
    else:
        source_archive = source_archive.expanduser().resolve()
        if not source_archive.is_file():
            raise FileNotFoundError(f"CountGD source archive does not exist: {source_archive}")
        shutil.copyfile(source_archive, archive_path)
    _require_sha256(archive_path, _SOURCE_ARCHIVE_SHA256, what="CountGD source archive")
    _extract_source_archive(archive_path, source_dir)
    archive_path.unlink()

    _download_space_artifacts(
        source_dir,
        cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )
    _apply_compatibility_patches(source_dir)
    _restore_qualified_build_layout(source_dir)

    base_python = python_executable.expanduser().resolve()
    _verify_base_interpreter(base_python)
    _build_environment(
        base_python,
        staging / "env",
        artifact_cache_dir=artifact_cache_dir,
        local_files_only=local_files_only,
    )
    environment = _verify_environment(staging / "env" / "bin" / "python", source_dir)

    runtime_tree = _runtime_tree_digest(source_dir)
    if runtime_tree.sha256 != COUNTGD_RUNTIME_TREE_SHA256:
        raise ValueError(
            "rebuilt CountGD runtime tree differs from the qualified canonical tree: "
            f"expected={COUNTGD_RUNTIME_TREE_SHA256}, actual={runtime_tree.sha256}",
        )
    manifest = _build_manifest(
        runtime_tree_sha256=runtime_tree.sha256,
        runtime_tree_file_count=runtime_tree.file_count,
        base_python=base_python,
        environment=environment,
    )
    (staging / "install_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _verify_install(source_dir)


def _source_archive_url() -> str:
    return (
        f"https://codeload.github.com/niki-amini-naieni/CountGD/tar.gz/{COUNTGD_SOURCE_REVISION}"
    )


def _download_url(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    attempts: int = 3,
) -> None:
    """Download atomically with bounded socket waits and exact integrity."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        partial = destination.with_name(
            f".{destination.name}.part-{uuid.uuid4().hex}",
        )
        try:
            print(f"Downloading {url} (attempt {attempt}/{attempts})", flush=True)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "VRL CountGD installer"},
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected_sha256 is not None:
                _require_sha256(
                    partial,
                    expected_sha256,
                    what=f"downloaded artifact {destination.name}",
                )
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(f"Retrying {url} after failed attempt {attempt}", flush=True)


def _extract_source_archive(archive_path: Path, source_dir: Path) -> None:
    extraction_root = archive_path.parent / "source-unpacked"
    extraction_root.mkdir()
    expected_root = f"CountGD-{COUNTGD_SOURCE_REVISION}"
    with tarfile.open(archive_path, "r:gz") as archive:
        roots = {
            Path(member.name).parts[0]
            for member in archive.getmembers()
            if Path(member.name).parts
        }
        if roots != {expected_root}:
            raise ValueError(f"unexpected CountGD source archive roots: {sorted(roots)}")
        archive.extractall(extraction_root, filter="data")
    extracted = extraction_root / expected_root
    if not extracted.is_dir():
        raise ValueError("CountGD source archive did not produce its pinned root directory")
    extracted.rename(source_dir)
    extraction_root.rmdir()


def _download_space_artifacts(
    source_dir: Path,
    *,
    cache_dir: Path | None,
    local_files_only: bool,
) -> None:
    from huggingface_hub import hf_hub_download

    for artifact in _SPACE_ARTIFACTS:
        print(
            f"Resolving {_SPACE_REPOSITORY}/{artifact.filename}@{artifact.revision}",
            flush=True,
        )
        kwargs: dict[str, Any] = {
            "repo_id": _SPACE_REPOSITORY,
            "filename": artifact.filename,
            "repo_type": "space",
            "revision": artifact.revision,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir.expanduser().resolve())
        cached_path = Path(hf_hub_download(**kwargs))
        destination = source_dir / artifact.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, destination)
        _require_sha256(
            destination,
            artifact.sha256,
            what=f"CountGD {artifact.role} artifact {artifact.filename}",
        )


def _apply_compatibility_patches(source_dir: Path) -> None:
    for patch in _COMPATIBILITY_PATCHES:
        path = source_dir / patch.path
        text = path.read_text(encoding="utf-8")
        text = _replace_text_exact(text, patch.replacements, context=patch.path)
        path.write_text(text, encoding="utf-8")
        _require_sha256(path, patch.sha256, what=f"patched CountGD source {patch.path}")


def _replace_text_exact(
    text: str,
    replacements: Sequence[_Replacement],
    *,
    context: str,
) -> str:
    for replacement in replacements:
        actual_count = text.count(replacement.old)
        if actual_count != replacement.expected_count:
            raise ValueError(
                f"CountGD patch input drift for {context}: expected "
                f"{replacement.expected_count} occurrences of {replacement.old!r}, "
                f"found {actual_count}",
            )
        text = text.replace(replacement.old, replacement.new)
    return "\n".join(text.splitlines()).rstrip("\n") + "\n"


def _restore_qualified_build_layout(source_dir: Path) -> None:
    source_root = source_dir / "models/GroundingDINO/ops"
    build_root = source_dir / _QUALIFIED_BUILD_LIB
    for relative_text in _QUALIFIED_BUILD_PYTHON_FILES:
        relative = Path(relative_text)
        source = source_root / relative
        destination = build_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        _require_sha256(
            destination,
            sha256_file(source),
            what=f"qualified CountGD build copy {relative_text}",
        )


def _verify_base_interpreter(python_executable: Path) -> None:
    if not python_executable.is_file():
        raise FileNotFoundError(f"CountGD base interpreter does not exist: {python_executable}")
    record = _query_python_platform(python_executable)
    errors: list[str] = []
    if record["python"] != QUALIFIED_PYTHON_VERSION:
        errors.append(
            f"python expected={QUALIFIED_PYTHON_VERSION} actual={record['python']}",
        )
    if record["platform"] != QUALIFIED_PLATFORM:
        errors.append(
            f"platform expected={QUALIFIED_PLATFORM} actual={record['platform']}",
        )
    if record["machine"] != QUALIFIED_MACHINE:
        errors.append(f"machine expected={QUALIFIED_MACHINE} actual={record['machine']}")
    if errors:
        raise ValueError("CountGD base interpreter is not qualified: " + "; ".join(errors))


def _build_environment(
    base_python: Path,
    env_dir: Path,
    *,
    artifact_cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> None:
    _run_command([str(base_python), "-m", "venv", str(env_dir)])
    env_python = env_dir / "bin" / "python"
    bootstrap = _query_environment(env_python)
    if bootstrap["packages"].get("pip") != QUALIFIED_PIP_VERSION:
        raise ValueError(
            "CountGD venv bootstrap pip mismatch: "
            f"expected={QUALIFIED_PIP_VERSION} "
            f"actual={bootstrap['packages'].get('pip')}",
        )

    owns_cache = artifact_cache_dir is None
    if owns_cache:
        artifact_dir = Path(
            tempfile.mkdtemp(prefix=".countgd-artifacts-", dir=env_dir.parent),
        ).resolve()
    else:
        assert artifact_cache_dir is not None
        artifact_dir = artifact_cache_dir.expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = _resolve_locked_artifacts(
            artifact_dir,
            local_files_only=local_files_only,
        )
        wheels = [
            str(path)
            for distribution, path in zip(ENVIRONMENT_LOCK, artifacts, strict=True)
            if distribution.artifact_kind is ArtifactKind.WHEEL
        ]
        _run_command(
            [
                str(env_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-index",
                "--no-deps",
                *wheels,
            ],
        )
        for distribution, path in zip(ENVIRONMENT_LOCK, artifacts, strict=True):
            if distribution.artifact_kind is not ArtifactKind.SDIST:
                continue
            _run_command(
                [
                    str(env_python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--no-index",
                    "--no-deps",
                    "--no-build-isolation",
                    str(path),
                ],
            )
        _run_command([str(env_python), "-m", "pip", "check"])
    finally:
        if owns_cache:
            shutil.rmtree(artifact_dir, ignore_errors=True)


def _resolve_locked_artifacts(
    artifact_dir: Path,
    *,
    local_files_only: bool,
) -> list[Path]:
    paths: list[Path] = []
    for distribution in ENVIRONMENT_LOCK:
        path = artifact_dir / distribution.filename
        if path.exists():
            if not path.is_file():
                raise ValueError(f"CountGD package artifact is not a file: {path}")
            _require_sha256(
                path,
                distribution.sha256,
                what=f"cached CountGD package artifact {distribution.filename}",
            )
        elif local_files_only:
            raise FileNotFoundError(
                f"CountGD package artifact is not cached: {path}",
            )
        else:
            _download_url(
                distribution.url,
                path,
                expected_sha256=distribution.sha256,
            )
        paths.append(path)
    return paths


def _verify_environment(env_python: Path, source_dir: Path) -> dict[str, Any]:
    if not env_python.is_file():
        raise FileNotFoundError(f"CountGD environment interpreter is missing: {env_python}")
    if _venv_inherits_system_site_packages(env_python):
        raise ValueError("CountGD locked environment unexpectedly inherits system packages")
    expected_packages = expected_environment_package_versions()
    record = _query_environment(env_python)
    errors = [
        f"{name} expected={version} actual={record['packages'].get(name)}"
        for name, version in expected_packages.items()
        if record["packages"].get(name) != version
    ]
    unexpected_packages = sorted(set(record["packages"]) - set(expected_packages))
    if unexpected_packages:
        errors.append(f"unexpected packages={unexpected_packages}")
    if record["python"] != QUALIFIED_PYTHON_VERSION:
        errors.append(
            f"python expected={QUALIFIED_PYTHON_VERSION} actual={record['python']}",
        )
    if record["platform"] != QUALIFIED_PLATFORM:
        errors.append(
            f"platform expected={QUALIFIED_PLATFORM} actual={record['platform']}",
        )
    if record["machine"] != QUALIFIED_MACHINE:
        errors.append(f"machine expected={QUALIFIED_MACHINE} actual={record['machine']}")
    if errors:
        raise ValueError("CountGD environment package mismatch: " + "; ".join(errors))

    _run_command([str(env_python), "-m", "pip", "check"])
    _run_model_import_smoke(env_python, source_dir)
    record["qualification_mode"] = "locked-isolated"
    record["inherits_system_site_packages"] = False
    record["environment_lock_sha256"] = environment_lock_digest()
    return record


def _verify_legacy_environment(env_python: Path, source_dir: Path) -> dict[str, Any]:
    """Behavior-check the pre-lock environment without claiming isolation."""

    if not env_python.is_file():
        raise FileNotFoundError(f"CountGD environment interpreter is missing: {env_python}")
    if not _venv_inherits_system_site_packages(env_python):
        raise ValueError(
            "CountGD manifest declares a legacy system-site environment, but pyvenv.cfg does not",
        )
    record = _query_python_platform(env_python)
    errors: list[str] = []
    if record["python"] != QUALIFIED_PYTHON_VERSION:
        errors.append(
            f"python expected={QUALIFIED_PYTHON_VERSION} actual={record['python']}",
        )
    if record["platform"] != QUALIFIED_PLATFORM:
        errors.append(
            f"platform expected={QUALIFIED_PLATFORM} actual={record['platform']}",
        )
    if record["machine"] != QUALIFIED_MACHINE:
        errors.append(f"machine expected={QUALIFIED_MACHINE} actual={record['machine']}")
    if errors:
        raise ValueError("CountGD legacy environment mismatch: " + "; ".join(errors))
    _run_model_import_smoke(env_python, source_dir)
    record["qualification_mode"] = "legacy-system-site-packages"
    record["inherits_system_site_packages"] = True
    return record


def _venv_inherits_system_site_packages(env_python: Path) -> bool:
    config_path = env_python.parent.parent / "pyvenv.cfg"
    if not config_path.is_file():
        raise FileNotFoundError(f"CountGD venv config is missing: {config_path}")
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip().lower()] = value.strip().lower()
    raw = values.get("include-system-site-packages")
    if raw not in {"true", "false"}:
        raise ValueError(
            "CountGD venv config lacks a boolean include-system-site-packages field",
        )
    return raw == "true"


def _run_model_import_smoke(env_python: Path, source_dir: Path) -> None:
    smoke = (
        "import sys\n"
        f"sys.path.insert(0, {str(source_dir)!r})\n"
        "import datasets_inference.transforms\n"
        "from models.registry import MODULE_BUILD_FUNCS\n"
        "from util.slconfig import SLConfig\n"
        "assert MODULE_BUILD_FUNCS.get('groundingdino') is not None\n"
        "assert SLConfig is not None\n"
    )
    _run_command([str(env_python), "-c", smoke])


def _run_service_smoke(env_python: Path, source_dir: Path) -> None:
    package_parent = Path(__file__).resolve().parents[3]
    smoke = f"""
import asyncio
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, {str(package_parent)!r})

import aiohttp
import yaml
from PIL import Image

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.service.server import _load_service
from vrl.rewards.service.wire import request_to_wire, score_response_from_wire

source_dir = Path({str(source_dir)!r})

async def run() -> None:
    with tempfile.TemporaryDirectory(prefix="countgd-service-smoke-") as raw_root:
        root = Path(raw_root)
        image_path = root / "blank.png"
        Image.new("RGB", (64, 64), color=(127, 127, 127)).save(image_path)
        payload = image_path.read_bytes()
        config_path = root / "service.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {{
                    "host": "127.0.0.1",
                    "port": 0,
                    "model_name": "countgd-install-smoke",
                    "model_version": {COUNTGD_MODEL_VERSION!r},
                    "artifact_roots": [str(root)],
                    "worker_config": {{
                        "model_factory": "vrl.rewards.models.countgd_person_count:CountGDPersonCountModel",
                        "reward_model_version": {COUNTGD_MODEL_VERSION!r},
                        "device": "cpu",
                        "source_dir": str(source_dir),
                    }},
                }},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        service = _load_service(config_path)
        try:
            await service.start()
            host, port = service.address
            request = RewardInferenceRequest(
                request_id="install-smoke",
                artifacts=(
                    RewardInferenceArtifact(
                        artifact_id="blank",
                        sample_id="blank",
                        path=str(image_path),
                        prompt="one person",
                        size_bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        metadata={{"expected_people": 1}},
                    ),
                ),
            )
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"http://{{host}}:{{port}}/info") as response:
                    info = await response.json()
                    assert response.status == 200, info
                    assert info["info"]["model_name"] == "countgd-install-smoke"
                    assert info["info"]["model_version"] == {COUNTGD_MODEL_VERSION!r}
                async with session.post(
                    f"http://{{host}}:{{port}}/score",
                    json=request_to_wire(request),
                ) as response:
                    body = await response.json()
                    assert response.status == 200, body
                    results = score_response_from_wire(
                        body,
                        expected_request_id=request.request_id,
                    )
                    assert len(results) == 1
                    assert "countgd_person_count" in results[0].scores
        finally:
            await service.shutdown_async()

asyncio.run(run())
"""
    _run_command([str(env_python), "-c", smoke])


def _query_environment(python_executable: Path) -> dict[str, Any]:
    script = (
        "import json, platform, re\n"
        "from importlib.metadata import distributions\n"
        "canonical = lambda name: re.sub(r'[-_.]+', '-', name).lower()\n"
        "packages = {}\n"
        "duplicates = []\n"
        "for dist in distributions():\n"
        "  name = canonical(dist.metadata['Name'])\n"
        "  if name in packages:\n"
        "    duplicates.append((name, packages[name], dist.version))\n"
        "  packages[name] = dist.version\n"
        "print(json.dumps({\n"
        "  'python': platform.python_version(),\n"
        "  'platform': platform.system(),\n"
        "  'machine': platform.machine(),\n"
        "  'packages': packages,\n"
        "  'duplicates': duplicates,\n"
        "}, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", script],
        check=True,
        capture_output=True,
        cwd=python_executable.parent,
        text=True,
    )
    record = json.loads(completed.stdout)
    if not isinstance(record, dict) or not isinstance(record.get("packages"), dict):
        raise TypeError("CountGD environment probe returned an invalid record")
    if record.get("duplicates"):
        raise ValueError(
            f"CountGD environment contains duplicate distributions: {record['duplicates']}",
        )
    record.pop("duplicates", None)
    return record


def _query_python_platform(python_executable: Path) -> dict[str, Any]:
    script = (
        "import json, platform\n"
        "print(json.dumps({\n"
        "  'python': platform.python_version(),\n"
        "  'platform': platform.system(),\n"
        "  'machine': platform.machine(),\n"
        "}, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", script],
        check=True,
        capture_output=True,
        cwd=python_executable.parent,
        text=True,
    )
    record = json.loads(completed.stdout)
    if not isinstance(record, dict):
        raise TypeError("CountGD Python platform probe returned an invalid record")
    return record


def _run_command(command: Sequence[str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(list(command), check=True)


def _build_manifest(
    *,
    runtime_tree_sha256: str,
    runtime_tree_file_count: int,
    base_python: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    bert_files = {
        artifact.filename: artifact.sha256
        for artifact in _SPACE_ARTIFACTS
        if artifact.role == "bert"
    }
    return {
        "schema": COUNTGD_INSTALL_SCHEMA,
        "source_repository": _SOURCE_REPOSITORY,
        "source_revision": COUNTGD_SOURCE_REVISION,
        "source_archive_url": _source_archive_url(),
        "source_archive_sha256": _SOURCE_ARCHIVE_SHA256,
        "checkpoint_repository": _SPACE_REPOSITORY_URL,
        "space_revision": COUNTGD_SPACE_REVISION,
        "checkpoint_file": "source/checkpoint_best_regular.pth",
        "checkpoint_sha256": COUNTGD_CHECKPOINT_SHA256,
        "model_version": COUNTGD_MODEL_VERSION,
        "inference_protocol": countgd_model_protocol(),
        "bert_repository": _SPACE_REPOSITORY_URL,
        "bert_revision": _BERT_REVISION,
        "bert_files": bert_files,
        "runtime_tree": {
            "schema": _RUNTIME_TREE_SCHEMA,
            "algorithm": _RUNTIME_TREE_ALGORITHM,
            "file_count": runtime_tree_file_count,
            "sha256": runtime_tree_sha256,
        },
        "compatibility_patches": [
            {
                "path": patch.path,
                "sha256": patch.sha256,
                "replacements": [
                    {
                        "old": replacement.old,
                        "new": replacement.new,
                        "count": replacement.expected_count,
                    }
                    for replacement in patch.replacements
                ],
            }
            for patch in _COMPATIBILITY_PATCHES
        ],
        "runtime_environment": {
            "python": environment["python"],
            "platform": environment["platform"],
            "machine": environment["machine"],
            "base_interpreter": str(base_python),
            "inherits_system_site_packages": False,
            "lock": {
                **environment_lock_payload(),
                "sha256": environment_lock_digest(),
            },
            "qualified_device": "cpu",
        },
    }


def _require_sha256(path: Path, expected: str, *, what: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{what} SHA-256 mismatch: expected={expected}, actual={actual}")


if __name__ == "__main__":
    main()
