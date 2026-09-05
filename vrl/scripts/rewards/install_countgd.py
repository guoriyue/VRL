"""Verify the pinned CountGD runtime used by VRL.

Validate the runtime assets and manifest provenance, qualify the isolated or
explicit legacy Python environment, and smoke-test the production reward service.

Example::

    python -m vrl.scripts.rewards.install_countgd verify
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vrl.rewards.models.countgd_person_count import (
    COUNTGD_CHECKPOINT_SHA256,
    COUNTGD_MODEL_VERSION,
    COUNTGD_RUNTIME_TREE_SHA256,
    _verify_install,
    countgd_model_protocol,
)
from vrl.scripts.rewards.countgd_environment_lock import (
    QUALIFIED_MACHINE,
    QUALIFIED_PLATFORM,
    QUALIFIED_PYTHON_VERSION,
    environment_lock_digest,
    environment_lock_payload,
    expected_environment_package_versions,
)
from vrl.utils.artifacts import default_data_root, sha256_file

_LEGACY_INSTALL_MANIFEST_SHA256 = (
    "f030350e3be3640652006530e7000699a7decb638355c171432a20c320b5188d"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Run production and environment checks.")
    verify.add_argument(
        "--target-dir",
        type=Path,
        default=default_data_root() / "countgd",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = verify_countgd_install(args.target_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


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


def _require_sha256(path: Path, expected: str, *, what: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{what} SHA-256 mismatch: expected={expected}, actual={actual}")


if __name__ == "__main__":
    main()
