"""Distribution contents are part of the public config-loading contract."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_wheel_and_sdist_include_layered_configs(tmp_path: Path) -> None:
    """Both release formats carry the canonical three-layer config tree."""

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(tmp_path)],
        cwd=repo_root,
        check=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())

    sdist = next(tmp_path.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        sdist_names = set(archive.getnames())

    expected = "configs/experiment/diffusion/sd3_5/online_grpo_ocr.yaml"
    assert f"vrl/{expected}" in wheel_names
    assert any(name.endswith(f"/{expected}") for name in sdist_names)
