"""Run pytest inside a Bazel test action.

Bazel passes the selected test files as arguments. The workspace's
pyproject.toml supplies the pytest configuration (markers, asyncio mode), and
every write goes under TEST_TMPDIR so the sandbox stays read-only.
"""

import os
import sys

import pytest

if __name__ == "__main__":
    tmp = os.environ.get("TEST_TMPDIR", os.getcwd())
    os.environ.setdefault("HOME", tmp)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tmp, "cache"))
    os.environ.setdefault("HF_HOME", os.path.join(tmp, "hf"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tmp, "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(tmp, "inductor"))
    args = [
        "-c",
        "pyproject.toml",
        "--rootdir",
        os.getcwd(),
        "-p",
        "no:cacheprovider",
        "-q",
        *sys.argv[1:],
    ]
    sys.exit(pytest.main(args))
