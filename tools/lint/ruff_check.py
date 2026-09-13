"""`bazel test //:ruff_check`: lint and format-check the tracked Python sources."""

import subprocess
import sys

from tools.lint._ruff_bin import ruff_bin

if __name__ == "__main__":
    ruff = ruff_bin()
    status = 0
    # Only the tracked tree: runfiles also hold rules_python's generated bootstrap.
    roots = ["vrl", "tests", "tools"]
    exclude = ["--exclude", "*_stage2_bootstrap.py,*.venv"]
    for command in (["check", *roots], ["format", "--check", *roots]):
        status |= subprocess.run([ruff, *command, "--no-cache", *exclude], check=False).returncode
    sys.exit(status)
