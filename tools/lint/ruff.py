"""Run the locked ruff from the pypi hub (bazel run //:ruff -- check vrl)."""

import os
import sys

from tools.lint._ruff_bin import ruff_bin

if __name__ == "__main__":
    os.chdir(os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd()))
    os.execv(ruff_bin(), [ruff_bin(), *sys.argv[1:]])
