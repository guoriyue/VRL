"""VRL's MAGI-1 environment preflight against the Bazel-built dedicated interpreter.

The adapter imports the official CLI in the dedicated Python before it will
download weights (``_probe_runtime_environment``). This drives exactly that
step with ``python_executable`` = //third_party/magi_1:python, so the
official stack (torch 2.4/cu124, flash-attn built from source, flashinfer)
is what resolves ``inference/pipeline/entry.py``.
"""

import os
import pathlib
import subprocess
import sys
import unittest

from python.runfiles import runfiles

from vrl.models.families.magi_1.model import (
    MAGI_1_SUPPORTED_SOURCE_REVISION,
    Magi1SubprocessConfig,
    _probe_runtime_environment,
)


class Magi1PreflightTest(unittest.TestCase):
    def test_official_cli_imports_in_the_dedicated_interpreter(self):
        r = runfiles.Create()
        interpreter = r.Rlocation("_main/third_party/magi_1/python")
        source = pathlib.Path(
            r.Rlocation("_main/third_party/MAGI-1/inference/pipeline/entry.py")
        ).parents[2]
        config = Magi1SubprocessConfig(
            source_path=source,
            source_revision=MAGI_1_SUPPORTED_SOURCE_REVISION,
            config_path=source / "example/4.5B/4.5B_base_config.json",
            python_executable=interpreter,
            timeout_seconds=600.0,
        )
        _probe_runtime_environment(config)
        # And the stack itself: the dedicated interpreter reports the pinned torch
        # and a source-built flash-attn, independent of this test's own torch.
        report = subprocess.run(
            [
                interpreter,
                "-c",
                "import torch, flash_attn, flashinfer; print(torch.__version__, flash_attn.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
            env={**os.environ, "PYTHONPATH": ""},
        ).stdout.split()
        self.assertEqual(report, ["2.4.0+cu124", "2.4.2"])
        self.assertNotEqual(report[0], __import__("torch").__version__)


if __name__ == "__main__":
    sys.exit(unittest.main())
