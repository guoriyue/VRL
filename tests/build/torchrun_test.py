"""Launch a real two-rank torchrun job from the Bazel interpreter."""

import os
import subprocess
import sys
import unittest


class TorchrunTest(unittest.TestCase):
    def test_two_gloo_ranks_import_vrl_from_runfiles(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCHELASTIC_ERROR_FILE="")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=2",
                "--max-restarts=0",
                "--module",
                "tests.build.torchrun_rank",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("rank 0/2 ok", result.stdout)
        self.assertIn("rank 1/2 ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
