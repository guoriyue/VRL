"""Run the CountGD reward service against the Bazel-assembled runtime tree.

The reward model resolves its source tree from ``VRL_DATA_ROOT``, so this
points that root at the assembled tree in runfiles and hands the remaining
arguments to the service CLI unchanged.
"""

import os
import sys

from python.runfiles import runfiles

from vrl.rewards.service.server import main

if __name__ == "__main__":
    root = runfiles.Create().Rlocation("_main/third_party/countgd/runtime")
    os.environ["VRL_DATA_ROOT"] = root
    sys.exit(main(sys.argv[1:]))
