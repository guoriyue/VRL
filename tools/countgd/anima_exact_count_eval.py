"""Run the exact-count checkpoint evaluation on the Bazel CountGD runtime tree."""

import os
import sys

from python.runfiles import runfiles

from vrl.scripts.eval.anima_exact_count_checkpoint_eval import main

if __name__ == "__main__":
    os.environ["VRL_DATA_ROOT"] = runfiles.Create().Rlocation("_main/third_party/countgd/runtime")
    sys.exit(main(sys.argv[1:]))
