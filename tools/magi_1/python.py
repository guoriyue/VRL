"""Act as the MAGI-1 dedicated interpreter (``model.python_executable``).

VRL's MAGI-1 adapter launches ``<python_executable> -c <code> <entry.py>``
with ``PYTHONPATH`` set to the upstream source tree. This target's own
interpreter is the runfiles venv of the MAGI-1 stack (torch 2.4/cu124,
flash-attn built from source), so it re-executes that interpreter with the
caller's arguments and environment unchanged, apart from letting the
caller's PYTHONPATH and current directory reach ``sys.path`` as a plain
Python would.
"""

import os
import sys

if __name__ == "__main__":
    environment = dict(os.environ)
    environment.pop("PYTHONSAFEPATH", None)
    caller_pythonpath = environment.get("VRL_CALLER_PYTHONPATH")
    if caller_pythonpath is not None:
        environment["PYTHONPATH"] = caller_pythonpath
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], environment)
