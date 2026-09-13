"""Replace one distribution's exported requirement line with a built wheel.

uv exports `name==version \\` followed by `--hash=` continuation lines. When a
`rust_wheel` repository built that distribution from source, the line pinning
exactly the built version becomes a `file://` reference to the wheel with the
wheel's own hash. A profile that pins another version of the same distribution
keeps its PyPI line: the lock can hold several versions of one package across
profiles (tokenizers 0.13.3 for VBench, 0.22.x for transformers 5), and the
built wheel must not shadow the other.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def replace(text: str, name: str, version: str, replacement: str) -> str:
    pattern = re.compile(
        r"^"
        + re.escape(name)
        + r"=="
        + re.escape(version)
        + r"[ \t]*\\\n(?:[ \t]+--hash=[^\n]*\\?\n)*",
        re.MULTILINE,
    )
    return pattern.sub(replacement + "\n", text, count=1)


def main(path: str, name: str, version: str, replacement: str) -> None:
    file = Path(path)
    file.write_text(
        replace(file.read_text(encoding="utf-8"), name, version, replacement),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(*sys.argv[1:5])
