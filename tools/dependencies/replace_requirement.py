"""Replace one distribution's exported requirement line with a built wheel.

uv exports `name==version \\` followed by `--hash=` continuation lines. When a
`rust_wheel` repository built that distribution from source, its line becomes
a `file://` reference to the wheel with the wheel's own hash.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def replace(text: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        r"^" + re.escape(name) + r"==[^\n]*\\\n(?:[ \t]+--hash=[^\n]*\\?\n)*",
        re.MULTILINE,
    )
    return pattern.sub(replacement + "\n", text, count=1)


def main(path: str, name: str, replacement: str) -> None:
    file = Path(path)
    file.write_text(replace(file.read_text(encoding="utf-8"), name, replacement), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
