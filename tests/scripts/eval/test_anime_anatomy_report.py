from __future__ import annotations

import pytest

from vrl.scripts.eval import anime_anatomy_report


def test_help_does_not_require_optional_anatomy_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        anime_anatomy_report,
        "require_rtmw_modules",
        lambda: pytest.fail("optional dependency check ran before argparse --help"),
    )

    with pytest.raises(SystemExit) as exit_info:
        anime_anatomy_report.main(["--help"])

    assert exit_info.value.code == 0
