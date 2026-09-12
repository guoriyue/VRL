"""Config lint: run `python -m vrl.config.lint` from the repo root.

One machine sweep, no LLM involved: compose every bundled experiment YAML and
parse it through the typed schema — the same gate every entrypoint runs. A
``???`` value is a launch-time decision (``+reward=...``, ``actor.optim.lr=``),
so a template's mandatory leaves are left out of the question; every other
key, type, and required field is checked, and any failure names the experiment
and the parse error. Exit code 0 = clean; 1 = findings (suitable for CI /
pre-commit). tests/config/test_unknown_keys.py runs the same sweep.
"""

from __future__ import annotations

import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from vrl.config.base import _extract_error_message


def _without_mandatory_values(cfg: DictConfig) -> tuple[Any, set[str]]:
    """Replace every ``???`` leaf with ``None`` and return the affected paths."""

    missing = set(OmegaConf.missing_keys(cfg))
    plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: (None if value == "???" else strip(value)) for key, value in node.items()}
        if isinstance(node, list):
            return [strip(value) for value in node]
        return node

    return strip(plain), missing


def experiment_parse_error(cfg: DictConfig) -> str | None:
    """The parse error for one composed config, or ``None`` when it parses.

    Errors caused only by a ``???`` leaf (a required value the template leaves
    to launch time) are not errors here; an unknown key beside it still is.
    """

    from vrl.config.schema import RootConfig

    plain, missing = _without_mandatory_values(cfg)
    try:
        RootConfig.model_validate(plain)
    except ValidationError as error:
        retained = []
        for detail in error.errors(include_url=False):
            message = _extract_error_message(
                ValidationError.from_exception_data(error.title, [detail]),
            )
            if any(
                message == f"config missing required field: {path}"
                or message.startswith(f"{path}: ")
                for path in missing
            ):
                continue
            retained.append(detail)
        if retained:
            return _extract_error_message(
                ValidationError.from_exception_data(error.title, retained),
            )
    return None


def experiment_parse_failures() -> dict[str, str]:
    """Config sweep: parse error per experiment, for every shipped experiment."""

    from vrl.config.loading import compose_config, list_bundled_configs

    failures: dict[str, str] = {}
    for logical_name in list_bundled_configs("experiment"):
        name = logical_name.removeprefix("experiment/").removesuffix(".yaml")
        error = experiment_parse_error(compose_config(f"experiment/{name}"))
        if error is not None:
            failures[name] = error
    return failures


def main() -> int:
    failures = experiment_parse_failures()
    if failures:
        print("✗ experiment configs fail to parse:")
        for name, error in failures.items():
            print(f"    {name}: {error}")
        print("  fix: typo/dead key in YAML, or a missing schema declaration")
        return 1
    print("✓ config sweep: every experiment config parses through the typed schema")
    return 0


if __name__ == "__main__":
    sys.exit(main())
