"""Maintained Markdown launch commands compile against the bundled config schema.

Scope is docs/*.md and docs/sprints/info/*.md. Fences explicitly tagged
`bash historical` retain recorded run commands. External standalone scorer YAMLs
and dynamic shell variables are not bundled RootConfig recipes and are excluded.
No documented command is executed and no model or reward backend is loaded.
"""

import re
import shlex
from pathlib import Path

import pytest

from vrl.config.loading import load_config
from vrl.config.schema import parse_config


def _compile_documented_commands(paths):
    compiled = []
    for path in paths:
        document = path.read_text()
        for fence in re.finditer(
            r"^```(?:bash|sh|shell)([^\n]*)\n(.*?)^```", document, re.M | re.S
        ):
            if "historical" in fence.group(1).split():
                continue
            line_number = document[: fence.start()].count("\n") + 1
            for line in fence.group(2).replace("\\\n", " ").splitlines():
                if "-m" not in line or "--config" not in line:
                    continue
                tokens = shlex.split(line, comments=True)
                config_tokens = [
                    index
                    for index, token in enumerate(tokens)
                    if token == "--config" or token.startswith("--config=")
                ]
                if "-m" not in tokens or not config_tokens:
                    continue
                if len(config_tokens) != 1:
                    raise AssertionError(f"{path}:{line_number}: ambiguous --config options")
                module = tokens[tokens.index("-m") + 1]
                if not module.startswith("vrl.scripts."):
                    continue
                config_index = config_tokens[0]
                config = (
                    tokens[config_index].split("=", 1)[1]
                    if "=" in tokens[config_index]
                    else tokens[config_index + 1]
                )
                if not config.startswith(("experiment/", "recipe/")):
                    continue
                overrides = [
                    token
                    for token in tokens[tokens.index("-m") + 2 :]
                    if re.match(r"\+?[A-Za-z_][\w./]*=", token)
                ]
                try:
                    parse_config(load_config(config, overrides=overrides))
                except Exception as error:
                    raise AssertionError(
                        f"{path}:{line_number}: {module} --config {config}: {error}"
                    ) from error
                compiled.append(path)
    return compiled


def test_maintained_documented_launches_compile_without_loading_models():
    docs = Path(__file__).resolve().parents[2] / "docs"
    compiled = _compile_documented_commands(
        [*sorted(docs.glob("*.md")), *sorted((docs / "sprints/info").glob("*.md"))]
    )
    assert any(path.parent == docs for path in compiled)
    assert any(path.parent == docs / "sprints/info" for path in compiled)


def test_document_gate_catches_removed_override_keys_and_preserves_marked_history(tmp_path):
    path = tmp_path / "example.md"
    path.write_text(
        "```bash\npython -m vrl.scripts.train --config=experiment/sd3_5/online_grpo_pickscore actor.optim.lr=1e-5\n```\n"
    )
    assert _compile_documented_commands([path]) == [path]
    command = "python -m vrl.scripts.train --config experiment/sd3_5/online_grpo_pickscore \\\n  production.kling_video_reward.enabled=false\n"
    path.write_text("```bash\n" + command + "```\n")
    with pytest.raises(AssertionError, match="production"):
        _compile_documented_commands([path])
    path.write_text("```bash historical\n" + command + "```\n")
    assert _compile_documented_commands([path]) == []
