# third_party/

Vendored upstream code that ships **no Python packaging**, so it cannot be
`pip install`ed from PyPI and must be made importable locally.

## Convention

A vendored dependency is **a git submodule** — the upstream source, pinned to a
commit (see `../.gitmodules`). E.g. `joyai_echo/`, `videophy/`.

A single **editable-install wrapper**, `third_party/pyproject.toml`, exposes
every submodule's un-packaged `src` tree as a real importable package via
`[tool.setuptools.packages.find]` (`where` = the src roots, `include` = the
package names). `pip install -e third_party` makes them all importable, so the
main repo (`vrl/`) needs no `sys.path` injection. `make setup` (repo root) runs
this for you after fetching the submodules.

## Current vendored packages

| submodule        | exposes                                                       |
| ---------------- | ------------------------------------------------------------ |
| `joyai_echo`     | `ltx_core`, `ltx_pipelines`, `ltx_distillation`              |
| `videophy`       | `mplug_owl_video`                                            |
| `PhyMotion`      | _(not imported — run via CLI)_ `astrolabe.rewards` via `vrl/scripts/eval/phymotion_score.py` |
| `VMBench`        | _(not imported — run via CLI)_ motion-eval benchmark; fold scores in with `--merge-json` |
| `DynamicEval`    | _(not imported — run via CLI)_ dynamic-scene eval; fold scores in with `--merge-json` |

Not every vendored repo is exposed through `third_party/pyproject.toml`: the
wrapper lists only submodules that `vrl/` **imports** in-process. The three
motion-eval benchmarks above are invoked as external commands (their own CLIs,
or the PhyMotion bridge run in PhyMotion's own conda env), so they are vendored
to pin the code but stay out of the editable install — `make setup` simply
skips them.

## Adding a new vendored dependency

```bash
git submodule add <url> third_party/<name>
# In third_party/pyproject.toml: add the submodule's src root(s) to
#   [tool.setuptools.packages.find].where  and the package name(s) to .include
# In .gitignore: add `!third_party/<name>` (pyproject.toml is already whitelisted)
make setup
```
