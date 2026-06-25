# third_party/

Vendored upstream code that ships **no Python packaging**, so it cannot be
`pip install`ed from PyPI and must be made importable locally.

## Convention

Each vendored dependency is two things, both living here:

1. **A git submodule** — the upstream source, pinned to a commit (see
   `../.gitmodules`). E.g. `joyai_echo/`, `videophy/`.
2. **An editable-install wrapper** — `<name>_packaging/pyproject.toml` that
   declares which packages the submodule's (un-packaged) `src` trees expose, so
   `pip install -e <name>_packaging` makes them importable. This keeps the main
   repo (`vrl/`) free of any `sys.path` injection.

`make setup` (repo root) discovers and editable-installs **every**
`third_party/*_packaging/` wrapper automatically, so adding a new vendored
dependency needs no change outside this folder — just add its submodule and its
`<name>_packaging/` wrapper here.

## Current vendored packages

| submodule        | wrapper                  | exposes                                           |
| ---------------- | ------------------------ | ------------------------------------------------- |
| `joyai_echo`     | `echo_packaging`         | `ltx_core`, `ltx_pipelines`, `ltx_distillation`   |
| `videophy`       | `videophy_packaging`     | `mplug_owl_video`                                 |
| `PhyMotion`      | _(none — run via CLI)_   | `astrolabe.rewards` via `vrl/scripts/eval/phymotion_score.py` |
| `VMBench`        | _(none — run via CLI)_   | motion-eval benchmark; fold scores in with `--merge-json` |
| `DynamicEval`    | _(none — run via CLI)_   | dynamic-scene eval; fold scores in with `--merge-json` |

Not every vendored repo needs a `*_packaging` wrapper: the wrapper exists only
when `vrl/` **imports** the submodule's source in-process. The three motion-eval
benchmarks above are invoked as external commands (their own CLIs, or the
PhyMotion bridge run in PhyMotion's own conda env), so they are vendored to pin
the code but need no editable install — `make setup` simply skips them.

## Adding a new vendored dependency

```bash
git submodule add <url> third_party/<name>
mkdir third_party/<name>_packaging
# write third_party/<name>_packaging/pyproject.toml pointing at the submodule's
# package dir(s) via [tool.setuptools.packages.find].where  (copy an existing one)
# add a `!third_party/<name>` and `!third_party/<name>_packaging` line to .gitignore
make setup
```
