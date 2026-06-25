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

## Adding a new vendored dependency

```bash
git submodule add <url> third_party/<name>
mkdir third_party/<name>_packaging
# write third_party/<name>_packaging/pyproject.toml pointing at the submodule's
# package dir(s) via [tool.setuptools.packages.find].where  (copy an existing one)
# add a `!third_party/<name>` and `!third_party/<name>_packaging` line to .gitignore
make setup
```
