# One-command dev setup. Vendored upstreams (JoyAI-Echo, videophy) ship no
# Python packaging, so they live as git submodules under third_party/ and are
# exposed as importable packages by thin editable-install wrappers. A bare
# `git clone` never fetches submodules, so this target does it + the wrappers in
# one step — run it once after cloning (and again after pulling a submodule bump).
#
# A single `third_party/pyproject.toml` exposes every submodule's un-packaged src
# tree as importable packages (one editable install for all of them). Adding a new
# vendored dep = add its submodule + one `where`/`include` entry in that file. See
# third_party/README.md.
.PHONY: setup
setup:
	git submodule update --init --recursive
	pip install -e .
	pip install -e third_party
	@echo "setup complete: vendored third_party packages importable"
