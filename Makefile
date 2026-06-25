# One-command dev setup. Vendored upstreams (JoyAI-Echo, videophy) ship no
# Python packaging, so they live as git submodules under third_party/ and are
# exposed as importable packages by thin editable-install wrappers. A bare
# `git clone` never fetches submodules, so this target does it + the wrappers in
# one step — run it once after cloning (and again after pulling a submodule bump).
#
# The per-dependency config lives WITH each dependency: every vendored package is
# a `third_party/<name>_packaging/pyproject.toml`. This target just discovers and
# installs them all, so adding a new vendored dep needs no edit here — drop its
# wrapper under third_party/ and `make setup` picks it up. See third_party/README.md.
.PHONY: setup
setup:
	git submodule update --init --recursive
	pip install -e .
	@for wrapper in third_party/*_packaging; do \
		echo "installing vendored wrapper: $$wrapper"; \
		pip install -e "$$wrapper" || exit 1; \
	done
	@echo "setup complete: vendored third_party packages importable"
