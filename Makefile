# Bazel owns dependencies, builds, tests and entry points; see
# docs/research/bazel_migration.md. The only setup step Bazel cannot do is
# fetching git submodules (vendored upstreams reached through //third_party).
.PHONY: setup verify
setup:
	git submodule update --init --recursive
	bazel build //:vrl_train //:vrl_reward_service
	@echo "setup complete: bazel run //:vrl_train -- --config <experiment>"

# The local gate is the CI gate: lint, config contracts, dead-flag audit,
# every CPU pytest lane, the local Ray lane and the torchrun launch check.
# GPU and real-weight lanes are manual: bazel test --config=gpu //tests:gpu_tests
verify:
	uv lock --check
	bazel test //...
