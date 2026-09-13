"""pytest-backed test targets that run from Bazel runfiles only."""

load("@rules_python//python:defs.bzl", "py_test")

def vrl_pytest(name, srcs, deps = [], data = [], args = [], env = {}, **kwargs):
    """Run the given test files with pytest under the workspace configuration.

    Args:
        name: target name.
        srcs: test modules (and conftest/helper modules they import).
        deps: py_library targets the tests import.
        data: extra runtime files.
        args: extra pytest arguments, e.g. ["-m", "not slow_test"].
        env: environment for the test action; CPU lanes pass CUDA_VISIBLE_DEVICES="".
        **kwargs: forwarded to py_test (size, timeout, tags, ...).
    """
    test_files = [s for s in srcs if s.endswith(".py") and "/test_" in ("/" + s)]
    py_test(
        name = name,
        srcs = srcs + ["//tools/pytest:main.py"],
        main = "//tools/pytest:main.py",
        deps = deps + ["//tools/pytest:main", "//tests:support"],
        data = data + ["//:pyproject.toml"],
        args = args + ["$(location %s)" % s for s in test_files],
        env = env,
        **kwargs
    )
