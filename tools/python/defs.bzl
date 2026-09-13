"""The VRL package built against one dependency hub.

pyproject.toml declares stacks that cannot share an environment (the model
runtime vs. vLLM vs. VBench). Each becomes a pypi hub in MODULE.bazel, and
`vrl_library` builds the same sources against one of them, so a target says
which stack it runs in instead of inheriting a developer venv.
"""

load("@rules_python//python:defs.bzl", "py_library")

VRL_SOURCES = ["vrl/**/*.py"]

VRL_DATA = [
    "vrl/**/*.yaml",
    "vrl/**/*.md",
    "vrl/**/*.json",
    "vrl/**/*.pth",
]

def vrl_library(name, requirements, **kwargs):
    """`py_library` of vrl/ with the given hub's requirement list as deps."""
    py_library(
        name = name,
        srcs = native.glob(VRL_SOURCES),
        data = native.glob(VRL_DATA, allow_empty = True),
        deps = requirements + [
            "//third_party:vendored",
            "//third_party/janus",
            # Interpreter-startup preload of the locked NVIDIA libraries.
            "//tools/python:nvidia_preload",
        ],
        **kwargs
    )
