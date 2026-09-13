"""Adapt the existing uv lock to rules_python without resolving dependencies."""

def _uv_exports_impl(ctx):
    ctx.symlink(ctx.attr.project, "pyproject.toml")
    ctx.symlink(ctx.attr.lock, "uv.lock")
    for name, options in ctx.attr.profiles.items():
        result = ctx.execute(
            [
                ctx.path(ctx.attr.uv),
                "export",
                "--frozen",
                "--no-emit-project",
                "--no-default-groups",
                "--format", "requirements-txt",
                "--output-file", name + ".txt",
            ] + options,
            environment = {"UV_PYTHON_DOWNLOADS": "never", "UV_NO_PROGRESS": "1"},
        )
        if result.return_code:
            fail("uv lock export failed for {}:\n{}".format(name, result.stderr))
    # uv drops cross-package extras from requirement lines; copy them back from
    # the lock so rules_python wires optional dependencies (torch's NVIDIA libs).
    result = ctx.execute(
        [ctx.path(ctx.attr.interpreter), ctx.path(ctx.attr.restore_extras), "uv.lock"] +
        [name + ".txt" for name in ctx.attr.profiles],
    )
    if result.return_code:
        fail("restoring exported extras failed:\n{}".format(result.stderr))
    ctx.file("BUILD.bazel", "exports_files(glob([\"*.txt\"]))\n")

uv_exports = repository_rule(
    implementation = _uv_exports_impl,
    attrs = {
        "project": attr.label(mandatory = True, allow_single_file = True),
        "lock": attr.label(mandatory = True, allow_single_file = True),
        "uv": attr.label(mandatory = True, allow_single_file = True),
        "interpreter": attr.label(mandatory = True, allow_single_file = True),
        "restore_extras": attr.label(
            default = "//tools/dependencies:restore_export_extras.py",
            allow_single_file = True,
        ),
        "profiles": attr.string_list_dict(mandatory = True),
    },
)
