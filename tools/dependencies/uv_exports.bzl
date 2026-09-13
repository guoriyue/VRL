"""Adapt the existing uv lock to rules_python without resolving dependencies."""

def _uv_exports_impl(ctx):
    # Re-export whenever the lock, the project file or the fix-up script changes;
    # symlinking alone does not register their contents as inputs.
    for label in (ctx.attr.project, ctx.attr.lock, ctx.attr.restore_extras, ctx.attr.replace_requirement):
        ctx.watch(label)
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
    # Distributions built from source by a `rust_wheel` repository replace
    # their exported line with the built wheel's file URL and hash -- only in
    # the profiles that pin the built version. A profile pinning another
    # version (the main stack's tokenizers for transformers 5 vs. VBench's
    # 0.13.3) keeps its PyPI line.
    for name, wheel_repo in ctx.attr.built_wheels.items():
        line, version = _built_wheel_line(ctx, name, wheel_repo)
        for profile in ctx.attr.profiles:
            result = ctx.execute([
                ctx.path(ctx.attr.interpreter),
                ctx.path(ctx.attr.replace_requirement),
                profile + ".txt",
                name,
                version,
                line,
            ])
            if result.return_code:
                fail("replacing {} in {}.txt failed:\n{}".format(name, profile, result.stderr))
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
        "replace_requirement": attr.label(
            default = "//tools/dependencies:replace_requirement.py",
            allow_single_file = True,
        ),
        "profiles": attr.string_list_dict(mandatory = True),
        "built_wheels": attr.string_dict(
            default = {},
            doc = "distribution name -> repository name of the rust_wheel that built it",
        ),
    },
)

def _built_wheel_line(ctx, name, wheel_repo):
    """The requirement line for a built wheel, and the version it carries.

    Wheel filenames are `{name}-{version}-{python}-{abi}-{platform}.whl`.
    """
    digest = ctx.read(Label(wheel_repo + "//:wheel.sha256")).strip()
    filename = ctx.read(Label(wheel_repo + "//:wheel.name")).strip()
    path = ctx.path(Label(wheel_repo + "//:wheel/" + filename))
    version = filename.split("-")[1]
    return "{} @ file://{} --hash=sha256:{}".format(name, path, digest), version

def _requirements_with_built_wheels_impl(ctx):
    """A hashed requirements file plus the wheels Bazel built from source."""
    ctx.watch(ctx.attr.requirements)
    text = ctx.read(ctx.attr.requirements)
    lines = [_built_wheel_line(ctx, name, repo)[0] for name, repo in ctx.attr.built_wheels.items()]
    ctx.file("requirements.txt", text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
    ctx.file("BUILD.bazel", 'exports_files(["requirements.txt"])\n')

requirements_with_built_wheels = repository_rule(
    implementation = _requirements_with_built_wheels_impl,
    attrs = {
        "requirements": attr.label(mandatory = True, allow_single_file = True),
        "built_wheels": attr.string_dict(mandatory = True),
        "interpreter": attr.label(allow_single_file = True),
    },
)
