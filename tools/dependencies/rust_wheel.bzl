"""Build one Python wheel from an sdist whose extension is written in Rust.

For a locked distribution that publishes no wheel for the managed CPython
(tokenizers 0.13.3 has none for 3.12), this fetches the sdist and a pinned
standalone Rust toolchain, both checksummed, and runs the standard
`pip wheel` build with the managed interpreter. The wheel and its sha256 are
the repository's outputs, so a requirements export can reference them by
`file://` URL with a hash, exactly like any other locked artifact.
"""

def _rust_wheel_impl(ctx):
    ctx.download(url = ctx.attr.sdist_urls, sha256 = ctx.attr.sdist_sha256, output = ctx.attr.sdist_filename)
    ctx.download_and_extract(
        url = ctx.attr.rust_urls,
        sha256 = ctx.attr.rust_sha256,
        stripPrefix = ctx.attr.rust_strip_prefix,
        output = "rust-dist",
    )
    result = ctx.execute(
        ["rust-dist/install.sh", "--prefix=" + str(ctx.path("rust")), "--disable-ldconfig"],
    )
    if result.return_code:
        fail("installing the Rust toolchain failed:\n" + result.stderr)
    ctx.execute(["rm", "-rf", "rust-dist"])
    python = ctx.path(ctx.attr.interpreter)
    # Repository rules inherit the invoking shell's environment; a developer's
    # conda/venv compiler flags must not steer this build.
    environment = {
        "PATH": str(ctx.path("rust/bin")) + ":/usr/bin:/bin",
        "HOME": str(ctx.path("home")),
        "CARGO_HOME": str(ctx.path("cargo-home")),
        "RUSTFLAGS": ctx.attr.rustflags,
        "CFLAGS": "",
        "CPPFLAGS": "",
        "CXXFLAGS": "",
        "LDFLAGS": "",
        "CC": "cc",
        "CXX": "c++",
        "PKG_CONFIG_PATH": "",
        "PYTHONPATH": "",
        "VIRTUAL_ENV": "",
        "CONDA_PREFIX": "",
    }
    result = ctx.execute(
        [python, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", "-w", "wheel", ctx.attr.sdist_filename],
        environment = environment,
        timeout = 3600,
    )
    if result.return_code:
        fail("building {} from source failed:\n{}".format(ctx.attr.sdist_filename, result.stderr[-4000:]))
    result = ctx.execute(["sh", "-c", "sha256sum wheel/*.whl"])
    if result.return_code:
        fail("hashing the built wheel failed:\n" + result.stderr)
    digest, _, path = result.stdout.strip().partition("  ")
    filename = path.rpartition("/")[2]
    ctx.file("wheel.sha256", digest + "\n")
    ctx.file("wheel.name", filename + "\n")
    ctx.file("BUILD.bazel", 'exports_files(glob(["wheel/*.whl"]) + ["wheel.sha256", "wheel.name"])\n')

rust_wheel = repository_rule(
    implementation = _rust_wheel_impl,
    attrs = {
        "sdist_urls": attr.string_list(mandatory = True),
        "sdist_sha256": attr.string(mandatory = True),
        "sdist_filename": attr.string(mandatory = True),
        "rust_urls": attr.string_list(mandatory = True),
        "rust_sha256": attr.string(mandatory = True),
        "rust_strip_prefix": attr.string(mandatory = True),
        "rustflags": attr.string(default = ""),
        "interpreter": attr.label(mandatory = True, allow_single_file = True),
    },
)
