"""Build one Python wheel from an sdist whose extension needs nvcc and torch.

For a locked distribution with no wheel for the managed CPython and the
pinned torch (flash-attn 2.4.2 against torch 2.4/cu124), this fetches the
sdist and the NVIDIA redistributable components that make up a CUDA toolkit
(nvcc, cudart, cccl; all checksummed), installs the stack's own hashed
requirements into a scratch environment so the build sees the exact torch
it will run with, and runs `pip wheel --no-build-isolation`. The host still
supplies the C/C++ compiler nvcc drives (`cc`/`c++` on PATH) and the NVIDIA
driver at run time; the wheel and its sha256 are the repository's outputs.
"""

def _cuda_wheel_impl(ctx):
    ctx.download(url = ctx.attr.sdist_urls, sha256 = ctx.attr.sdist_sha256, output = ctx.attr.sdist_filename)
    for component, spec in ctx.attr.cuda_components.items():
        url, _, sha256 = spec.partition(" ")
        ctx.download_and_extract(url = [url], sha256 = sha256, output = "cuda-parts/" + component, stripPrefix = url.rpartition("/")[2].removesuffix(".tar.xz"))
    # Merge the components into one CUDA_HOME layout (bin/, include/, lib/).
    result = ctx.execute(["sh", "-c", "mkdir -p cuda && for d in cuda-parts/*; do cp -a \"$d\"/. cuda/; done && rm -rf cuda-parts"])
    if result.return_code:
        fail("assembling the CUDA toolkit failed:\n" + result.stderr)
    python = ctx.path(ctx.attr.interpreter)
    cuda_home = str(ctx.path("cuda"))
    environment = {
        "PATH": cuda_home + "/bin:/usr/bin:/bin",
        "HOME": str(ctx.path("home")),
        "CUDA_HOME": cuda_home,
        "CFLAGS": "",
        "CPPFLAGS": "",
        "CXXFLAGS": "",
        "LDFLAGS": "",
        "CC": "cc",
        "CXX": "c++",
        "PYTHONPATH": "",
        "VIRTUAL_ENV": "",
        "CONDA_PREFIX": "",
        "MAX_JOBS": ctx.attr.max_jobs,
    }
    environment.update(ctx.attr.build_environment)
    steps = [
        [python, "-m", "venv", "env"],
        ["env/bin/python", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--require-hashes", "-r", str(ctx.path(ctx.attr.requirements))],
        ["env/bin/python", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir"] + ctx.attr.build_requirements,
        ["env/bin/python", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir", "-w", "wheel", ctx.attr.sdist_filename],
    ]
    for step in steps:
        result = ctx.execute(step, environment = environment, timeout = 7200)
        if result.return_code:
            # Bazel discards a failed repository, so surface the compiler's own
            # diagnostics here rather than only the tail of the log.
            diagnostics = [
                line
                for line in (result.stdout + "\n" + result.stderr).splitlines()
                if "error" in line or "Error" in line or "fatal" in line
            ]
            fail("{} failed for {}:\n{}\n---\n{}".format(
                " ".join([str(s) for s in step[:4]]),
                ctx.attr.sdist_filename,
                "\n".join(diagnostics[:40]),
                result.stderr[-3000:],
            ))
    ctx.execute(["rm", "-rf", "env", "home"])
    result = ctx.execute(["sh", "-c", "sha256sum wheel/*.whl"])
    if result.return_code:
        fail("hashing the built wheel failed:\n" + result.stderr)
    digest, _, path = result.stdout.strip().partition("  ")
    ctx.file("wheel.sha256", digest + "\n")
    ctx.file("wheel.name", path.rpartition("/")[2] + "\n")
    ctx.file("BUILD.bazel", 'exports_files(glob(["wheel/*.whl"]) + ["wheel.sha256", "wheel.name"])\n')

cuda_wheel = repository_rule(
    implementation = _cuda_wheel_impl,
    attrs = {
        "sdist_urls": attr.string_list(mandatory = True),
        "sdist_sha256": attr.string(mandatory = True),
        "sdist_filename": attr.string(mandatory = True),
        "cuda_components": attr.string_dict(
            mandatory = True,
            doc = "component name -> '<archive url> <sha256>' from NVIDIA's redistrib JSON",
        ),
        "requirements": attr.label(mandatory = True, allow_single_file = True, doc = "hashed lock of the stack the wheel builds against"),
        "build_requirements": attr.string_list(default = [], doc = "extra build-time packages (pinned)"),
        "build_environment": attr.string_dict(default = {}),
        "max_jobs": attr.string(default = "16"),
        "interpreter": attr.label(mandatory = True, allow_single_file = True),
    },
)
