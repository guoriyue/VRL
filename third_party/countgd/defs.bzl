"""Assemble the qualified CountGD runtime tree as one directory output."""

def _countgd_runtime_impl(ctx):
    output = ctx.actions.declare_directory(ctx.attr.name)
    source_files = ctx.files.source
    first = source_files[0]
    source_root = "/".join([part for part in [first.root.path, first.owner.workspace_root] if part])
    args = ctx.actions.args()
    args.add("--source-root", source_root)
    args.add("--output", output.path)
    asset_files = []
    for target, relative in ctx.attr.assets.items():
        file = target.files.to_list()[0]
        asset_files.append(file)
        args.add("--asset", relative + "=" + file.path)
    ctx.actions.run(
        executable = ctx.executable._assembler,
        arguments = [args],
        inputs = source_files + asset_files,
        outputs = [output],
        mnemonic = "CountGDRuntime",
        progress_message = "Assembling the qualified CountGD runtime tree",
    )
    return [DefaultInfo(files = depset([output]), runfiles = ctx.runfiles(files = [output]))]

countgd_runtime = rule(
    implementation = _countgd_runtime_impl,
    attrs = {
        "source": attr.label(mandatory = True, doc = "filegroup of the patched upstream source"),
        "assets": attr.label_keyed_string_dict(
            mandatory = True,
            allow_files = True,
            doc = "Space asset file -> path inside the source tree",
        ),
        "_assembler": attr.label(
            default = "//tools/countgd:assemble_runtime",
            executable = True,
            cfg = "exec",
        ),
    },
)
