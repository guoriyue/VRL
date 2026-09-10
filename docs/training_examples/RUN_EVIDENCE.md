# Training launch evidence

The online runner writes `run_evidence/<launch_id>.json` in its output directory
before entering the training loop. Each resume produces another immutable file;
it does not replace the original launch record. Rank 0 publishes the file and
shares output-preparation failures with the other training ranks.

The record contains:

- The resolved configuration and its canonical JSON SHA-256.
- The model identity already used by strict checkpoint restore.
- Content hashes of configured training/evaluation manifests and source reports.
- Whether callers supplied examples directly instead of loading the configured data.
- The VRL Git commit, dirty status, and tracked-diff hash; unavailable Git evidence
  is explicit. A dirty diff hash alone does not reconstruct modified source.
- Python/platform, installed package versions, the PyTorch build, CUDA/cuDNN,
  locally observed NVIDIA driver versions, and visible GPU properties.
- Determinism/TF32 settings and an allowlist of runtime environment keys.

`captured_at` marks this snapshot, after model/runtime construction; it is not
process start time or a measurement of model-loading latency.

This is launch provenance, not a verification grade. It does not establish a
successful exit, a full learning curve, or bitwise reproducibility. Consult the
run verdict, metrics, checkpoints and separately identified evaluation outputs
for those claims. Existing runs without a launch record cannot acquire historical
software/hardware evidence by recording today's environment after the fact.

Scope limits are deliberate: visible devices describe the trainer process, not
remote rollout or reward workers. Manifest hashes do not hash every referenced
image/video. Supplied in-memory examples are identified as an override but their
contents are not captured here. Installed version strings do not fully identify
uncommitted edits in external editable dependencies. These gaps must remain
visible when assessing a recipe's reproducibility.

Sampler restart and numerical restart are also different: restoring the prompt
RNG preserves the next prompt draw, but an interrupted asynchronous run may lose
old-policy preview trajectories and regenerate them with restored current weights.
