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

After the online loop saves `checkpoint-final`, rank 0 also publishes
`run_evidence/<launch_id>.artifacts.json`. It binds the launch JSON, `metrics.csv`,
and the entire final checkpoint directory by content. Checkpoint hashing reuses
`local_checkpoint_content`: its SHA-256 includes file/tree structure and names,
not just file bytes, and it detects mutation during each read. This streams bytes
from disk and never deserializes checkpoint tensors. It adds one full checkpoint
read at run end; account for that IO when timing large-model jobs.

Verify an archived output directory with:

```python
from vrl.trainers.evidence import verify_run_artifacts

record = verify_run_artifacts(
    "outputs/my-run/run_evidence/<launch_id>.artifacts.json"
)
```

Verification raises on missing, changed, added, or removed checkpoint files,
changed metrics or launch contents, config digest mismatch, and incomplete or
redirected artifact references. Paths are relative so the whole output directory
can be archived elsewhere. The receipt is published without overwriting an
existing receipt for that launch. Archive the output directory **before resuming**:
resume appends metrics and replaces `checkpoint-final`, so the prior receipt will
correctly fail against those newer files. The receipt does not retain old bytes.

The phase is explicitly `after-training-loop-before-cleanup`. A later shutdown
failure can still fail the run. The receipt neither binds the supervisor's final
verdict nor includes held-out evaluations or intermediate checkpoints. Its hashes
establish internal consistency, not authenticity: retain a trusted external digest
or immutable archive if the receipt itself must be protected against replacement.
No learning-curve or deterministic-regression grade is inferred from these files.
