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
run result, metrics, checkpoints and separately identified evaluation outputs
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

Online runs also write `metrics.full_precision.csv` from the same `OnlineMetricRow`
as the display CSV. Finite floating-point aggregates use Python float `repr`, which
round-trips binary64 values and signed zero; integer columns retain integer syntax.
The display CSV keeps its existing rounding and column order. Full-precision output
preserves logged aggregates, not original per-sample tensor bits or NaN payloads.

Both files use the existing schema and checkpoint-position alignment on resume.
Resuming an older run with no full-precision file starts that file at the resumed
position; it cannot reconstruct earlier precision from the rounded CSV. Missing
reward components remain NaN, indicating missing values.

Full-precision metrics remain available for analysis, but there is no built-in
exact cross-run metric comparison command. Neither CSV alone establishes training
determinism or learning quality.

For the online trainer, `trainer.seed` now seeds Python, NumPy's legacy global RNG,
and Torch before model construction. Every rank uses the same seed for model
initialization. After model/reward/runtime construction, trainer process RNGs are
reset to `(trainer.seed + rank) mod 2**64` (NumPy uses its low 32 bits), isolating
training randomness from construction and separating rank streams. The prompt
sampler keeps its existing explicitly seeded Generator. Resume restores saved
Python/NumPy/Torch/CUDA and prompt Generator state after this initialization.
This intentionally changes fresh-run behavior: previously the configured seed
controlled prompt selection but not global model initialization. Old unseeded
initializations cannot be reconstructed retroactively.

Enable strict trainer numerical settings with `trainer.deterministic=true` and
establish the cuBLAS workspace policy before process launch, for example:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17 \
  .venv/bin/python -m vrl.scripts.train --config experiment/sd3_5/online_grpo_ocr \
  trainer.seed=17 trainer.deterministic=true
```

This example enables settings; it is not a claim that the recipe has passed them.
The online run owner enables `torch.use_deterministic_algorithms(True, warn_only=False)`,
sets deterministic cuDNN and disables cuDNN benchmarking. It does not change the
precision policy or silently select another attention/compile backend when an
operation fails. The workspace defaults to `:4096:8` only before CUDA initialization;
a missing workspace policy after CUDA initialization or an incompatible configured
value is rejected. The supported workspace values follow the
[PyTorch deterministic-operation requirements](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html).
`PYTHONHASHSEED` must be supplied at process startup; this option cannot reset it
inside an already running interpreter.

The default `trainer.deterministic=false` leaves existing global numerical switches
unchanged and still applies the configured RNG seed. Resolution itself remains
side-effect-free; the online entrypoint explicitly applies the run-owned policy.
Offline DPO rejects the unsupported field through its existing consumption guard.
No model family registry, algorithm list, or second RNG checkpoint format is added.

This policy applies to the trainer process, including local code that uses its
global RNGs. It does not configure remote Ray actors, external reward services,
independently constructed NumPy Generator instances, or opaque backend RNGs.
The existing multi-rank rollout entropy broadcast remains unchanged. Actual
repeatability must therefore be tested across the complete selected recipe;
strict trainer flags alone are insufficient.
