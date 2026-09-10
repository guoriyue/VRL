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

For supervised runs, verify process completion separately:

```python
from vrl.trainers.evidence import verify_run_completion

verdict = verify_run_completion(
    "outputs/my-run/run_evidence/<launch_id>.artifacts.json",
    "outputs/my-run/run_verdict.json",
)
```

The supervisor generates a fresh `VRL_RUN_ATTEMPT_ID` for each child launch,
including every retry. It passes the ID through the child environment, without
changing its own environment. Torchrun workers inherit the same ID. The online
launch record and each worker verdict record it as `attempt_id`; the supervisor
rejects missing or mismatched attempt IDs when collecting a live attempt. Custom
supervised commands should call `write_run_verdict`, or include the inherited ID
in their existing verdict writer. Old untagged custom verdicts are no longer
accepted as evidence for a live supervised attempt.

After joining the child, the supervisor records `supervisor_exit_code` in the
outcome. Completion verification requires zero, as well as a successful verdict
from the matching attempt. In distributed runs it additionally requires every
rank exactly once, with matching world size and attempt ID, and successful rank
verdicts. A worker's success JSON alone cannot certify a torchrun process that
later exited nonzero. Missing verdicts and cleanup failures do not become success.

Standalone launches and historical artifacts without a shared attempt identity
and observed supervisor exit remain eligible for content integrity checks, but
cannot pass this completion check. This check reads the final verdict separately;
it does not add its bytes to the earlier artifact receipt. Archive both together
and retain an external trusted digest when authenticity is required. Held-out
evaluation association and numerical reproducibility checks remain separate work.

Completed native image checkpoint evaluations can now be associated with training:

```bash
python -m vrl.scripts.eval.image_checkpoint_eval \
  --run-dir outputs/my-run \
  --output-dir outputs/my-run/checkpoint_evaluation \
  --verify-training-evidence outputs/my-run/run_evidence/LAUNCH_ID.artifacts.json
```

Add the same evaluation policy, manifest, sampling, seed and device options used
to produce the evaluation. This mode resolves the expected plan and
verifies the existing archive; it does not generate images or call rewards.
It rejects changed protocols, missing or altered scores, changed original PNGs,
model identity mismatches, and reports that do not evaluate the final checkpoint's
actual `checkpoint.pt` bytes. Renamed/copied checkpoints can match by content.
The native image evaluator restores this payload, including for LoRA training;
its exported adapter directory is not the evaluation source of truth.

The JSON result identifies the launch/attempt, matching checkpoint labels, a
canonical evaluation-protocol hash, and the complete evaluation tree identity.
Retain this association with the archived run if needed. It does not modify the
prior training receipt or award a verification grade. Resolving a different
runtime identity from the one recorded during generation fails protocol matching;
recording today's environment cannot repair missing historical evidence.

Programmatic callers can pass their independently specified `EvaluationArchive`
to `verify_training_evaluation(receipt, verdict, archive)`. Do not derive the
expected protocol from an untrusted report merely to make it match. This path
currently covers the native full-sequence denoise image evaluator. Video/token
benchmarks need their own existing protocol adapters. A matching evaluation does
not itself establish held-out data independence, human quality, a repeated
learning curve, or numerical determinism.

Online runs also write `metrics.full_precision.csv` from the same `OnlineMetricRow`
as the display CSV. Finite floating-point aggregates use Python float `repr`, which
round-trips binary64 values and signed zero; integer columns retain integer syntax.
The display CSV keeps its existing rounding and column order. Full-precision output
preserves logged aggregates, not original per-sample tensor bits or NaN payloads.

Both files use the existing schema and checkpoint-position alignment on resume.
Resuming an older run with no full-precision file starts that file at the resumed
position; it cannot reconstruct earlier precision from the rounded CSV. Missing
reward components remain NaN and must not pass a finite numerical regression.

Artifact receipts include `full_precision_metrics` when the new file is present,
so subsequent edits or deletion fail verification. Historical receipts without
that role remain valid integrity records, but are insufficient for full-precision
regression. Exact metric matching still needs an explicit metric/step protocol,
compatible run identities, and successful independent attempts; neither CSV alone
is a determinism certificate.

For same-revision repeatability, compare two independent supervised attempts with
an explicitly chosen metric set and complete epoch count:

```bash
python -m vrl.scripts.eval.compare_training_metrics \
  --reference-receipt archive/reference/run_evidence/REFERENCE.artifacts.json \
  --reference-verdict archive/reference/run_verdict.json \
  --candidate-receipt archive/candidate/run_evidence/CANDIDATE.artifacts.json \
  --candidate-verdict archive/candidate/run_verdict.json \
  --columns loss reward_mean grad_norm r_ocr \
  --expected-epochs 20 \
  --report archive/comparison-new-attempt.json
```

Choose columns and epochs before inspecting results. The result pins that protocol
and both receipt/verdict hashes. The only ignored config difference is
`trainer.output_dir`; seed and total epochs must be explicit and the requested
count must equal the configured count. Runs must be fresh, have different launch
and attempt IDs, stable configured data hashes, the same clean code identity,
model identity and trainer runtime, and strict deterministic algorithms enabled.
Warn-only determinism and cuDNN benchmark mode are rejected. GPU records also
require driver identity and deterministic cuDNN. These flags must be established
by the training setup; this comparison command does not turn them on retroactively.

The checker requires bound full-precision metrics, a matching column schema and
exactly epochs `0..N-1`. It compares the canonical serialized finite aggregates
without tolerance; NaN, infinity, missing/duplicate epochs and changed scalar text
fail. Signed zero is preserved. Rounded historical CSVs cannot pass this lane.
No report or baseline is silently replaced, and mismatch exits with the first
metric/epoch difference. Archive a fresh report for each independently specified
protocol; selecting fewer metrics changes what is proved.

Passing means the declared logged aggregates matched across those two completed
attempts. It does not compare complete trajectory/checkpoint tensors, establish
rollout/reward-process environment identity, validate held-out quality, or certify
an externally nondeterministic reward backend. Cross-revision baseline migration
and statistical comparisons need separate explicit protocols. These limitations
also apply when the values happen to match exactly.

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
