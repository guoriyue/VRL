# SD3.5 corrected streaming: short hardware acceptance

Runtime: `e11c04bc` in `/home/ubuntu/VRL-mgpu-integration`. All three arms used the
unchanged shared environment (Torch 2.12.0+cu130, Transformers 4.57.6), fixed
seed 1234, 512px, 10 denoising steps, 8 prompts x 16 samples per update,
generation/replay batch 1, global_std=true and four-way accumulation. Model,
OCR, optimizer, KL and EMA configuration match. No shared runtime was merged.

## Results

All three arms exited 0 with success verdicts and final checkpoints at global_step=2.
Each update processed 128 samples, trained eight groups and had finite, nonzero gradients.

| Deployment | First update | Second update | Second-update samples/s |
| --- | ---: | ---: | ---: |
| Single GPU, colocated strict | 669.088 s | 628.586 s | 0.2036 |
| Four GPUs, dedicated 3x1 strict | 503.338 s | 519.956 s | 0.2462 |
| Four GPUs, dedicated 3x1 continuous | 512.493 s | 479.876 s | 0.2667 |

Observed post-warmup speedups:

- Single strict to four-GPU strict: **1.2089x**, or 17.28% lower wall time.
- Four-GPU strict to continuous: **1.0835x**, or 7.71% lower wall time.
- Single strict to four-GPU continuous: **1.3099x**, or 23.66% lower wall time.

The second
interval is measured between update-completion log timestamps, including the
same checkpoint-1 save in every arm. First-update timing starts at the training
loop log, not process launch; model initialization is excluded. These are only
two updates per arm, not a statistical performance or learning-quality result.

- Both strict deployments have exactly 0 replay mismatch for both updates. Continuous: 0 on the
  initial same-version update; 0.0281927 on update 2 with one-version-old
  prefetched samples. The latter is not a same-weight parity measurement and
  must not be presented as passing the strict 0.01 parity gate.
- Gradients: single 0.00181760 / 0.00241191; continuous 0.00160312 / 0.00492541.
  Four-GPU strict: 0.00163308 / 0.00198806.
  Generated samples differ, so these are not a gradient-equivalence comparison.
  Fixed-rollout equivalence remains covered by the separate CPU regression.
- Three independent real GPU receivers passed exact installation of the final
  continuous checkpoint's 486 trainable tensors (95,551,488 bytes each), after
  deliberately replacing their contents. Both installs required content checks
  and version acknowledgements. Fleet verification took 0.356 / 0.345 s.
  This probe covers in-place snapshot installation, not retained-slot activation.
- Continuous update 2: replay 236.310 s, backward 143.524 s, trajectory disk
  read/write 0.803 s. The training path dominates, not disk or weight transport.
- Four-GPU strict update 2: collection 125.876 s, replay 244.889 s, backward
  146.971 s, trajectory read/write 0.925 s and weight sync 0.237 s. OCR time
  varied from 50.069 to 70.055 s across its two updates; a single post-warmup
  observation cannot establish a stable 7.71% scheduling benefit.
- EMA refresh is configured every eight updates; this two-update run does not
  validate an EMA refresh. No multi-trainer DDP/FSDP equivalence claim is made.

## Evidence and boundaries

Artifacts: `/mnt/nvme/outputs/sd35_global_std_controlled/`. `comparison_three_arm.json`
contains configs' differences, metrics, phase statistics and the weight report;
the original two-arm `comparison.json` is preserved.
`summarize.py` reproduces the summary and checks completion, sample counts,
gradients and applicable parity gates. Per-arm logs, configs and checkpoints
are under `single`, `strict` and `continuous`. `weight_delivery.json` records receivers.

The added four-GPU strict arm separates the configured topology and scheduling
comparisons: between the four-GPU arms, only the schedule settings and output
directory differ. Runs use different generated samples and occur at different
times, so repeat measurements are needed to establish a stable benefit. This
does not reproduce the user's historical single-GPU run or compare identical
saved rollouts on hardware. Do not infer learning improvement from these
different generated samples or their reward means.

The user-authorized stop was enforced on the old queue, including an accidental
restart. The old script is guarded against further automatic launches. Original
strict artifacts and `continuous.interrupted_11ep` are preserved. All acceptance
processes exited and the GPU process inventory was empty after completion.
Hardware is released; no new long experiment is queued by this acceptance.
