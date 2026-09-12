# SD3.5 corrected streaming: short hardware acceptance

Runtime: `e11c04bc` in `/home/ubuntu/VRL-mgpu-integration`. Both arms used the
unchanged shared environment (Torch 2.12.0+cu130, Transformers 4.57.6), fixed
seed 1234, 512px, 10 denoising steps, 8 prompts x 16 samples per update,
generation/replay batch 1, global_std=true and four-way accumulation. Model,
OCR, optimizer, KL and EMA configuration match. No shared runtime was merged.

## Results

Both arms exited 0 with success verdicts and final checkpoints at global_step=2.
Each update processed 128 samples and had finite, nonzero gradients.

| Deployment | First update | Second update | Second-update samples/s |
| --- | ---: | ---: | ---: |
| Single GPU, colocated strict | 669.088 s | 628.586 s | 0.2036 |
| Four GPUs, dedicated 3x1 continuous | 512.493 s | 479.876 s | 0.2667 |

Observed post-warmup speedup: **1.3099x**, or 23.66% lower wall time. The second
interval is measured between update-completion log timestamps, including the
same checkpoint-1 save in both arms. First-update timing starts at the training
loop log, not process launch; model initialization is excluded. These are only
two updates per arm, not a statistical performance or learning-quality result.

- Single-GPU replay mismatch: exactly 0 for both updates. Continuous: 0 on the
  initial same-version update; 0.0281927 on update 2 with one-version-old
  prefetched samples. The latter is not a same-weight parity measurement and
  must not be presented as passing the strict 0.01 parity gate.
- Gradients: single 0.00181760 / 0.00241191; continuous 0.00160312 / 0.00492541.
  Generated samples differ, so these are not a gradient-equivalence comparison.
  Fixed-rollout equivalence remains covered by the separate CPU regression.
- Three independent real GPU receivers passed exact installation of the final
  continuous checkpoint's 486 trainable tensors (95,551,488 bytes each), after
  deliberately replacing their contents. Both installs required content checks
  and version acknowledgements. Fleet verification took 0.356 / 0.345 s.
  This probe covers in-place snapshot installation, not retained-slot activation.
- Continuous update 2: replay 236.310 s, backward 143.524 s, trajectory disk
  read/write 0.803 s. The training path dominates, not disk or weight transport.
- EMA refresh is configured every eight updates; this two-update run does not
  validate an EMA refresh. No multi-trainer DDP/FSDP equivalence claim is made.

## Evidence and boundaries

Artifacts: `/mnt/nvme/outputs/sd35_global_std_controlled/`. `comparison.json`
contains configs' differences, metrics, phase statistics and the weight report;
`summarize.py` reproduces the summary and checks completion, sample counts,
gradients and applicable parity gates. Per-arm logs, configs and checkpoints
are under `single` and `continuous`. `weight_delivery.json` records receivers.

The comparison changes topology and orchestration together. It does not isolate
continuous versus dedicated strict, reproduce the user's historical single-GPU
run, or compare identical saved rollouts on hardware. Do not infer learning
improvement from these different generated samples or their reward means.

The user-authorized stop was enforced on the old queue, including an accidental
restart. The old script is guarded against further automatic launches. Original
strict artifacts and `continuous.interrupted_11ep` are preserved. All acceptance
processes exited and the GPU process inventory was empty after completion.
Hardware is released; no new long experiment is queued by this acceptance.
