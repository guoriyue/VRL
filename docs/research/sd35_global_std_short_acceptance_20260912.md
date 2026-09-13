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

## Fourth arm: four rollout / four training ranks on the same GPUs

Completed from candidate `75d69be2`, with the same shared Python environment.
The same four physical GPUs alternate rank-local rollout and synchronous
adapter-only FSDP training. This is not eight GPUs, DDP, or simultaneous
rollout/training on each card. OCR runs on CPU, independently in each rank.
Frozen BF16 transformer weights are replicated; FP32 LoRA weights are sharded.
Each rank processes two groups with two accumulation microsteps, preserving
the global eight groups / 128 samples. The four prompt slices reproduce the
single-rank global draws for both updates. Other training settings are unchanged.

| Deployment | Second update | Speedup over single |
| --- | ---: | ---: |
| Single GPU strict | 628.586 s | 1.00x |
| Four GPUs, dedicated 3x1 strict | 519.956 s | 1.21x |
| Four GPUs, dedicated 3x1 continuous | 479.876 s | 1.31x |
| Four GPUs, phased 4x rollout / 4x FSDP training | **227.247 s** | **2.77x** |

The fourth arm's first update was 243.825 s. Its second update is measured
between the latest rank completion timestamps, including checkpoint-1 save,
weight delivery, CPU/GPU handoffs and trajectory spooling. Throughput is
0.5633 samples/s; speedup over dedicated strict is 2.2881x and over dedicated
continuous is 2.1117x. There is still only one post-warmup observation per arm.
This is an equal-global-work deployment comparison, not pure GPU-kernel scaling:
four CPU OCR instances and torchrun's default OMP_NUM_THREADS=1 are also topology
differences. Generated rollouts differ, and the historical user baseline remains
unverified. No learning improvement is inferred from training rewards.

Correctness and completion:

- All four rank verdicts are success; torchrun exited 0. Each rank collected
  32 samples and two groups per update. The balanced-survivor guard and rank-0
  trained-group count confirm eight trained groups globally in both updates.
- Global pre-update replay mismatch is exactly 0 in both updates. Gradient
  norms are 0.0016626533 and 0.0026402727, finite and nonzero.
- Final checkpoint is global_step=2, 383,105,789 bytes. Its 486 model tensors
  and 1,944 trainer-state tensors are finite FP32; 12 RNG tensors are uint8.
- Four-rank fixed-rollout CPU tests preserve gradient norms, updated adapters
  and Adam state within tolerance. GPU toy tests cover CPU-loaded frozen-weight
  placement and repeated parking/restoration of parameters, live gradients,
  optimizer and EMA tensors, followed by further training. These tests do not
  prove identical saved SD3.5 rollout gradients on hardware or an EMA refresh.
- Uneven surviving-group counts across ranks fail explicitly. Arbitrary uneven
  filtering is not supported, and these two updates did not trigger that guard.
- Initial broad regression: 560 passed, 2 skipped. Parking regression: 300
  passed. Final placement regression: 53 passed; four-GPU parking test passed.

Rank-0 second-update phases: replay 80.302 s, backward 50.737 s, collection
58.079 s, driver parking 23.679 s, rollout activation/offload 3.986/3.922 s,
and driver restore 1.156 s. These are rank-local durations, not additive fleet
maxima. Parking includes cross-rank coordination waits, not only transfer time;
OCR varied from 13.010 to 38.272 s across ranks in this update.

Two failed attempts are preserved: `colocated_fsdp.failed_parking` failed on
cross-device DTensor storage conversion; `colocated_fsdp.failed_frozen_placement`
failed because ignored frozen weights were left on CPU. Neither completed an
optimizer update or contributes a timing result. The fixes remain in the
candidate worktree; no shared runtime or dependency was changed under live jobs.

Evidence: `comparison_four_arm.json`, `summarize_four_arm.py`,
`launch_colocated_fsdp.sh`, `colocated_fsdp`, and `colocated_fsdp_torchrun` under
the controlled output root. Original three-arm artifacts remain unchanged.
The short comparison is complete, GPUs are released and the old queue remains
user-stopped. Further learning tests require a separate matched evaluation plan.

## Interpretation: when dedicated rollout/training pools help

Model size alone does not determine whether disaggregation wins. For a fixed
GPU budget, splitting trades fewer GPUs per stage for persistent residency,
independent stage parallelism, and (with asynchronous scheduling) overlap.
Strict scheduling does not gain that inter-update overlap just by separating
the pools. Asynchronous overlap also requires an explicitly accepted policy-lag
contract; throughput does not establish equivalent learning behavior.

This SD3.5 result points to a training bottleneck in the tested 3+1 allocation:
dedicated strict replay + backward took 391.860 s on one trainer GPU, versus
131.039 s on rank 0 of the four-trainer phased arm. Dedicated continuous hid
some collection, but it could not remove the one-GPU training bottleneck.
This does not prove that every dedicated split loses: 2+2 and dynamically
rebalanced pools were not measured.

Dedicated pools can be useful when generation and training service rates can
be balanced, rollout latency is large or variable, role-switching costs are
substantial, or the stages benefit from different parallelism/hardware. More
GPUs may make it easier to provision both stages adequately. Larger models can
increase switching costs, but can also make splitting a small fixed pool worse
by leaving too few GPUs or too little memory for either stage. They are neither
a necessary nor a sufficient condition for a dedicated-pool speedup.

For context, [verl's V1 asynchronous trainer documentation](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)
describes avoiding switch/offload costs with separate pools and lending idle
trainer resources to rollout when a fixed allocation is imbalanced. That is
external architectural guidance, not measured evidence for this SD3.5 runtime.

Working choice for this four-L40S recipe: retain phased four-rollout/four-trainer
execution as the fastest short-tested deployment. Keep the global workload,
precision, CPU reward budget, update semantics and timing boundaries explicit
in future comparisons. Do not infer a model-size threshold or start another
GPU experiment from this interpretation alone.
