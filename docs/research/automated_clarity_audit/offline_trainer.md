# Offline DPO trainer

Reviewed complete offline/dpo.py and trainer root/offline/online initializers,
Wan construction and forward adapter, existing timestep/optimizer/config tests.
Previous audit commit: f12906bb4.

Changes: remove long() after integer torch.randint output, convert gradient norm
directly to its float metric, and correct two descriptions: the forward adapter
takes four arguments, and step consumes a microbatch rather than unconditionally
performing an optimizer update. No training math or scheduler behavior changes.

Retain and why:

- OfflineDPOTrainerConfig.from_root owns the public-to-runtime projection and
  effective-batch LR scaling. The local required accessor shares actor-path
  diagnostics; another configuration wrapper would add indirection.
- The trainer owns optimizer and accumulation state. DPOStepMetrics is the actual
  output record, not an unnecessary validation object. Standard AdamW remains the
  default; Adafactor is an explicit configured alternative.
- Caller-supplied pixel/text encoders and forward adapter keep family operations
  outside this generic loop. Encoder batch-size checks preserve winner/loser and
  caption alignment at that callback boundary, where annotations cannot prove it.
- _sample_timestep_indices samples positions in the configured schedule; the
  resulting positions index both timesteps and fallback sigmas. Keep those two
  concepts distinct. Real DDPMScheduler, flow Euler and flow UniPC tests exercise
  the supported noise injection conventions, including the scale_noise versus
  sigma-table adapter. Do not remove a real scheduler compatibility path as guesswork.
- _reference_forward selects a separate frozen model or an adapter-off policy.
  Its caller owns no_grad/autocast. This is a synchronous offline forward contract,
  distinct from ReplayModel token/denoise evaluators, so a universal reference
  wrapper is unnecessary.
- Accumulation counter reset on restore is intentional because gradients are not
  checkpointed. It cannot be inferred from global_step, which counts consumed
  batches rather than optimizer updates. Non-strict optimizer restore explicitly
  permits skipping incompatible state; do not silently change that public mode.
- Offline initializer preserves the recipe's public imports. Online's
  _PUBLIC_EXPORTS is a lazy API table keeping config imports separate from the
  heavy trainer. Root exports nothing. These facades and names are real import
  boundaries, not ALL_CAPS workflow data to move elsewhere.

Non-goals: new validation matrices, classes or examples, changing default optimizer,
reference mode, sampling/noise pairing, accumulation/checkpoint format or loader
precision projection. Preserve meaningful family/framework adapter shapes.

55 existing offline timestep, DPO config and Wan adapter tests passed on CPU with
dependency warnings. They cover actual scheduler noise conventions, pair ordering,
loss/gradient metrics, optimizer settings and restore counters. Ruff checks pass.
No tests were added. This does not establish real Wan GPU throughput or a complete
pretrained training/checkpoint round trip.
