# Flash-GRPO official implementation reproduction — 2026-09-20

Status: stopped on user instruction. External official execution is NOT the desired path. User will prepare native VRL integration on another machine. No completed optimizer update has been verified.

The user requested an established experiment, rather than a new Wan2.2/Kling
combination. Therefore the active attempt uses the authors' training loop and
HPSv3 implementation in an isolated environment. The latest VRL checkout is
preserved for a later port/performance comparison.

## Pinned sources and data

- Flash-GRPO: `Shredded-Pork/Flash-GRPO`,
  `bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d`.
- HPSv3 `upgrade_transformers_version`:
  `ad33a29d5896fe46ae74f9865444d6adbadbc750`.
- Missing `wan_prompt_embedding.py` restored from the acknowledged Flow-GRPO
  upstream, whose HEAD at retrieval was
  `879042cf5707f8b90daa98d147d7deac2317c5da`; downloaded file retained on NVMe.
- Wan2.1-T2V-1.3B-Diffusers:
  `0fad780a534b6463e45facd96134c9f345acfa5b`, original base, no warm-start adapter.
- Official 19,700 training / 300 test prompts, no relabeling or synthetic subset.
  Training SHA-256 `610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016`;
  test SHA-256 `96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5`.
  Existing VRL manifests match these byte for byte.

## Important correction to the earlier plan

The VRL 96-sample recipe is NOT a verified faithful port of the released loop.
Reading configuration alone missed the hard-coded `num_videos_per_prompt=2` and
`repeat_interleave(2)`. The released eight-rank loop uses 24 requests/rank/epoch,
two videos/request, and 12 requests per optimizer accumulation. Before zero-
advantage filtering this means 384 generated videos/epoch, 192 videos/update,
and nominally eight videos per repeated prompt, despite the config's G=4.
Rewards are normalized over the collected epoch before two optimizer updates.
The proposed four-rank port uses 48 requests/rank and accumulation 24, retaining
these effective counts and the original normalization/update order. Filtering
may change the realized count; event logs must distinguish actual and nominal.

The first VRL candidate was stopped during loading, before any optimizer update.
Its evidence remains at `runs/flash_grpo_official_20260920`. Do not restart its
automatic continuation. It is superseded by `runs/flash_grpo_upstream_20260920`.

## Preserved experimental choices

Official Flash-GRPO loss and time-dependent rectification; one stochastic step
chosen among the first ten; original prompt sampler and group statistics;
20-step training / 50-step deterministic evaluation, CFG 4.5; 480x832x81 frames;
LoRA rank 16, alpha 32, Gaussian initialization; AdamW lr 1e-4, weight decay
1e-4, betas .9/.999, eps 1e-8; clip 1e-3; no KL; original EMA implementation
(including its adaptive decay and update cadence); HPSv3 on JPEG-encoded frames,
mean of the best floor(81*.3)=24 frame scores. HPSv3 remains an image preference
reward and does not establish motion or physical-correctness improvement.

## Explicit adaptations and remaining limits

- Four L40S instead of the released eight-rank launch; DeepSpeed ZeRO-2 retained.
- Fixed the missing launcher target by invoking the existing training filename;
  supplied the missing upstream prompt-embedding helper and empty assets package.
- Repeated identical negative conditioning for the actual two-video replay batch;
  explicitly expanded supplied rollout conditioning to the same batch shape.
- GPU phases are serialized: deferred rewards retain CPU video tensors; frozen
  pipeline weights park on CPU while a loopback-only upstream HPSv3 server uses
  GPU3. Trainable ZeRO parameter aliases stay on GPU. Restore before replay.
- Reward inference uses eight-frame microbatches and SDPA. Batch/kernel/library
  differences can change floating-point scores; not claimed bitwise equivalent
  to the authors' unavailable runtime/hardware.
- VAE slicing/tiling enabled for L40S memory. No geometry reduction.
- Independent Python 3.11 / Torch 2.6 environment. The published dependency pins
  do not run the current released source: e.g. its Wan2.2-style pipeline interface
  needs a newer Diffusers than the pinned .33.1. Working versions and patches are
  archived; these are compatibility adaptations, not an unchanged environment.
  Trainer uses Transformers 4.51.3 / Diffusers 0.35.1 / PEFT 0.17.1; upstream
  HPSv3's upgraded model layout needs a separate Transformers 4.57.1 overlay.
  Its default image processor is fast; this is another recorded numerical
  difference from older HPSv3 environments. The initial 4.51 HPS smoke failure
  is preserved in `startup_attempt1`; no update occurred in that attempt.
- DeepSpeed optional-op compatibility probing patched to return unavailable when
  no CUDA toolkit exists. No custom CUDA ops are built; standard AdamW is used.
- Exact unchanged-policy parity is checked across ALL collected transitions
  before either optimizer update. It is not required after the first update,
  because the official second update deliberately reuses the old trajectories.
  Any failed check stops the run; old log probabilities are never rewritten.
- Initial evaluation uses the first 24 official test prompts with fixed seeds,
  preserving before/after MP4s. This is a subset evaluation, not the full paper
  benchmark. A full 300-prompt evaluation is still needed for a stronger claim.
- Bounded initial run: eight-hour target, finish at an epoch/checkpoint boundary;
  18-hour hard watchdog. Save each trained epoch, including optimizer/RNG state.
  Offline/disabled experiment tracking; logs stay on this machine.

## Artifacts and operation

NVMe root: `/mnt/nvme/vrl-night-20260918/runs/flash_grpo_upstream_20260920`.
`status.json`, `training.log`, `hps.log`, `events.rank*.jsonl`,
`upstream.patch`, `accelerate.yaml`, `hps.yaml`, and checkpoint directories.
Supervisor: `tools/overnight/run_official_flash.py`; preparation script:
`tools/overnight/prepare_official_flash.py`. Polling during sustained GPU work is
15 minutes. `STOP_AFTER_EPOCH` requests a checkpoint-boundary stop.

References: https://github.com/Shredded-Pork/Flash-GRPO,
https://github.com/MizzenAI/HPSv3/tree/upgrade_transformers_version,
https://github.com/yifan123/flow_grpo.
