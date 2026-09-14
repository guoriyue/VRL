# Full-geometry Wan physics update on the locked runtime

## Scope and prerequisite evidence

The original Wan 2.1 I2V physics update remains open. Its interrupted run at
`/mnt/nvme/outputs/wan_i2v_full_physics_cpu_default_l40s` produced six videos
and both real rewards, but no optimizer checkpoint or saved rollout tensors.
There is no checkpoint from which to resume that update.

On 2026-09-13, the rebased runtime at `75b546c7` passed these prerequisites:

- CPU migration/data audit: all 309 training references present; twelve retained
  480x832/81-frame/8fps artifacts represent six unique samples, and the paired
  Kling/VideoCon artifact hashes match exactly.
- Real Wan I2V 14B weights, three FSDP CPU-offloaded ranks, BF16 actor policy,
  rank-32 LoRA and `full_cpu` checkpointing passed one full-shape forward and
  backward. Synthetic conditioning and squared-output loss were used, not GRPO.
  All ranks have finite gradients and 400 nonzero gradient tensors. Slowest
  forward is 41.495392s, backward 74.647215s; peak allocated 24,367,493,632B and
  reserved 31,681,675,264B on each rank. Torchrun session 95468 exited 0.
- Actual HTTP/MultiReward scoring with both real reward models on GPU 3 passed
  twice on a preserved video with its original prompt and verified SHA256.
  The decoded input is 81x480x832x3. Both iterations give Kling motion quality
  -0.635356328289045, VideoCon physical commonsense 0.408203125, and exact
  `0.3 * Kling + 0.7 * VideoCon` score 0.09513528901328647. Cold scoring including
  lazy model load took 68.330435s; warm scoring took 4.095011s. Client CUDA was
  never initialized. Session 39811 and both services exited 0, GPUs released.

These are not full-update, learning-quality, cross-version reward-parity or
combined policy/reward host-capacity claims. In particular, one backward does
not prove all 38 replay backward calls per rank fit with retained rollout state.

## Vendor setup issue found and resolved

The initial HTTP probe exited 1 on the first VideoCon score with
`ModuleNotFoundError: No module named 'mplug_owl_video'`. `/ready` had passed:
the service admits requests before lazily loading its model. The failed report
and service logs are preserved in `rebased_physics_http_probe` under the cache
root below. Both services were stopped and reaped before retrying.

Main exposes the unpackaged vendor via `third_party/BUILD.bazel` imports;
ordinary venv Python does not inherit those imports. A clean, detached clone
of the exact gitlink `01d366c979e4a82875852adbf427ccc036e7373a` now lives at
`videophy-pinned-01d366c9` under the cache root. It was cloned from existing Git
objects, not copied from the original dirty working files. The experiment
explicitly sets `PYTHONPATH` to its `videocon/training/pipeline_video` directory.
No original submodule edits, model math edits, dependency downgrade, vendor
source patch or lockfile change was made. The successful retry uses the same
locked Torch 2.11.0+cu130 / Transformers 5.13.0 environment as the other rebased
experiments. Historical Torch 2.12 scores are not a controlled comparison.

## Original workload retained

Migration replaces old prompt-microbatch keys with `prompts_per_collection=1`
and `training_microbatch_size=1`. Each of three ranks still collects one prompt
with two samples; six global samples, 20 generation steps, 19 replay timesteps
per sample, 480x832 and 81 frames. The pinned Wan I2V revision, real reference
manifest, seed 7, original reward weights, strict scheduling, `full_cpu`, BF16
actor FSDP policy, 0.95 host budget and 0.01 replay gate are unchanged.

`launch_wan_full_physics_rebased.py` prepares new service ports and artifact
roots, validates the migrated run, checks the pinned clean vendor, requires an
empty GPU inventory, performs real reward prewarm, then launches exactly one
native update. It records processes, received signals, memory samples and
terminal codes, and rejects Ray memory-pressure reports. The three-hour bound
is an abort deadline, not permission to reduce workload or declare success.
`--prepare-only` passed in session 64230; this did not start a training process.

The verifier uses current rank verdicts and checkpoint/RNG APIs, retains the
0.01 replay gate, and checks six global samples, real reward receipts, finite
nonzero optimizer moments, and paired retained video hashes. It requires a
healthy supervisor verdict as well as exit 0. Actual successful full-run
verification remains open. Negative validation against the historical failed
run exited 1 at its nonzero training return code (session 40100), as required.
Scoped Ruff check/format passed for the three new/migrated external scripts;
repository `git diff --check` passed. No native training process is active.

## Native failure: retained scheduler history

This section supersedes the prerequisite handoff's inactive-process statement.
The first native attempt at `wan_i2v_full_physics_rebased` exited 1 before
policy weight loading: Diffusers 0.40 queried Hub shard metadata despite the
offline environment. Reward prewarm had succeeded. Supervisor 909502/session
40899 and torchrun 910006 are terminal; wall time 63.219825s, no update or
checkpoint. Both services exited 0 during cleanup. The observed GPU release
was terminal service shutdown, not normal scoring-time parking.

The launcher now explicitly sets the existing `model.local_files_only=true`
option. This is passed through `ModelBuild.pretrained_kwargs` to both the
component and pipeline loaders and preserves the pinned Hub identity/revision.
No production source or model math was edited.

The attempt at `/mnt/nvme/outputs/wan_i2v_full_physics_rebased_local` is terminal:
supervisor 910531/session 21086 and torchrun 911014 exited 1 after 1705.464352s.
All six full-size samples generated and received both real rewards, but all
three workers failed physical parking before replay/backward. There is no
optimizer update or checkpoint. Both reward services exited 0; GPUs released.
`launch_fidelity_audit.json` establishes equality with the original workload
after the documented config/path migration. Native source was d2d01db8.

`failure_artifact_audit.json` verifies twelve MP4s, each 81 frames at
480x832/8fps, six unique sample IDs, six finite receipts per reward and exact
paired artifact hashes. The metrics file has only its header and no checkpoint
exists. Minimum sampled host availability was 177,338,544,128 bytes; no Ray
memory-pressure kill was recorded. This was not a host-memory failure.

All workers had a 426 MiB physical baseline and the unchanged 256 MiB allowance.
Their parked physical usage was 1000/988/1592 MiB, while Torch reserved
424/412/1016 MiB. Only approximately 25 MiB of live tensors pinned those large
allocator segments. Thresholds are not relaxed to admit this failure.

A real 14B, full-geometry, one-step native probe reproduced it at
`wan_i2v_parking_diagnostic_native` under the cache root: native parking retained
979,369,984 physical bytes with 413,138,944 reserved. GC ownership identified
UniPC `last_sample` and prediction history, each holding an 8,386,560-byte
latent tensor. Moving module buffers alone did not release the segments.
Replacing only `pipeline.scheduler` in the following diagnostic also failed,
because that diagnostic retained the original scheduler via `RuntimeBundle`.
Neither diagnostic is a full generation or training acceptance run.

The candidate fix gives both Wan T2V and I2V sampling states a fresh scheduler
via its public `from_config` API. Per-batch solver history no longer belongs to
the resident pipeline or runtime bundle. It changes ownership, not the scheduler
configuration, parking allowance, generation geometry or training objective.
Focused CPU tests compare two complete UniPC trajectories against the old
shared-scheduler behavior exactly and verify history is collectible afterward.
The fixed real full-shape probe at `wan_i2v_parking_batch_local_scheduler`
exited 0 (session 49430). Native parking, before any diagnostic intervention,
uses 568,328,192 physical bytes (542 MiB), only 116 MiB over its 426 MiB baseline,
with 2 MiB reserved. The unchanged 256 MiB allowance passes. No scheduler
prediction/last-sample tensors remain. The probe itself took 133.920868s.

`output_parity.json`, produced by `compare_wan_scheduler_probe.py`, verifies
exact equality of ten tensors: initialized observation/action/log-prob/timestep/
KL slots, full decoded video and all four conditioning tensors. Context also
matches. The initial whole-buffer comparison failed because the one-step probe
allocates twenty slots with `torch.empty` but only writes slot zero. The corrected
audit explicitly excludes unwritten slots 1:20; it does not hide a sample mismatch.
This proves only one native step at full geometry, not full-update acceptance.

After `uv sync --frozen --group test --group lint --extra cosmos --extra reward`,
the expanded regression passes: 141 tests, two GPU deselections. This includes
Wan family tests, worker sleep, real tiny pipeline wiring and replay runtime
wiring. Scoped Ruff and diff checks pass. The sync reinstalled locked decord
0.6.0; its dry-run still proposes reinstall because the published py3 wheel's
internal WHEEL tag says cp36-cp36m. Verbose uv identifies that metadata mismatch,
not a version change. No dependency/source/lockfile edits were made to mask it.
All diagnostic GPU processes are terminal and the GPU inventory is empty.
The next required gate is the original six-sample, full-geometry native update,
followed by checkpoint/replay/resume acceptance; that gate remains open.

## Artifacts and next command

Cache root: `/mnt/nvme/outputs/wan22_i2v_cache`.

- `wan_full_physics_rebased_preflight.json`, migrated candidate YAML.
- `wan_full_shape_rebased_cpu/rank-{0,1,2}.json` and torchrun logs.
- `rebased_physics_http_probe_vendor/result.json`, executed probe, service logs.
- `wan_full_physics_rebased_launch_preflight/`: prepared config/environment.
- `launch_wan_full_physics_rebased.py`, `verify_wan_full_physics_rebased.py`.

From `/home/ubuntu/VRL-review-all`, after rechecking hardware ownership:

```bash
env -u PYTHONPATH /mnt/nvme/venvs/vrl-review-all/bin/python \
  /mnt/nvme/outputs/wan22_i2v_cache/launch_wan_full_physics_rebased.py \
  --output-dir /mnt/nvme/outputs/wan_i2v_full_physics_batch_local_scheduler
```

Preserve any existing output instead of overwriting or automatically restarting
it. All separate Cosmos/H3/full-workload requirements remain open in the master
execution ledger; this document does not narrow the overall hardware goal.
