# Primary-rank fill: four-L40S overnight acceptance

> The `tools/overnight/` scripts these notes cite are not on `main`; they are
> the night's operational scripts and live in commit `f38c5afc3` on
> `exp/overnight-primary-fill-20260918`. Results and receipts are under `outputs/`.

In progress. Candidate `3ab166bb`, isolated worktree
`/home/ubuntu/VRL-night-20260918`. Parent control: `8c7fec35` in
`/home/ubuntu/VRL-primary-fill-baseline`. Production code is unchanged.

Priority: SD3.5 adapter-only regression and matched parent control, Wan 1.3B
full-shard regression, then full-size random H3 trainer-only comparison. The
older CP, Wan2.2 residency, rollout SP, reward isolation, miles ablations and
Wan 14B long-run requests remain pending behind these gates.

## Hardware and storage

Four idle L40S, each 46,068 MiB; host 372 GiB RAM, no swap. Root disk initially
22 GiB free. Historical `/mnt/nvme/outputs` and model copies are absent after
instance-storage loss. Two 1.7 TB devices have no detected filesystem signatures;
no format has been performed. Rebuilt temporary model assets live under
`/dev/shm/vrl-night-20260918/models`, are volatile, and count against host RAM.
Do not conflate their footprint with trainer process memory.

HPSv3 and Qwen2-VL were re-downloaded. H3 full random base is being rebuilt
from Diffusers defaults (50 blocks, 33B), deterministic seed 731, original
computed buffers preserved, BF16 frozen matrices/native FP32 exceptions.
The original historical random checkpoint and executed probe are unavailable;
new partitioned and FSDP arms must use the same rebuilt checkpoint.

## Environments

SD3.5/Wan regression: original `/home/ubuntu/VRL/.venv`, Torch 2.12.0+cu130,
Transformers 4.57.6, Diffusers 0.38.0. Candidate and parent use the same environment.
H3: separate worktree `.venv`, created offline from the repository `uv.lock`:
Torch 2.11.0, Transformers 5.13.0, Diffusers 0.40.0, PEFT 0.19.1. Both H3 arms
use that same environment. Shared environment was not modified.
CPU primary-fill tests in the locked environment: 2 passed in 13.81 seconds.

## Initial SD3.5 evidence

Two updates requested, original workload/precision/seed. Update 0:

- Pre-update replay mismatch 0, pre-update clip fraction 0.
- Loss -4.190951585769653e-09.
- Reward mean 0.3506826162338257.
- Gradient norm 0.0018951534293591976, finite and nonzero.

Historical four-rank first gradient was 0.0016626533. This is not identical;
matched parent control is queued before claiming numerical regression acceptance.
Historical results used older code and cannot alone isolate the new loader.

Before/after bundle-build RSS, MiB, ranks 0..3:
`577.6 -> 1012.0`, `581.4 -> 1328.7`, `581.2 -> 1328.4`, `577.2 -> 1324.5`.
These observations do NOT show the expected zero non-primary delta: imports,
adapter construction, lazy/mmap checkpoint pages and later materialization all
matter. RSS at these two points is not a count of checkpoint copies. Compare
the parent and the full resource trace before stating a memory reduction.

## Execution and gates

`outputs/overnight_20260918/queue.json` is the appendable job manifest.
`tools/overnight/queue_runner.py` serializes GPU ownership using an exclusive
supervisor lock and checks the GPU inventory before each launch. It records
per-job logs, launch status, five-second GPU/host traces and explicit rank outcomes.
Runtime results must show completed updates, finite loss/reward, nonzero finite
gradients and the requested exact-zero pre-update replay difference. Failures
block dependent jobs; no automatic threshold relaxation or failure retry.
The worker has a 12-hour scheduling window and per-job timeouts.

Wan's unchanged compiled-rollout preset documents historical ~0.002 numeric
replay drift. Exact-zero acceptance is intentionally retained; any measured
nonzero result requires diagnosis and cannot be called passing merely because
the preset's looser built-in debug gate allowed it.

H3 probe records native 768x1344/124f replay on deterministic synthetic
conditioning; rank-32 FP32 LoRA, frozen BF16 base, gradient checkpointing,
one MSE backward and AdamW step. Reference and FSDP compare prediction and
all adapter gradients at predeclared atol=rtol=1e-3; maximum absolute differences
are reported. This is capacity/numerical coverage, not GRPO, real-weight quality,
rollout support, or long-training acceptance. Historical placement remains in
production until this and the requested follow-up gates pass.

SD3.5 candidate finished both updates; four successful rank receipts and final
checkpoint. Update 1: loss 3.2121328533523614e-05, reward 0.37429606914520264,
gradient norm 0.0023531040642410517, pre-update replay difference 0. Parent
control was automatically launched only after the candidate released all GPUs.
H3 gradient comparison additionally requires global relative L2 error <= 0.01,
so tiny gradients cannot pass solely due to the absolute tolerance.

## Matched controls and backward determinism

Parent first update has exactly matching loss/reward and zero replay drift,
but gradient norm 0.0018895446555688977 versus candidate 0.0018951534293591976.
All 243 LoRA A tensors match bitwise; Adam first-moment relative L2 difference
is 0.018158277288997635 (cosine 0.9998390415067732). The strict parent comparison
is retained as failed, not silently reclassified as passing.

A short real SD3.5 four-GPU probe then removed rollout/reward variation: same
fixed rank-specific inputs and name-seeded nonzero rank-32 A/B adapters, native
BF16 frozen weights and FP32 adapter-only FSDP. Ordinary mode: exact forward,
gradient relative L2 7.552893369787814e-05 (failed the predeclared 1e-5 gate).
With deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` on BOTH
arms, prediction difference and full adapter-gradient difference are exactly
zero. Both norms are 0.11441794357837136. All four rank receipts passed.
Artifacts: `sd35_fixed_*` and `sd35_deterministic_*` under the output root.

This establishes unchanged deterministic forward/backward for the loader on
this real model; ordinary full-update nondeterminism remains explicitly visible.
It does not retroactively prove bitwise equality of the historical GRPO updates.
H3 prerequisites use the two-update runtime gate plus this deterministic
zero-difference control, and still require the Wan gate.

## Wan environment preflight

HPSv3 tests in the historical environment exposed two failures: missing
`transformers.conversion_mapping` in Transformers 4.57.6. A process-local
`.wan_runtime` overlay selects locked Diffusers 0.40.0, Transformers 5.13.0,
Hugging Face Hub 1.23.0, PEFT 0.19.1 and related packages while retaining the
original Torch 2.12/CuMem runtime. HPSv3 tests then pass: 8 passed in 1.69s.
The shared environment is unchanged. Wan comparison must account for this
necessary environment change; historical timing/numerics are not matched controls.
The initial Wan loading attempt was terminated before updates to prioritize
the short SD3.5 gradient diagnosis; its artifacts are preserved separately.
A fresh one-update Wan run is queued (within the requested 1–2 updates).

H3 checkpoint construction completed in 691.60 seconds: 33,122,992,896 base
parameters. 34 CPU H3 loading/backbone tests passed in the isolated environment.

## Prepared downstream work (not executed/accepted yet)

Wan2.2 T2V A14B weights are staged under `/dev/shm/vrl-night-20260918/models/wan22_bf16`
from pinned revision `5be7df9619b54f4e2667b2755bc6a756675b5cd7`. Preparation streamed
the original FP32 shards into the requested BF16 runtime representation while
preserving the actual transformer's native FP32 module patterns. Per-shard source
SHA and byte accounting are retained; no quantized weights or global cache edits.
KlingVideoReward and its Qwen2-VL-2B base are also staged, pinned and locally
configured. Kling loading/scoring checks: 19 passed, 1 skipped. This is not a GPU
reward acceptance result.

Executable downstream jobs compare the original sequential-offload/CPU-offload
Wan2.2 arm against active-expert model offload and GPU-resident FSDP shards at the
same 320px/17f/32-sample geometry. Both full experts cannot reside unsharded on one
46GB GPU. The gradient comparison requires Adam first-moment relative L2 <=1e-4,
and each training arm retains parity=0. A separate 480x832/33f one-update smoke
must pass before the eight-epoch total resume run (seven additional updates).
The long run additionally depends on the outstanding CP/SP/reward/miles work;
it is not running and cannot jump those gates.

A bounded H3 `reshard_after_forward=false` experiment is queued after the primary
H3 gate, before temporary H3 weights are released for Wan2.2. It retains failure
and rank-local peak-memory receipts if the full unsharded base exhausts a card.
An OOM here does not invalidate the independent `reshard_after_forward=true` result.
HPSv3 assets remain available for the outstanding second reward service update.

SP acceptance now has an executable single-rank repeat control and production
Ray N=2 comparison at the unchanged pixel tolerance 0.02. Miles A/B/C each run
20 updates. The no-recompute arm explicitly retains the historical 0.05 diagnostic
drift allowance; completing an ablation is not strict replay parity acceptance.
Current production disallows in-process online rewards, so reproducing the old
five-arm table requires a separately scoped historical control. Cosmos colocated
DDP remains capability-gated. CP and the full reward-isolation gate remain pending;
manifest placeholders must not be mistaken for executable completed experiments.

All these models in `/dev/shm` are volatile. Training receipts are copied to the
persistent output tree on each job's exit; the proposed long run copies its last
two complete numbered checkpoints on exit. Root free space is approximately18GiB.
No raw NVMe device has been formatted or mounted.

Wan host loading evidence is already available: ranks 1/2/3 RSS changes are
789.5→896.2 / 789.5→896.3 / 788.1→894.8 MiB (about107MiB each); rank0 is
787.9→9052.6 MiB. This is consistent with skeleton construction on nonprimary
ranks, unlike the SD3 import/mmap-sensitive RSS observation above. Do not call
the actual deltas zero. A deterministic four-rank full Wan actor-precision
parent/candidate one-update control is queued after the online recipe; it
requires bitwise matching predictions, gradients and updated adapter tensors.
Its synthetic MSE objective does not replace the online replay-parity gate.

The H3 comparison includes both video and audio predictions; the synthetic
objective is the sum of their squared means. All adapter gradients are compared.
A host-headroom guard may delete ONLY the night's reproducible random H3 fixture
if available RAM drops below50GiB while the active Wan run parks GPU weights.
It retains the preparation receipt and rebuilds the same seed after Wan ends.
No user checkpoints or downloaded released weights are deleted by that guard.

Additional reviewed job JSONs can be added with `tools/overnight/enqueue.py`;
it locks manifest edits and inserts each job before the fallback long run,
which gains a dependency on that job. It refuses to interrupt an active long run.

The prospective long-run sampler differs deliberately from the historical
placement benchmark: `denoise_mode=sde`, `sde.type=flow_grpo`, full stochastic
window and strided 0.99 timestep fraction. Native deterministic actions scored
under an SDE density are not a valid on-policy GRPO training choice. Baseline and
resident placement arms retain their identical historical settings only for the
paired performance comparison. The 480p smoke tests the actual stochastic long-run
configuration from scratch; only its own checkpoint can resume into that run.

The guard triggered during Wan reward loading (available48,521,641,984bytes).
For the subsequent H3 rebuild, CPU timing showed 16million normal draws take
0.318s directly in BF16 versus0.088s in FP32 followed by BF16 cast. The rebuild
therefore uses generator `float32_normal_then_native_dtype_v2`, still seed731.
It is NOT byte-identical to the discarded v1 fixture; neither GPU comparison
had consumed v1. Both H3 reference and FSDP must use the same completed v2 fixture.
This reduces fixture rebuild time without changing architecture, native dtypes,
adapter setup, acceptance tolerances or any released model.

## Host-capacity failure and corrected staging order

At22:57–22:58, Ray killed all four parked rollout actors as host usage crossed
its unchanged95% limit (354.30GiB/372.73GiB). The guard's earlier62GiB H3 release
was insufficient: prematurely staging the downstream Wan2.2 weights in tmpfs still
consumed another64GiB. Each parked rollout/reward actor occupied about31GiB.
This was a preparation error in this session, not evidence of a primary-fill
regression. Raylet logs and exact process identities are preserved under
`wan13_host_oom`. No completed online update or clean acceptance is claimed.
The unrecoverable arm was stopped rather than continuing toward a weight-sync
failure against dead rollout actors. All its process descendants exited; no Xid.

The64GiB temporary Wan2.2 staging was removed, preserving source/config/hash
metadata. Four private rollout spool files were copied to a volatile diagnostic
fixture before their verified root-disk copies were removed. Root free space
returned to18GiB. `primary_fill_wan13_clean` repeats the same one-update recipe,
with output/spooling in tmpfs and without H3/14B prefetch. It follows the Wan
fixed-input controls. H3 generation now occurs only inside its own gated job;
Wan2.2 staging occurs only after H3 releases its fixture and HPSv3 assets.

The first Wan fixed-input probe failed before GPU execution because its ModelBuild
omitted `local_files_only=true`; current Diffusers attempted Hub metadata access
under offline mode. This is fixed in the probe, with a separate retained failed
job/log and fresh output directory for the rerun. Production code remains unchanged.

## Wan fixed-input result

The four-rank full-model Wan1.3B parent/candidate control passed on all ranks.
It exercises full sharding, actor parameter normalization, 480x832/33f latent
inputs, deterministic nonzero rank32 adapters, backward and one AdamW update.
Prediction max difference0, all adapter-gradient max difference0, all updated
adapter max difference0. Both global gradient norms0.05725382689534056.
Forward/backward/update/gather time about10.9seconds after loading; peak tensor
allocation4,193,806,336bytes per GPU. These synthetic inputs isolate loading;
they are not an online GRPO result and use plain AdamW, not the online optimizer
wrapper's full precision-master/EMA lifecycle.

`primary_fill_wan13_clean` is now running after those controls. Its H3 dependents
also require a separate runtime-health gate rejecting Ray host-OOM worker kills,
CUDA OOM and Xid evidence; this does not loosen numerical parity. All12hour queue
dependencies remain conditional. New sources are prepared on demand rather than
prefetching all models into RAM while a phase-cycled job is active.

Two final SD3.5 online controls are queued after the active Wan retry: parent and
candidate, one real GRPO update each with deterministic algorithms and the same
cuBLAS workspace. Unlike the already-passing synthetic control, these exercise
the real reward, objective, optimizer and weight-sync path. Candidate first-update
loss/reward/gradient norm must match the parent at unchanged rel1e-5/abs1e-8,
and both must retain zero replay mismatch. They address the unresolved ordinary
full-update gradient-norm difference rather than treating synthetic equivalence
as proof of complete GRPO equivalence. H3 depends on their successful completion.

## User-authorized NVMe setup and strict gate reaffirmation

On2026-09-18 at23:46PDT the user authorized initializing one local NVMe and
explicitly reaffirmed that the original Wan recipe must attain parity=0 before
H3 may run, regardless of exact fixed-input loader equivalence. The queue retains
that zero limit and H3's dependency on `primary_fill_wan13_clean`.

Initialized only `/dev/nvme1n1` (EC2 instance-storage serial
`AWS22215871D135A270A`) as ext4, label `vrl-experiments`, UUID
`b749750e-c0df-4bde-9bc3-a8215c2bd164`, mounted at `/mnt/nvme`.
The existing fstab device-name entry was replaced with this UUID; backup is
`/etc/fstab.before-vrl-20260918`. Root and `/dev/nvme2n1` were not modified.
Root-run fstab verification reports no warnings/errors, and an ubuntu-owned
1MiB write/fsync/read comparison passed. Approximately1.7TiB is available.

Inactive Kling/Qwen2-VL-2B assets and the private Wan spool fixture have been
copied to `/mnt/nvme/vrl-night-20260918`, verified with a checksum rsync dry-run,
and replaced by compatibility symlinks in their previous tmpfs locations.
The active Wan job retains its existing paths. A background migration waits for
its completion before moving HPSv3/base weights, and for the runtime-health audit
before moving its output. Every completed migration appends a receipt to
`nvme_migration.jsonl`. Future H3 and Wan2.2 prep/configs/queue paths use NVMe;
prep scripts require an actual mount and refuse root-disk fallback. Prior model
assets no longer need deletion to free RAM before the Wan2.2 arm.

## 2026-09-19: parity debugging

See [Wan parity gate investigation](wan_parity_gate_debug_20260919.md). Four GPU
arms show BF16 trajectory storage alone creates ~0.00195 log-probability drift;
parameter normalization, compilation, and batch shape each add independent
nonzero differences. The original trainer used its default 0.01 tolerance even
though the external queue required zero. The Wan preset now explicitly enforces
zero before the optimizer step. Original acceptance remains failed; H3 stays
blocked. Diagnostic passes do not replace that gate.
