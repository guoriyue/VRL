# Full-parameter host-memory recovery

The first training attempt completed one optimizer update and saved checkpoint 1, then caused a host OOM at 366.7 GiB. The terminal was in the same systemd scope and was killed with the workers. GPU capacity alone did not establish full-cycle fit.

## Changes under validation

- Training-state NVMe parking now covers trainable model shards, FP32 optimizer masters, Adam moments and gradients, rather than frozen parameters only. Shared file mappings preserve tensor identity while allowing inactive pages to be reclaimed. State is restored before the optimizer runs.
- Checkpoint loading maps modern PyTorch checkpoint storages; multiple ranks no longer eagerly deserialize separate ~100 GB anonymous copies. Legacy non-ZIP serialization stays readable.
- Checkpoint save retains the gathered model on the primary rank only. The general export path still retains state on every rank for strict restore verification.
- A separate systemd user service runs the resume check with MemoryHigh=280G, MemoryMax=300G, MemorySwapMax=0 and no automatic restart. It is outside the interactive terminal's scope.

## Acceptance

The four-GPU disk-parking test passed next-update equivalence. Distributed FP32-master resume and checkpoint mapping tests passed. A real resume then exposed that saving and verification needed distinct export paths; that was corrected and covered by a real two-rank regression. The first capped attempt stopped before training, with a service memory peak below 50 GiB.

The production check resumes checkpoint 1 into a separate `resume_check` output directory, retaining 1024px/40 steps/noise 0.2 but using two samples per prompt. It runs through total epoch 3 (two additional updates), saving each update, to exercise a checkpoint followed by another rollout. This checks memory and correctness, not aesthetic efficacy. Do not promote to a long run until it finishes and the peak is reviewed.

Artifacts: `/mnt/nvme/outputs/qwen21_aesthetic_fullparam/incident/` and `/mnt/nvme/outputs/qwen21_aesthetic_fullparam/resume_check/`.

Status: `systemctl --user status qwen21-aesthetic-resume-check`.
Memory: `systemctl --user show qwen21-aesthetic-resume-check -p MemoryCurrent -p MemoryPeak -p MemoryMax`.
Stop: `systemctl --user stop qwen21-aesthetic-resume-check`.

Strict optimizer verification was also changed to use the primary-only checkpoint exporter. It still gathers and compares every restored shard, then agrees across ranks on the result; it no longer holds a complete Adam/master snapshot on each rank. Tests cover both a successful comparison and corrupt-state rejection on primary and peer paths.

Validation logs: `final_fix_tests.log` (39 passed, including CUDA distributed resume), `gpu_fix_tests.log` (four-GPU disk-parking equivalence passed), `save_restore_tests.log` (30 passed) and `optimizer_verify_tests.log` (28 passed). Some suites overlap; these are not unique test totals. Ruff and diff whitespace checks passed.

## Live acceptance status (2026-09-26 05:52 UTC)

The resumed full-parameter update completed with finite replay values, maximum log-probability difference 2.384e-7 (limit 1e-5), and gradient norm 0.004611. The job is still gathering checkpoint 2, not yet accepted as a complete cycle. Cgroup memory has reached about 285 GiB, above the 280 GiB soft limit, with no hard-limit hits or OOM kills at this observation. Reclaim throttling makes checkpoint gathering slow. The 300 GiB hard cap remains unchanged.

Process mapping inspection found roughly 13.6 GiB of resident plasma weight storage per rank (about 54 GiB across the four independent stores), in addition to sleeping generation/reward allocations. This remains a source of checkpoint-phase overhead; NVMe trainer parking alone does not remove it. Do not interpret an active service or a completed optimizer update as proof of full-cycle fit. No longer training run or evaluation has been launched.

## Resume check final outcome

The second capped run failed on 2026-09-25 around 23:00 local time. The first
reported failure was an NCCL collective exceeding its 600-second timeout during
checkpoint gathering. systemd recorded 284.8 GiB peak and exit-code failure,
not an OOM result. No checkpoint 2 was saved. Memory reclaim throttling is a
likely contributor, not a proven complete root cause. The original checkpoint 1
is preserved. The user subsequently authorized switching to four-GPU LoRA; see
[LORA_CAMPAIGN.md](LORA_CAMPAIGN.md). Full-parameter acceptance remains failed.
