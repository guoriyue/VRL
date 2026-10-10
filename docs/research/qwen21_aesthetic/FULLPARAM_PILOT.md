# Qwen 2.1 aesthetic full-transformer pilot

**Original pilot: STOPPED by host OOM after checkpoint 1. The capped resume check also stopped during checkpoint gathering with a 600-second NCCL timeout; it did not save a new checkpoint. Work has moved to the four-GPU LoRA campaign. See [host-memory recovery](HOST_MEMORY_FIX.md). The old evaluation queue is not running.**

The user explicitly requested proceeding with aesthetic enhancement and using all four GPUs for full-parameter tuning after discussing the inconclusive reward qualification. This is an exploratory training experiment, not a claim that the earlier reward card passed. Preserve the original card and results.

- Four L40S GPUs, FSDP sharding over all four ranks. Full diffusion transformer is trainable; text encoder and VAE stay frozen.
- PickScore SFW prompt manifest, SigLIP Aesthetic Predictor V2.5 reward. Reward is image-only and does not enforce prompt adherence.
- Pilot configuration: `experiment/qwen_image_21/online_grpo_aesthetic_fullparam_4gpu`.
- Corrected sampling candidate: 1024 square, 40 steps, flow-GRPO noise 0.2. Native sampler remains the deployment/evaluation baseline.
- Conservative learning rate 1e-6, 1 PPO epoch, 4 replay timesteps per trajectory, 8 samples per prompt, 1 prompt per rank/update. Three-update acceptance run; checkpoint every update.
- No KL reference: existing full-model path has no adapter-off frozen reference. Do not describe PPO clipping as a replacement for a frozen-base KL penalty.
- Full optimizer/master state and checkpoint output belong on NVMe, not the nearly full root volume.
- Require finite gradients, unchanged-policy replay parity, optimizer progress and successful checkpoint save before promoting run length. Higher training reward alone does not demonstrate visual improvement.
- Evaluate checkpoint versus unchanged base with identical native sampling, prompts and seed mapping. Keep adherence and diversity visible alongside aesthetics. The old 120-prompt final test remains reserved.

Artifacts and launcher: `/mnt/nvme/outputs/qwen21_aesthetic_fullparam/`.

## First update evidence (2026-09-26 UTC)

The four-rank pilot reached a completed optimizer update. `training_debug.jsonl` records finite unchanged-policy replay with max absolute log-probability difference 4.76837158203125e-7 (limit 1e-5). Rank-local trainable shard size is 1,778,781,184 elements, approximately 7.115B parameters across four ranks. The first metrics row has finite gradient norm 0.002112 and zero pre-update clipping. This is an integration acceptance result, not evidence of aesthetic improvement.

Observed GPU allocation: about 17GB during rollout, 20.7GB during replay, 37GB after optimizer state initialization. CPU text encoding contributes significant rollout time (about 34–40 seconds per image); a subsequent run can separately validate GPU-resident text encoding. The initial corrected-sampling preflight used GPU text encoding and took about 17.5 seconds per image.

`evaluate_when_finished.py` waits for the current torchrun process to end and requires fresh successful result files from all four ranks plus the final checkpoint before running `evaluate_pilot.sh`. Evaluation is queued locally; it does not automatically publish new website results. The three-update pilot is deliberately bounded.

The clean base sampler comparison is published at https://qwen21-aesthetic-review.rigue.chatgpt.site/clean_sampling.html. All images there remain base-model outputs.

Checkpoint 1 was successfully published with global_step=1, `uses_lora=false`, and a 99,612,297,059-byte checkpoint containing optimizer/master state. An mmap-based comparison of eight attention-query weight slices against the pinned base found 1,916 changed elements across 1,048,576 checked elements; all checked weights were finite. This confirms a real model update, not just a metric/logging change. Full details: `/mnt/nvme/outputs/qwen21_aesthetic_fullparam/first_update_weight_check.json`.

Observed host memory peaked around 349 GiB while gathering the full checkpoint and dropped to about 221 GiB after publication. Checkpoint overhead is substantial and constrains this topology even though GPU memory fits.


## Host OOM incident

At 2026-09-25 22:18:10 local time, the kernel reported global host OOM and killed Ray generation workers. At 22:18:16 systemd killed the enclosing tmux scope, which recorded a 366.7 GiB peak. This also terminated the interactive session and evaluation queue. Checkpoint 1 remains on disk; no final checkpoint or completed evaluation exists. The earlier 349 GiB checkpoint observation did not cover the full subsequent transition. The precise peak allocation remains unisolated. Do not rerun this unchanged topology; reduce host duplication and isolate the training job under a memory-limited service before resuming. Incident evidence is saved under `/mnt/nvme/outputs/qwen21_aesthetic_fullparam/incident/`.
