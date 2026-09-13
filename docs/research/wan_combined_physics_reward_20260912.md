# Wan combined physics reward hardware evidence

Date: 2026-09-12 UTC. Candidate `ff2b7857` in
`/home/ubuntu/VRL-mgpu-integration`, Torch 2.12.0+cu130 with isolated
Transformers 5.13.0 overlay. One dedicated L40S, BF16, no quantization.

## What passed

Both real reward models can reside on one dedicated GPU. Production
`MultiReward` with `KlingVideoReward` and `VideoConPhysicsReward` completes
actual MP4 materialization, in-process scoring, component selection, weighted
aggregation, artifact cleanup and shutdown. No fake scorers were injected.

Pinned checkpoints:

- KlingTeam/VideoReward: `4f26600130683e6f1de9f5d463887f28e8ef995c`, step 11352.
- VideoCon: `2b908dfc044350a1441efc785235c0dc110f14e1`.
- Kling base config references Qwen2-VL-2B-Instruct `main`, resolved offline
  from the existing cache; this is not a new pinned-base loader capability.

The existing generated Wan input has SHA256
`b4b6e4c9cd85e5aa25ed1900260c08544e189bc44c23fcf4296e480512f9dc7a`.
Production input tensor is `[3,81,480,832]` at 16 fps, decoded from that video
without frame or geometry reduction. Kling preserves recipe minimum pixels
200704 and checkpoint sampling; VideoCon samples its default 32 frames.

| Measurement | Direct models | Production reward path |
| --- | --- | --- |
| Repeated calls | 2, identical scores | 2, identical scores |
| Hot wall | 2.351018 s | 4.051575 s |
| Peak allocated bytes | 19,407,443,456 | 19,409,704,960 |
| Kling motion score | -0.595759268 | -0.612729437 |
| VideoCon physics score | 0.439453125 | 0.439453125 |
| 0.3 motion + 0.7 physics | 0.128889407 | 0.123798356 |

Production hot scoring includes 1.657011 s of artifact encoding and 2.392817 s
reported inference. Cold production call including lazy loads is 42.124124 s.
Direct and production inputs are different encoded videos: do not interpret
their score difference as same-input model parity error. Scores are observations,
not quality labels or evidence of learning improvement.

## Reproduce

Artifacts under `/mnt/nvme/outputs/wan22_i2v_cache`:

- `combined_reward_gpu_probe.py`, `.log`, `.json`: direct co-residency.
- `combined_reward_runtime_probe.py`, `.log`, `.json`: production path.
- `combined_runtime_debug/`: request and result provenance.

Run either script from the candidate checkout with:

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/nvme/hf/huggingface \
  PYTHONPATH=/mnt/nvme/venvs/transformers-5.13-overlay:/home/ubuntu/VRL-mgpu-integration \
  /home/ubuntu/VRL/.venv/bin/python \
  /mnt/nvme/outputs/wan22_i2v_cache/combined_reward_runtime_probe.py
```

Both processes exited 0; runtime probe asserts finite weighted scores, repeated
total equality, no remaining temporary MP4s and completed shutdown. Fresh GPU
process inventory is empty. Shared dependencies and vendor source unchanged.

## Remaining gates

This supports a dedicated GPU reward layout without CPU VideoCon, but does not
prove distributed resource assignment, shared-GPU CuMem parking, full-size Wan
policy update or learning quality. The default recipe remains unchanged: moving
both scorers onto a shared policy GPU would violate its existing lifecycle
constraint. Validate dedicated placement before changing the execution recipe.
