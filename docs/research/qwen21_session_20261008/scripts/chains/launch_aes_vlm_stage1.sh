#!/bin/bash
# Stage 1: shared-noise early-window preference GRPO with the Qwen3-VL-32B generative judge.
# FSDP ranks on GPUs 0-2 (colocated trainer/rollout); judge service on GPU 3 (port 8318) must be up:
#   CUDA_VISIBLE_DEVICES=3 python -m vrl.rewards.service.server --config vrl/config/reward_service/${SERVICE:-qwen3vl32b_pairwise}.yaml
cd /home/ubuntu/VRL
export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/tmp CUDA_VISIBLE_DEVICES=0,1,2
exec /mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise --config experiment/qwen_image_21/online_grpo_aesthetic_shared_noise_4gpu \
  --max-attempts 6 \
  /reward=${REWARD:-vlm_preference_http} distributed.training.gpus_per_node=3 \
  data.manifest=${MANIFEST:-/mnt/nvme/aes2/stage1_prompts.jsonl} \
  actor.optim.lr=${LR:-1.0e-4} actor.optimizer_steps_per_batch=${OPT_STEPS:-4} \
  rollout.prompts_per_batch=${PPB:-2} \
  trainer.total_epochs=${EPOCHS:-60} trainer.save_freq=${SAVE:-5} \
  trainer.output_dir=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/${OUT:-vlm_stage1} ${EXTRA}
