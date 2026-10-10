#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/VRL
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/ubuntu/VRL CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/nvme/outputs/qwen21_aesthetic_fullparam/tmp
export RAY_TMPDIR=/mnt/nvme/q21ray
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$RAY_TMPDIR" /mnt/nvme/outputs/qwen21_aesthetic_fullparam/parking
exec /mnt/nvme/venv-main/bin/torchrun --standalone --nproc_per_node=4 --max-restarts=0 -m vrl.scripts.train --config experiment/qwen_image_21/online_grpo_aesthetic_fullparam_4gpu "$@"
