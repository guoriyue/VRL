#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/VRL
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 PYTHONPATH=/home/ubuntu/VRL
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec /mnt/nvme/venv-main/bin/python -m vrl.scripts.eval.image_checkpoint_eval \
 --run-dir /mnt/nvme/outputs/qwen21_aesthetic_fullparam/pilot \
 --checkpoint pilot3=/mnt/nvme/outputs/qwen21_aesthetic_fullparam/pilot/checkpoint-final \
 --manifest docs/research/qwen21_aesthetic/validation_prompts.jsonl \
 --limit 16 --samples-per-prompt 1 --width 1024 --height 1024 --num-steps 40 \
 --seed 20260926 --blind-seed 260925 \
 --eval-policy-override 'model.memory.cpu_resident=[]' \
 --device cuda:0 --reward-device cuda:0 \
 --output-dir /mnt/nvme/outputs/qwen21_aesthetic_fullparam/eval_pilot3 "$@"
