#!/bin/bash
# Pilot 3: nudity removal on the uncensored Qwen-Image-2.1 transformer (0xSojalSec UC-BF16, local diffusers dir)
# with the Qwen3-VL-32B judge as the reward (no_nudity x2, faithful-to-safe_prompt x1, correct x1).
# Clean prompts only (audit2). 20 rounds at lr 1e-4, then base/ck10/ck20 renders, NudeNet + VLM evaluation.
cd /home/ubuntu/VRL; export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/tmp
until grep -q "rendered gallery" /mnt/nvme/logs/chain_render_uc_base.log; do sleep 30; done
pkill -f "^/mnt/nvme/venv-main/bin/python -m vrl.rewards.service.server"; sleep 15
CUDA_VISIBLE_DEVICES=3 setsid nohup /mnt/nvme/venv-main/bin/python -m vrl.rewards.service.server --config docs/research/qwen21_session_20261008/judge_configs/qwen3vl32b_nudity_erase.yaml > /mnt/nvme/logs/judge_nudity_erase.log 2>&1 < /dev/null &
until curl -sf http://127.0.0.1:8318/ready >/dev/null; do sleep 5; done
echo "judge nudity_erase ready $(date +%T)"
R=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/nsfw_erase_vlm_uc
REWARD=qwen3vl_preference_http MANIFEST=/mnt/nvme/safety/nsfw_train_mix_clean.jsonl OUT=nsfw_erase_vlm_uc LR=1.0e-4 OPT_STEPS=6 PPB=2 EPOCHS=20 SAVE=10 \
  EXTRA="model.path=/mnt/nvme/models/qwen_image_21_uc model.revision=null rollout.sde.window_range=[0,6] distributed.rollout.pipelined=true trainer.rollout_orchestration.lookahead_next_group=true distributed.rollout.health_check_timeout_s=120" \
  setsid nohup /mnt/nvme/work/launch_aes_vlm_stage1.sh > /mnt/nvme/logs/nsfw_erase_vlm_uc_supervise.log 2>&1 < /dev/null &
echo "pilot 3 launched $(date +%T)"
until [ -f $R/checkpoint-20/checkpoint_meta.json ] || grep -q "attempt 4" /mnt/nvme/logs/nsfw_erase_vlm_uc_supervise.log; do sleep 60; done; sleep 30
PID=$(pgrep -f "^/mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise" | head -1)
if [ -n "$PID" ]; then PG=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PG; while kill -0 $PID 2>/dev/null; do sleep 5; done; sleep 15; fi
echo "pilot 3 done $(date +%T)"
CKS=""; for e in 10 20; do [ -f $R/checkpoint-$e/checkpoint_meta.json ] && CKS="$CKS --checkpoint ucvlm$e=$R/checkpoint-$e"; done
for set in unsafe gallery; do
  P=(); for s in 0 1 2; do
    if [ $set = unsafe ]; then M=/mnt/nvme/safety/nsfw_test_clean_s$s.jsonl; else M=/mnt/nvme/aes2/visual_prompts_$s.jsonl; fi
    CUDA_VISIBLE_DEVICES=$s /mnt/nvme/venv-main/bin/python -m vrl.scripts.eval.image_checkpoint_eval --run-dir $R $CKS \
      --eval-policy-config /mnt/nvme/eval/policy_uc.yaml --manifest $M \
      --samples-per-prompt 1 --width 1024 --height 1024 --num-steps 40 --device cuda:0 --reward-device cpu \
      --output-dir /mnt/nvme/eval/ucvlm_${set}_s$s > /mnt/nvme/logs/render_ucvlm_${set}_s$s.log 2>&1 & P+=($!)
  done; wait "${P[@]}"
done
echo "pilot 3 eval rendered $(date +%T)"
ARGS=""; for lab in base ucvlm10 ucvlm20; do
  D=/mnt/nvme/eval/ucvlm_merged/$lab; rm -rf $D; mkdir -p $D
  for s in 0 1 2; do for f in /mnt/nvme/eval/ucvlm_unsafe_s$s/images/$lab/*.png; do [ -f "$f" ] && cp "$f" $D/s${s}_$(basename $f); done; done
  ARGS="$ARGS $lab=$D"
done
/mnt/nvme/venv-main/bin/python /mnt/nvme/safety/nudity_eval.py /mnt/nvme/eval/ucvlm_nudity_eval.json $ARGS > /mnt/nvme/logs/ucvlm_nudity_eval.log 2>&1
CUDA_VISIBLE_DEVICES=0 /mnt/nvme/venv-main/bin/python /mnt/nvme/safety/vlm_nudity_eval.py /mnt/nvme/eval/ucvlm_vlm_nudity_eval.json $ARGS > /mnt/nvme/logs/ucvlm_vlm_nudity_eval.log 2>&1
echo "pilot 3 nudity eval done $(date +%T)"
