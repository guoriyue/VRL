#!/bin/bash
# Limb RL v2 chain (attached to the running trainer after ck10): pauses from ck20; finishes the ck10 count eval
# (96 original + 480 Codex prompts with 2/3/4/6/8 adults, 8 interaction types). Every 10 rounds: render 64
# held-out prompts (32 original 6/8-person + 32 new 2/3/4-person) x 1 for base and the checkpoint, score limb
# defects and exact people count, resume. Deadline 10 h after launch.
cd /home/ubuntu/VRL; export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/tmp
DEADLINE=$(date -d "today 11:18" +%s)
R1=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/limb_rl_v1
R2=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/limb_rl_v2
stop_trainer() {
  PID=$(pgrep -f "^/mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise" | head -1)
  if [ -n "$PID" ]; then PG=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PG; while kill -0 $PID 2>/dev/null; do sleep 5; done; sleep 15; fi
}
evaluate() {  # $1 run dir, $2 ckpt epoch, $3 shard prefix, $4 samples
  local R=$1 E=$2 PFX=$3 N=$4
  P=(); for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$s /mnt/nvme/venv-main/bin/python -m vrl.scripts.eval.image_checkpoint_eval --run-dir $R --checkpoint limbck$E=$R/checkpoint-$E \
      --eval-policy-config /mnt/nvme/eval/policy_sharp_stage1.yaml --manifest /mnt/nvme/limb/$PFX$s.jsonl \
      --samples-per-prompt $N --width 1024 --height 1024 --num-steps 40 --device cuda:0 --reward-device cpu \
      --output-dir /mnt/nvme/eval/limb_ck${E}_s$s > /mnt/nvme/logs/render_limb_ck${E}_s$s.log 2>&1 & P+=($!)
  done; wait "${P[@]}"
  CUDA_VISIBLE_DEVICES=0 /mnt/nvme/venv-main/bin/python /mnt/nvme/limb/vlm_limb_eval.py /mnt/nvme/eval/limb_eval_ck$E.json \
    /mnt/nvme/limb/${PFX}0.jsonl,/mnt/nvme/limb/${PFX}1.jsonl,/mnt/nvme/limb/${PFX}2.jsonl \
    "base=/mnt/nvme/eval/limb_ck${E}_s{s}/images/base" "limbck$E=/mnt/nvme/eval/limb_ck${E}_s{s}/images/limbck$E" > /mnt/nvme/logs/limb_eval_ck$E.log 2>&1
  echo "evaluated ck$E $(date +%T)"
}
launch_v2() {
  REWARD=qwen3vl_preference_http MANIFEST=/mnt/nvme/limb/corpus2/train_all_v2.jsonl OUT=limb_rl_v2 LR=1.0e-4 OPT_STEPS=6 PPB=2 EPOCHS=200 SAVE=10 \
    EXTRA="rollout.sde.window_range=[0,6] distributed.rollout.pipelined=true trainer.rollout_orchestration.lookahead_next_group=true distributed.rollout.health_check_timeout_s=120 $2" \
    setsid nohup /mnt/nvme/work/launch_aes_vlm_stage1.sh > /mnt/nvme/logs/limb_rl_v2_supervise_$1.log 2>&1 < /dev/null &
}
echo "chain v2c attached $(date +%T)"
for e in 20 30 40 50 60 70 80 90 100; do
  until [ -f $R2/checkpoint-$e/checkpoint_meta.json ] && [ "$(tail -1 $R2/metrics.csv | cut -d, -f1)" -ge $((e-1)) ]; do
    sleep 60; [ $(date +%s) -gt $DEADLINE ] && break
  done
  LAST=$e; [ -f $R2/checkpoint-$e/checkpoint_meta.json ] || LAST=$((e-10))
  stop_trainer
  [ -d /mnt/nvme/eval/limb_ck${LAST}_s0 ] || evaluate $R2 $LAST corpus2/eval_all_s 1
  if [ ! -f /mnt/nvme/eval/limb_eval_ck10.json ]; then
    CUDA_VISIBLE_DEVICES=0 /mnt/nvme/venv-main/bin/python /mnt/nvme/limb/vlm_limb_eval.py /mnt/nvme/eval/limb_eval_ck10.json \
      /mnt/nvme/limb/corpus2/eval_all_s0.jsonl,/mnt/nvme/limb/corpus2/eval_all_s1.jsonl,/mnt/nvme/limb/corpus2/eval_all_s2.jsonl \
      "base=/mnt/nvme/eval/limb_ck10_s{s}/images/base" "limbck10=/mnt/nvme/eval/limb_ck10_s{s}/images/limbck10" > /mnt/nvme/logs/limb_eval_ck10.log 2>&1
    echo "evaluated ck10 (late) $(date +%T)"
  fi
  if [ $(date +%s) -gt $DEADLINE ]; then echo "deadline reached $(date +%T)"; break; fi
  launch_v2 ck$LAST ""
done
echo "limb rl v2 chain finished $(date +%T)"
