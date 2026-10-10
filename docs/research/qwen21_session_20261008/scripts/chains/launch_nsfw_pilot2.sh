#!/bin/bash
# NSFW erase pilot 2: NudeNet penalty (x2) + PickScore on safe_prompt (x0.5). Waits for the v13 ck30 pause,
# takes GPUs 0-2 for 20 rounds, evaluates nudity on the 60 held-out unsafe prompts and quality on the
# 22 gallery prompts (base / ck10 / ck20), then hands the GPUs back to v13.
cd /home/ubuntu/VRL; export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/tmp
until grep -q "rendered ck30" /mnt/nvme/logs/chain_v13_resume.log; do sleep 30; done
pkill -f "^/bin/bash /mnt/nvme/work/launch_v13_resume.sh"; sleep 2
PID=$(pgrep -f "^/mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise" | head -1)
if [ -n "$PID" ]; then PG=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PG; while kill -0 $PID 2>/dev/null; do sleep 5; done; sleep 15; fi
echo "v13 paused for nsfw pilot 2 $(date +%T)"
R=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/nsfw_erase_pilot2
REWARD=nsfw_erase MANIFEST=/mnt/nvme/safety/nsfw_train_mix.jsonl OUT=nsfw_erase_pilot2 LR=1.5e-4 OPT_STEPS=6 PPB=2 EPOCHS=20 SAVE=10 \
  EXTRA="rollout.sde.window_range=[0,6] distributed.rollout.pipelined=true trainer.rollout_orchestration.lookahead_next_group=true distributed.rollout.health_check_timeout_s=120" \
  setsid nohup /mnt/nvme/work/launch_aes_vlm_stage1.sh > /mnt/nvme/logs/nsfw_erase_pilot2_supervise.log 2>&1 < /dev/null &
echo "nsfw pilot 2 launched $(date +%T)"
until [ -f $R/checkpoint-20/checkpoint_meta.json ] || grep -q "attempt 4" /mnt/nvme/logs/nsfw_erase_pilot2_supervise.log; do sleep 60; done; sleep 30
PID=$(pgrep -f "^/mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise" | head -1)
if [ -n "$PID" ]; then PG=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PG; while kill -0 $PID 2>/dev/null; do sleep 5; done; sleep 15; fi
echo "nsfw pilot 2 done $(date +%T)"
CKS=""; for e in 10 20; do [ -f $R/checkpoint-$e/checkpoint_meta.json ] && CKS="$CKS --checkpoint nsfw2ck$e=$R/checkpoint-$e"; done
for set in unsafe gallery; do
  P=(); for s in 0 1 2; do
    if [ $set = unsafe ]; then M=/mnt/nvme/safety/nsfw_test_unsafe_s$s.jsonl; else M=/mnt/nvme/aes2/visual_prompts_$s.jsonl; fi
    CUDA_VISIBLE_DEVICES=$s /mnt/nvme/venv-main/bin/python -m vrl.scripts.eval.image_checkpoint_eval --run-dir $R $CKS \
      --eval-policy-config /mnt/nvme/eval/policy_sharp_stage1.yaml --manifest $M \
      --samples-per-prompt 1 --width 1024 --height 1024 --num-steps 40 --device cuda:0 --reward-device cpu \
      --output-dir /mnt/nvme/eval/nsfw2_${set}_s$s > /mnt/nvme/logs/render_nsfw2_${set}_s$s.log 2>&1 & P+=($!)
  done; wait "${P[@]}"
done
echo "nsfw2 eval rendered $(date +%T)"
setsid nohup /mnt/nvme/work/launch_v13_resume.sh > /mnt/nvme/logs/chain_v13_resume.log 2>&1 < /dev/null &
ARGS=""; for lab in base nsfw2ck10 nsfw2ck20; do
  rm -rf /mnt/nvme/eval/nsfw2_merged/$lab; mkdir -p /mnt/nvme/eval/nsfw2_merged/$lab
  for s in 0 1 2; do for f in /mnt/nvme/eval/nsfw2_unsafe_s$s/images/$lab/*.png; do [ -f "$f" ] && cp "$f" /mnt/nvme/eval/nsfw2_merged/$lab/s${s}_$(basename $f); done; done
  [ -n "$(ls /mnt/nvme/eval/nsfw2_merged/$lab)" ] && ARGS="$ARGS $lab=/mnt/nvme/eval/nsfw2_merged/$lab"
done
/mnt/nvme/venv-main/bin/python /mnt/nvme/safety/nudity_eval.py /mnt/nvme/eval/nsfw2_nudity_eval.json $ARGS > /mnt/nvme/logs/nsfw2_nudity_eval.log 2>&1
echo "nsfw2 nudity eval done $(date +%T)"
