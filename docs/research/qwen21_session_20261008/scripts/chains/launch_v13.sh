#!/bin/bash
# v13 = three-criterion judge (correct 2 / stray_text 1 / accurate 1 / like_ref pairwise 1), from base,
# 846 GPT-reference prompts with objects-only checklists. Pauses every 10 rounds to render the 22 gallery prompts.
cd /home/ubuntu/VRL; export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/tmp
M=/mnt/nvme/aes2/gptref_950_objects_abs.jsonl
# wait for the checklist expansion to finish (846 rows, process gone)
until [ "$(wc -l < $M 2>/dev/null)" -ge 846 ] && ! pgrep -f "^/mnt/nvme/venv-main/bin/python /mnt/nvme/aes2/expand_checklist.py" >/dev/null; do sleep 30; done
echo "checklists ready $(date +%T)"
# restart the judge so it loads the committed config (stray_text guard, correct weight 2)
pkill -f "^/mnt/nvme/venv-main/bin/python -m vrl.rewards.service.server"; sleep 15
CUDA_VISIBLE_DEVICES=3 setsid nohup /mnt/nvme/venv-main/bin/python -m vrl.rewards.service.server --config docs/research/qwen21_session_20261008/judge_configs/qwen3vl32b_pairwise_three.yaml > /mnt/nvme/logs/judge_three_v13.log 2>&1 < /dev/null &
until curl -sf http://127.0.0.1:8318/ready >/dev/null; do sleep 5; done
echo "judge three ready $(date +%T)"
R=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/vlm_gptref_v13_three
launch() {
  REWARD=qwen3vl_preference_http_tonal MANIFEST=$M OUT=vlm_gptref_v13_three LR=1.5e-4 OPT_STEPS=6 PPB=2 EPOCHS=300 SAVE=10 \
    EXTRA="reward.kwargs.tonal_anchor.luma_tolerance=0.08 reward.kwargs.tonal_anchor.saturation_tolerance=0.10 rollout.sde.window_range=[0,6] data.artifact_data_root=/mnt/nvme/data/external distributed.rollout.pipelined=true trainer.rollout_orchestration.lookahead_next_group=true distributed.rollout.health_check_timeout_s=120" \
    setsid nohup /mnt/nvme/work/launch_aes_vlm_stage1.sh > /mnt/nvme/logs/vlm_gptref_v13_supervise_$1.log 2>&1 < /dev/null &
}
launch 0; echo "v13 launched $(date +%T)"
for e in 10 20 30 40 50 60 80 100; do
  until [ -f $R/checkpoint-$e/checkpoint_meta.json ] && [ "$(tail -1 $R/metrics.csv | cut -d, -f1)" -ge $((e-1)) ]; do sleep 60; done; sleep 20
  PID=$(pgrep -f "^/mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise" | head -1); PG=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PG
  while kill -0 $PID 2>/dev/null; do sleep 5; done; sleep 15
  P=(); for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$s /mnt/nvme/venv-main/bin/python -m vrl.scripts.eval.image_checkpoint_eval --run-dir $R --checkpoint v13ck$e=$R/checkpoint-$e \
      --eval-policy-config /mnt/nvme/eval/policy_sharp_stage1.yaml --manifest /mnt/nvme/aes2/visual_prompts_$s.jsonl \
      --samples-per-prompt 1 --width 1024 --height 1024 --num-steps 40 --device cuda:0 --reward-device cpu \
      --output-dir /mnt/nvme/eval/visual_v13ck${e}_s$s > /mnt/nvme/logs/render_v13ck${e}_s$s.log 2>&1 & P+=($!)
  done; wait "${P[@]}"
  echo "rendered ck$e $(date +%T)"
  launch ck$e
done
