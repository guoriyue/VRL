#!/bin/bash
# v15 = v14 (rewritten prompts, three-criterion judge) + single_scene collage penalty in the judge.
cd /home/ubuntu/VRL; export HF_HOME=/mnt/nvme/hf TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/tmp
until curl -sf http://127.0.0.1:8318/ready >/dev/null; do sleep 5; done
M=/mnt/nvme/aes2/gptref_950_rewritten_abs.jsonl
R=/mnt/nvme/outputs/qwen21_aesthetic_shared_noise/vlm_gptref_v15_single_scene
launch() {
  REWARD=qwen3vl_preference_http_tonal MANIFEST=$M OUT=vlm_gptref_v15_single_scene LR=1.5e-4 OPT_STEPS=6 PPB=2 EPOCHS=300 SAVE=10 \
    EXTRA="reward.kwargs.tonal_anchor.luma_tolerance=0.08 reward.kwargs.tonal_anchor.saturation_tolerance=0.10 rollout.sde.window_range=[0,6] data.artifact_data_root=/mnt/nvme/data/external distributed.rollout.pipelined=true trainer.rollout_orchestration.lookahead_next_group=true distributed.rollout.health_check_timeout_s=120" \
    setsid nohup /mnt/nvme/work/launch_aes_vlm_stage1.sh > /mnt/nvme/logs/vlm_gptref_v15_supervise_$1.log 2>&1 < /dev/null &
}
launch 0; echo "v15 launched $(date +%T)"
for e in 10 20 30 40 50 60; do
  until [ -f $R/checkpoint-$e/checkpoint_meta.json ] && [ "$(tail -1 $R/metrics.csv | cut -d, -f1)" -ge $((e-1)) ]; do sleep 60; done; sleep 20
  PID=$(pgrep -f "^/mnt/nvme/venv-main/bin/python -m vrl.scripts.supervise" | head -1); PG=$(ps -o pgid= -p $PID | tr -d ' '); kill -TERM -$PG
  while kill -0 $PID 2>/dev/null; do sleep 5; done; sleep 15
  P=(); for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$s /mnt/nvme/venv-main/bin/python -m vrl.scripts.eval.image_checkpoint_eval --run-dir $R --checkpoint v15ck$e=$R/checkpoint-$e \
      --eval-policy-config /mnt/nvme/eval/policy_sharp_stage1.yaml --manifest /mnt/nvme/aes2/visual_prompts_$s.jsonl \
      --samples-per-prompt 1 --width 1024 --height 1024 --num-steps 40 --device cuda:0 --reward-device cpu \
      --output-dir /mnt/nvme/eval/visual_v15ck${e}_s$s > /mnt/nvme/logs/render_v15ck${e}_s$s.log 2>&1 & P+=($!)
  done; wait "${P[@]}"
  echo "rendered ck$e $(date +%T)"
  launch ck$e
done
