#!/usr/bin/env bash
# Run only after reward, rollout/replay, and baseline gates have passed.
set -Eeuo pipefail
cd /home/ubuntu/VRL-layering
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/ray
PY=/mnt/nvme/venv-layering/bin/python
TORCHRUN=/mnt/nvme/venv-layering/bin/torchrun
RUN=/mnt/nvme/layering_integrity_20261025
mkdir -p "$RUN/logs" "$RUN/trainer-parking"
stage() {
  "$PY" - "$RUN/stage.json" "$1" <<'PY'
import datetime,json,pathlib,sys
record={'stage':sys.argv[2],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
}
trap 'stage failed' ERR
"$PY" - "$RUN" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); pre=json.loads((p/'preflight.json').read_text());assert pre['all_hidden_edits_lower_reward']
r=json.loads((p/'rollout_preflight/rank_0/reward_check.json').read_text());assert r['passed'] and r['reward_media_directory_parity']
rows=list(map(json.loads,(p/'eval_rows.jsonl').read_text().splitlines()))
assert len(rows)==25
for row in rows:assert (p/'eval/baseline_28'/row['id']/'seed4401/main_scores.json').is_file()
PY
stage training_28_to_32
"$TORCHRUN" --standalone --nproc-per-node=4 -m vrl.scripts.train \
 --config experiment/qwen_image_layered/online_grpo_integrity_pilot \
 model.lora.path=null trainer.resume_from=/mnt/nvme/layering/checkpoint-28-fsdp4-localcuda \
 > "$RUN/logs/train.log" 2>&1
for step in 30 32; do
 stage "exporting_$step"
 "$PY" outputs/layered_grpo/eval/export_live_adapter.py "$RUN/train/checkpoint-$step" "$RUN/adapters/step_$step"
 stage "evaluating_$step"
 "$TORCHRUN" --standalone --nproc-per-node=4 -m tools.layering_eval_four_gpu \
  --rows "$RUN/eval_rows.jsonl" --adapter "$RUN/adapters/step_$step" \
  --output "$RUN/eval/step_$step" --integrity-weight 1 \
  > "$RUN/logs/eval_$step.log" 2>&1
 done
stage comparison
"$PY" -m tools.layering_integrity_report --rows "$RUN/eval_rows.jsonl" --output "$RUN/comparison.json" \
 "$RUN/eval/baseline_28" "$RUN/eval/step_30" "$RUN/eval/step_32" > "$RUN/logs/comparison.log"
"$PY" -m tools.layering_compare_checkpoints --baseline /mnt/nvme/layering/checkpoint-28-fsdp4-localcuda \
 --output "$RUN/weight_changes.json" "$RUN/train/checkpoint-30" "$RUN/train/checkpoint-32"
stage complete
