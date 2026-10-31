#!/usr/bin/env bash
# Run only after controlled reward gates and rollout ranking review pass.
set -Eeuo pipefail
cd /home/ubuntu/VRL-layering
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/ray
RUN=/mnt/nvme/layering_natural_20261025
PY=/mnt/nvme/venv-layering/bin/python
TORCHRUN=/mnt/nvme/venv-layering/bin/torchrun
mkdir -p "$RUN/logs" "$RUN/trainer-parking"
stage() {
 "$PY" - "$RUN/stage.json" "$1" <<'PY'
import datetime,json,pathlib,sys
record={'stage':sys.argv[2],'utc':datetime.datetime.now(datetime.UTC).isoformat()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
PY
}
trap 'stage failed' ERR
"$PY" - "$RUN" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);assert json.loads((r/'reward_audit.json').read_text())['passed']
review=json.loads((r/'rollout_review.json').read_text());assert review['passed'] and review['media_directory_parity']
for i in range(4):assert json.loads((r/f'rollout_preflight/rank_{i}/result.json').read_text())['passed']
for row in map(json.loads,(r/'eval_rows.jsonl').read_text().splitlines()):
 assert len(list((r/'eval/baseline_28'/row['id']/'seed4401').glob('layer_*.png')))==row['layers']
PY
resume=/mnt/nvme/layering/checkpoint-28-fsdp4-localcuda
for step in 30 32; do
 stage "training_to_$step"
 "$TORCHRUN" --standalone --nproc-per-node=4 -m vrl.scripts.train \
  --config experiment/qwen_image_layered/online_grpo_natural_pilot \
  model.lora.path=null "trainer.resume_from=$resume" "trainer.total_epochs=$step" \
  > "$RUN/logs/train_to_$step.log" 2>&1
 stage "exporting_$step"
 "$PY" outputs/layered_grpo/eval/export_live_adapter.py "$RUN/train/checkpoint-$step" "$RUN/adapters/step_$step"
 stage "evaluating_$step"
 "$TORCHRUN" --standalone --nproc-per-node=4 -m tools.layering_eval_four_gpu \
  --rows "$RUN/eval_rows.jsonl" --adapter "$RUN/adapters/step_$step" \
  --output "$RUN/eval/step_$step" --reward-kwargs "$RUN/reward_kwargs.json" \
  > "$RUN/logs/eval_$step.log" 2>&1
 "$PY" -m tools.layering_natural_report --rows "$RUN/eval_rows.jsonl" --output "$RUN/comparison_$step.json" \
  "$RUN/eval/baseline_28" "$RUN/eval/step_$step" > "$RUN/logs/comparison_$step.log"
 "$PY" -m tools.layering_compare_checkpoints --baseline /mnt/nvme/layering/checkpoint-28-fsdp4-localcuda \
  --output "$RUN/weight_changes_$step.json" "$RUN/train/checkpoint-$step"
 decision=$("$PY" - "$RUN/comparison_$step.json" "step_$step" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));a=x['arms'][sys.argv[2]]['groups']['coco'];b=x['arms'][x['baseline']]['groups']['coco']
stop=a['total_complete_pure_objects']<b['total_complete_pure_objects']-2 or a['means']['merge_layers']>b['means']['merge_layers']+2/15+1e-9
print('stop' if stop else 'continue')
PY
 )
 if [ "$decision" = stop ]; then stage "stopped_photo_regression_$step"; exit 0; fi
 resume="$RUN/train/checkpoint-$step"
done
stage complete_pending_visual_review
