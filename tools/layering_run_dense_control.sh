#!/usr/bin/env bash
# Paired pure-RL annotation comparison, with four ranks and serialized evaluation.
set -Eeuo pipefail
cd /home/ubuntu/VRL-layering
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/ray
RUN=/mnt/nvme/layering_dense_20261025
BASE=/mnt/nvme/layering_natural_20261025/eval/baseline_28
PY=/mnt/nvme/venv-layering/bin/python
TORCHRUN=/mnt/nvme/venv-layering/bin/torchrun
mkdir -p "$RUN/logs" "$RUN/trainer-parking"
exec 9>"$RUN/runner.lock"
flock -n 9 || exit 1
stage() {
 "$PY" - "$RUN/stage.json" "$1" <<'PY'
import datetime,json,os,pathlib,sys
record={'stage':sys.argv[2],'utc':datetime.datetime.now(datetime.UTC).isoformat(),'runner_pid':os.getppid()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
PY
}
trap 'stage failed' ERR
"$PY" - "$RUN" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1])
assert json.loads((r/'scheduled_reward_audit.json').read_text())['passed']
for p,digest in json.loads((r/'source_hashes.json').read_text()).items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==digest
for i in range(4):
 x=json.loads((r/f'preflight/rank_{i}/result.json').read_text())
 assert x['passed'] and all(x['conditioning_exact'].values()) and x['layers']==11
for p,digest in json.loads((r/'protocol.json').read_text())['manifest_sha256'].items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==digest
PY
for step in 32 36; do
 for arm in expanded legacy; do
  if [ -f "$RUN/$arm/stopped.json" ]; then continue; fi
  resume=/mnt/nvme/layering/checkpoint-28-fsdp4-localcuda
  if [ "$step" = 36 ]; then resume="$RUN/$arm/train/checkpoint-32"; fi
  stage "training_${arm}_to_$step"
  "$TORCHRUN" --standalone --nproc-per-node=4 -m vrl.scripts.train \
   --config experiment/qwen_image_layered/online_grpo_dense_control \
   model.lora.path=null "trainer.resume_from=$resume" "trainer.total_epochs=$step" \
   "data.manifest=$RUN/$arm/train.jsonl" "trainer.output_dir=$RUN/$arm/train" \
   "reward.kwargs.layered_grouping.dump_dir=$RUN/$arm/rollouts" \
   > "$RUN/logs/train_${arm}_to_$step.log" 2>&1
  stage "exporting_${arm}_$step"
  "$PY" outputs/layered_grpo/eval/export_live_adapter.py "$RUN/$arm/train/checkpoint-$step" "$RUN/$arm/adapters/step_$step"
  stage "evaluating_${arm}_$step"
  "$TORCHRUN" --standalone --nproc-per-node=4 -m tools.layering_eval_four_gpu \
   --rows "$RUN/eval_rows.jsonl" --adapter "$RUN/$arm/adapters/step_$step" \
   --output "$RUN/$arm/eval/step_$step" --reward-kwargs "$RUN/reward_kwargs.json" \
   > "$RUN/logs/eval_${arm}_$step.log" 2>&1
  for labels in original expanded; do
   rows="$RUN/eval_rows.jsonl"
   if [ "$labels" = expanded ]; then rows="$RUN/eval_expanded.jsonl"; fi
   "$PY" -m tools.layering_natural_report --rows "$rows" \
    --output "$RUN/$arm/comparison_${step}_${labels}.json" \
    "$BASE" "$RUN/$arm/eval/step_$step" > "$RUN/logs/report_${arm}_${step}_${labels}.log"
  done
  "$PY" - "$RUN/$arm" "$step" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);step=sys.argv[2]
x=json.loads((r/f'comparison_{step}_original.json').read_text())
a=x['arms'][f'step_{step}']['groups']['coco'];b=x['arms'][x['baseline']]['groups']['coco']
stop=a['total_complete_pure_objects']<b['total_complete_pure_objects']-2 or a['means']['merge_layers']>b['means']['merge_layers']+2/15+1e-9
if stop:(r/'stopped.json').write_text(json.dumps({'step':int(step),'reason':'original-photo regression gate'},indent=2)+'\n')
PY
 done
done
stage complete_pending_review
