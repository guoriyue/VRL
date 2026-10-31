#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/VRL-layering
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/nvme/tmp
RUN=/home/ubuntu/VRL-layering/outputs/layered_grpo/diagnostics_20261024
PY=/mnt/nvme/venv-layering/bin/python
stage() {
  "$PY" - "$RUN/stage.json" "$1" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps({'stage':sys.argv[2],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n');t.replace(p)
PY
}
trap 'stage failed' ERR
for arm in v3_seg3 v3c_28 v3e_44; do
  stage "evaluating_$arm"
  /mnt/nvme/venv-layering/bin/torchrun --standalone --nproc-per-node=4 \
    tools/layering_diagnostic_eval.py --plan "$RUN/plan.json" --arm "$arm" \
    --output "$RUN/eval" > "$RUN/logs/$arm.log" 2>&1
done
stage summarizing
"$PY" tools/layering_diagnostic_summary.py "$RUN" > "$RUN/logs/summary.log" 2>&1
stage complete
