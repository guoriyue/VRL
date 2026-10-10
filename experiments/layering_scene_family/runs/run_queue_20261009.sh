#!/bin/bash
# Phase 3 queue over the randomized scene family. Sequential on one GPU.
# Evaluation protocol: 24 held-out scenes, seed 4401, Heun 25 steps (6-7 layer scenes cost ~8 min per 50-step sample).
set -u
S=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family
B=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007
R=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family/run_20261009
PY=/home/mingfeiguo/Desktop/vrl2/VRL/.venvs/layered_rl/bin/python
PLAN=$B/known4_candidate_probe/plan.json
INIT=$B/known4_candidate_probe/initial_adapter.pt
DS=$S/dataset
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
log() { echo "$(date '+%F %T') $*"; }
until grep -q '^done' $S/generate.log 2>/dev/null; do sleep 60; done
log "generation done"
$PY $S/cache_scene_family.py --dataset $DS --plan $PLAN --splits train heldout > $R/cache.log 2>&1; log "cache exit $?"
# Frozen controls on the full held-out split.
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/heldout_zero_b --adapter $INIT --zero-b --seeds 4401 --steps 25 > $R/heldout_zero_b.log 2>&1; log "zero_b heldout exit $?"
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/heldout_official --adapter $INIT --seeds 4401 --steps 25 > $R/heldout_official.log 2>&1; log "official heldout exit $?"
# Weighted arm, first 1000 updates, then the predeclared continue/stop decision.
$PY $S/train_scene_family.py --dataset $DS --plan $PLAN --initial-adapter $INIT --output $R/weighted_seg1 --arm weighted --steps 1000 --checkpoint-every 250 > $R/weighted_seg1.log 2>&1; log "weighted seg1 exit $?"
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/quick_official --adapter $INIT --seeds 4401 --max-scenes 6 --steps 25 > $R/quick_official.log 2>&1
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/quick_weighted_500 --adapter $R/weighted_seg1/adapter_500.pt --seeds 4401 --max-scenes 6 --steps 25 > $R/quick_weighted_500.log 2>&1
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/quick_weighted_1000 --adapter $R/weighted_seg1/adapter.pt --seeds 4401 --max-scenes 6 --steps 25 > $R/quick_weighted_1000.log 2>&1
/home/mingfeiguo/Desktop/vrl2/VRL/.venv/bin/python $S/decide_continue.py $R/quick_official $R/quick_weighted_500 $R/quick_weighted_1000 $R/decision_1000.json > $R/decision.log 2>&1
CONTINUE=$?
log "decision at 1000: exit $CONTINUE (0 = continue)"
WFINAL=$R/weighted_seg1/adapter.pt
TOTAL=1000
if [ $CONTINUE -eq 0 ]; then
  $PY $S/train_scene_family.py --dataset $DS --plan $PLAN --initial-adapter $INIT --output $R/weighted_seg2 --arm weighted --steps 1000 --checkpoint-every 250 --resume-from $R/weighted_seg1 > $R/weighted_seg2.log 2>&1; log "weighted seg2 exit $?"
  WFINAL=$R/weighted_seg2/adapter.pt; TOTAL=2000
fi
# Control arm with the same total budget.
$PY $S/train_scene_family.py --dataset $DS --plan $PLAN --initial-adapter $INIT --output $R/control_seg1 --arm control --steps 1000 --checkpoint-every 250 > $R/control_seg1.log 2>&1; log "control seg1 exit $?"
CFINAL=$R/control_seg1/adapter.pt
if [ $TOTAL -eq 2000 ]; then
  $PY $S/train_scene_family.py --dataset $DS --plan $PLAN --initial-adapter $INIT --output $R/control_seg2 --arm control --steps 1000 --checkpoint-every 250 --resume-from $R/control_seg1 > $R/control_seg2.log 2>&1; log "control seg2 exit $?"
  CFINAL=$R/control_seg2/adapter.pt
fi
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/heldout_weighted --adapter $WFINAL --seeds 4401 --steps 25 > $R/heldout_weighted.log 2>&1; log "weighted heldout exit $?"
$PY $S/evaluate_scene_family.py --dataset $DS --plan $PLAN --output $R/heldout_control --adapter $CFINAL --seeds 4401 --steps 25 > $R/heldout_control.log 2>&1; log "control heldout exit $?"
# Regression on previously inspected scenes (8-layer protocol, seed 4401), weighted final vs official.
for arm in official weighted; do
  root=$R/regression_$arm; mkdir -p $root; cp $B/stable8_noise02/plan.json $root/plan.json; ln -sfn $B/stable8_noise02/conditions $root/conditions; ln -sfn $INIT $root/initial_adapter.pt
  if [ $arm = official ]; then ckpt=$INIT; else ckpt=$WFINAL; fi
  for case in kitchen bedroom photo872 manga003; do
    $PY -m vrl.scripts.generation.qwen_layered_evaluate --root $root --arm tuned --checkpoint $ckpt --cases $case --seed 4401 >> $root/evaluate.log 2>&1; log "regression $arm $case exit $?"
  done
done
log "queue done"
