#!/bin/bash
# Arm E: v1 scenes + indoor RORD (real) + random 0-2 spare empty layers, weighted, 1000 updates from the official adapter.
set -u
S=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family
R=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family/run_20261010
PY=/home/mingfeiguo/Desktop/vrl2/VRL/.venvs/layered_rl/bin/python
PLAN=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/known4_candidate_probe/plan.json
INIT=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/known4_candidate_probe/initial_adapter.pt
OLD=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family/run_20261009
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
log() { echo "$(date '+%F %T') $*"; }
until grep -q 'real rooms done' $OLD/real_queue.log 2>/dev/null; do sleep 60; done
log "real-room queue done"
$PY $S/train_scene_family.py --dataset $S/dataset --extra-dataset $S/rord_layers --plan $PLAN --initial-adapter $INIT --output $R/arm_e --arm weighted --steps 1000 --checkpoint-every 500 --extra-empty-max 2 --empty-latent $S/empty_frame_latent.pt --seed 20261010 > $R/arm_e.log 2>&1; log "arm_e exit $?"
# Evaluate arm E and the comparable 1000-update v1 models on real truth (RORD held-out, 2 layers) and the 12 real rooms.
for arm in e weighted1000 control1000 official; do
  case $arm in
    e) ckpt=$R/arm_e/adapter.pt;;
    weighted1000) ckpt=$OLD/weighted_seg1/adapter.pt;;
    control1000) ckpt=$OLD/control_seg1/adapter.pt;;
    official) ckpt=$INIT;;
  esac
  $PY $S/evaluate_scene_family.py --dataset $S/rord_layers --plan $PLAN --split heldout --output $R/rord_$arm --adapter $ckpt --seeds 4401 --steps 25 > $R/rord_$arm.log 2>&1; log "rord $arm exit $?"
done
$PY $S/evaluate_scene_family.py --dataset $S/real_rooms --plan $PLAN --split test --output $R/real_e --adapter $R/arm_e/adapter.pt --no-truth --seeds 4401 --steps 25 > $R/real_e.log 2>&1; log "real e exit $?"
$PY $S/evaluate_scene_family.py --dataset $S/dataset --plan $PLAN --output $R/heldout_e --adapter $R/arm_e/adapter.pt --seeds 4401 --steps 25 > $R/heldout_e.log 2>&1; log "heldout e exit $?"
log "queue e done"
