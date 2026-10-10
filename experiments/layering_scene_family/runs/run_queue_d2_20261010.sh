#!/bin/bash
# Arm D: v2 scenes (1000) + v1 scenes (200) + indoor RORD (180), random 0-2 spare layers, weighted, 2000 updates from the official adapter.
set -u
S=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family
R=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/scene_family/run_20261010
PY=/home/mingfeiguo/Desktop/vrl2/VRL/.venvs/layered_rl/bin/python
PLAN=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/known4_candidate_probe/plan.json
INIT=/home/mingfeiguo/Desktop/vrl2/VRL/outputs/layered_rl_pipeline_20261007/known4_candidate_probe/initial_adapter.pt
OLD=$S/run_20261009
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
log() { echo "$(date '+%F %T') $*"; }
# Until a reboot the loaded kernel module is 580.173.02; load the matching userspace libcuda.
if ! nvidia-smi > /dev/null 2>&1; then export LD_LIBRARY_PATH=$S/cuda_580.173.02${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
# D2: same as D but RORD foreground layers come from the difference matte (rord_layers_matte) instead of the coarse polygon.
while kill -0 581667 2>/dev/null; do sleep 60; done
grep -q cached_total $R/cache_v2.log || { log "cache v2 did not finish"; exit 1; }
$PY $S/cache_rord_layers.py --root $S/rord_layers_matte --plan $PLAN --prompt-cache $(ls $S/dataset/train/scene_*/cache.pt | head -1) --foreground-weight 0.5 --occluder matte > $R/cache_rord_matte.log 2>&1; log "cache rord matte exit $?"
log "d2 start"
$PY $S/train_scene_family.py --dataset $S/dataset_v2 --extra-dataset $S/dataset --extra-dataset $S/rord_layers_matte --plan $PLAN --initial-adapter $INIT --output $R/arm_d_seg1 --arm weighted --steps 1000 --checkpoint-every 500 --extra-empty-max 2 --empty-latent $S/empty_frame_latent.pt --seed 20261011 > $R/arm_d_seg1.log 2>&1; log "arm_d seg1 exit $?"
$PY $S/train_scene_family.py --dataset $S/dataset_v2 --extra-dataset $S/dataset --extra-dataset $S/rord_layers_matte --plan $PLAN --initial-adapter $INIT --output $R/arm_d_seg2 --arm weighted --steps 1000 --checkpoint-every 500 --extra-empty-max 2 --empty-latent $S/empty_frame_latent.pt --seed 20261011 --resume-from $R/arm_d_seg1 > $R/arm_d_seg2.log 2>&1; log "arm_d seg2 exit $?"
D=$R/arm_d_seg2/adapter.pt
$PY $S/evaluate_scene_family.py --dataset $S/rord_layers --plan $PLAN --split heldout --output $R/rord_d --adapter $D --seeds 4401 --steps 25 > $R/rord_d.log 2>&1; log "rord d exit $?"
$PY $S/evaluate_scene_family.py --dataset $S/rord_layers --plan $PLAN --split heldout --output $R/rord_control2000 --adapter $OLD/control_seg2/adapter.pt --seeds 4401 --steps 25 > $R/rord_control2000.log 2>&1; log "rord control2000 exit $?"
$PY $S/evaluate_scene_family.py --dataset $S/real_rooms --plan $PLAN --split test --output $R/real_d --adapter $D --no-truth --seeds 4401 --steps 25 > $R/real_d.log 2>&1; log "real d exit $?"
$PY $S/evaluate_scene_family.py --dataset $S/dataset --plan $PLAN --output $R/heldout_d --adapter $D --seeds 4401 --steps 25 > $R/heldout_d.log 2>&1; log "heldout v1 d exit $?"
for arm in d official control2000; do
  case $arm in d) ckpt=$D;; official) ckpt=$INIT;; control2000) ckpt=$OLD/control_seg2/adapter.pt;; esac
  $PY $S/evaluate_scene_family.py --dataset $S/dataset_v2 --plan $PLAN --output $R/heldout_v2_$arm --adapter $ckpt --seeds 4401 --steps 25 --max-scenes 24 > $R/heldout_v2_$arm.log 2>&1; log "heldout v2 $arm exit $?"
done
log "queue d done"
