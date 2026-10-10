#!/bin/bash
# One training arm on one GPU: 2 x 1000 supervised updates from the official adapter, then the frozen evaluations.
# Usage: GPU=0 ARM=sim_only DATA=/data/layering PY=/path/python PLAN=/data/layering/plan.json ./run_arm.sh [extra train flags]
# DATA holds the bundle (see README): dataset/ dataset_v2/ rord_layers/ rord_layers_matte/ real_rooms/ empty_frame_latent.pt initial_adapter.pt
# Arm presets pick the training data; anything after the script name is passed to train_scene_family.py.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${GPU:?}" "${ARM:?}" "${DATA:?}" "${PY:?}" "${PLAN:?}"
OUT=${OUT:-$DATA/runs/$ARM}
SEED=${SEED:-20261011}
WEIGHTING=${WEIGHTING:-weighted}
export CUDA_VISIBLE_DEVICES=$GPU VRL_GPU_LOCK=/tmp/vrl-layering-gpu$GPU.lock OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
case $ARM in
  sim_only*) DATASETS=(--dataset $DATA/dataset_v2 --extra-dataset $DATA/dataset);;
  sim_rord_matte*) DATASETS=(--dataset $DATA/dataset_v2 --extra-dataset $DATA/dataset --extra-dataset $DATA/rord_layers_matte);;
  sim_v1_only*) DATASETS=(--dataset $DATA/dataset);;
  *) echo "unknown ARM preset: $ARM (sim_only*, sim_rord_matte*, sim_v1_only*)"; exit 2;;
esac
mkdir -p $OUT
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/queue.log; }
COMMON=(--plan $PLAN --initial-adapter $DATA/initial_adapter.pt --arm $WEIGHTING --steps 1000 --checkpoint-every 500 --extra-empty-max 2 --empty-latent $DATA/empty_frame_latent.pt --seed $SEED)
$PY $HERE/train_scene_family.py "${DATASETS[@]}" "${COMMON[@]}" --output $OUT/seg1 "$@" > $OUT/seg1.log 2>&1; log "seg1 exit $?"
$PY $HERE/train_scene_family.py "${DATASETS[@]}" "${COMMON[@]}" --output $OUT/seg2 --resume-from $OUT/seg1 "$@" > $OUT/seg2.log 2>&1; log "seg2 exit $?"
A=$OUT/seg2/adapter.pt
EVAL=(--plan $PLAN --adapter $A --seeds 4401 --steps 25)
$PY $HERE/evaluate_scene_family.py --dataset $DATA/rord_layers --split heldout --output $OUT/eval_rord "${EVAL[@]}" > $OUT/eval_rord.log 2>&1; log "eval rord exit $?"
$PY $HERE/evaluate_scene_family.py --dataset $DATA/real_rooms --split test --no-truth --output $OUT/eval_real_rooms "${EVAL[@]}" > $OUT/eval_real_rooms.log 2>&1; log "eval real rooms exit $?"
$PY $HERE/evaluate_scene_family.py --dataset $DATA/dataset --output $OUT/eval_heldout_v1 "${EVAL[@]}" > $OUT/eval_heldout_v1.log 2>&1; log "eval heldout v1 exit $?"
$PY $HERE/evaluate_scene_family.py --dataset $DATA/dataset_v2 --max-scenes 24 --output $OUT/eval_heldout_v2 "${EVAL[@]}" > $OUT/eval_heldout_v2.log 2>&1; log "eval heldout v2 exit $?"
log "arm $ARM done"
