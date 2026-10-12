#!/bin/bash
# One training arm on one GPU: 2 x 1000 supervised updates from the official adapter, then the frozen evaluations.
# Usage, from the repository root (repo installed editable in the active environment):
#   GPU=0 ARM=sim_only DATA=/data/layering docs/research/layering_scene_family/run_arm.sh [extra vrl-train overrides]
# DATA layout (see README.md): dataset/ dataset_v2/ rord_layers/ rord_layers_matte/ real_rooms/ [diverse_photos/]
#   stable_layers_peft/
# Arm presets pick the training data; anything after the script name is passed to vrl-train.
# Optional: OUT (default $DATA/runs/$ARM), SEED, WEIGHTING (weighted|control), SPARE_LAYERS_MAX (default 2),
#   STREAM (true on 32 GB cards), EVAL_EXTRA (for example precision.rollout.quantization.format=fp8 on 32 GB cards).
set -u
: "${GPU:?}" "${ARM:?}" "${DATA:?}"
OUT=${OUT:-$DATA/runs/$ARM}
SEED=${SEED:-20261011}
WEIGHTING=${WEIGHTING:-weighted}
SPARE_LAYERS_MAX=${SPARE_LAYERS_MAX:-2}
STREAM=${STREAM:-false}
EVAL_EXTRA=${EVAL_EXTRA:-}
EXPERIMENT=qwen_image_layered/offline_sft_layered_scenes
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
case $ARM in
  sim_only*) MANIFEST="{$DATA/dataset_v2: null, $DATA/dataset: null}";;
  sim_rord_matte*) MANIFEST="{$DATA/dataset_v2: null, $DATA/dataset: null, $DATA/rord_layers_matte: null}";;
  sim_v1_only*) MANIFEST="$DATA/dataset";;
  *) echo "unknown ARM preset: $ARM (sim_only*, sim_rord_matte*, sim_v1_only*)"; exit 2;;
esac
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/queue.log"; }
COMMON=("data.manifest=$MANIFEST" "data.token_weights=$WEIGHTING" "data.spare_layers_max=$SPARE_LAYERS_MAX"
  "model.lora.path=$DATA/stable_layers_peft" "trainer.seed=$SEED" "trainer.max_train_steps=1000"
  "trainer.checkpointing_steps=500" "actor.stream_frozen_blocks=$STREAM")
vrl-train --config "experiment/$EXPERIMENT" "trainer.output_dir=$OUT/seg1" "${COMMON[@]}" "$@" > "$OUT/seg1.log" 2>&1; log "seg1 exit $?"
vrl-train --config "experiment/$EXPERIMENT" "trainer.output_dir=$OUT/seg2" "trainer.resume_from=$OUT/seg1/checkpoint-final" \
  "trainer.max_train_steps=2000" "${COMMON[@]}" "$@" > "$OUT/seg2.log" 2>&1; log "seg2 exit $?"
EVALUATE=(python -m vrl.scripts.eval.layered_decompose_eval --experiment "$EXPERIMENT" --adapter "$OUT/seg2/checkpoint-final" --seeds 4401 --steps 25)
"${EVALUATE[@]}" --dataset "$DATA/rord_layers" --split heldout --output "$OUT/eval_rord" $EVAL_EXTRA > "$OUT/eval_rord.log" 2>&1; log "eval rord exit $?"
"${EVALUATE[@]}" --dataset "$DATA/real_rooms" --split test --no-truth --output "$OUT/eval_real_rooms" $EVAL_EXTRA > "$OUT/eval_real_rooms.log" 2>&1; log "eval real rooms exit $?"
if [ -d "$DATA/diverse_photos" ]; then
  "${EVALUATE[@]}" --dataset "$DATA/diverse_photos" --split test --no-truth --output "$OUT/eval_diverse" $EVAL_EXTRA > "$OUT/eval_diverse.log" 2>&1; log "eval diverse exit $?"
fi
"${EVALUATE[@]}" --dataset "$DATA/dataset" --output "$OUT/eval_heldout_v1" $EVAL_EXTRA > "$OUT/eval_heldout_v1.log" 2>&1; log "eval heldout v1 exit $?"
"${EVALUATE[@]}" --dataset "$DATA/dataset_v2" --max-scenes 24 --output "$OUT/eval_heldout_v2" $EVAL_EXTRA > "$OUT/eval_heldout_v2.log" 2>&1; log "eval heldout v2 exit $?"
log "arm $ARM done"
