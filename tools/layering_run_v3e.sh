#!/usr/bin/env bash
# Continue the authorized new-reward experiment after baseline and rollout checks.
set -Eeuo pipefail
cd /home/ubuntu/VRL-layering
export HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/nvme/tmp RAY_TMPDIR=/mnt/nvme/ray
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
PY=/mnt/nvme/venv-layering/bin/python
TORCHRUN=/mnt/nvme/venv-layering/bin/torchrun
RUN=/mnt/nvme/layering
CONFIG=experiment/qwen_image_layered/online_grpo_layered_grouping_fsdp_4x_l4
mkdir -p "$RUN/trainer-parking"

stage() {
  "$PY" - "$RUN/stage.json" "$1" <<'PY'
import datetime, json, pathlib, sys
record = {"stage": sys.argv[2], "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
PY
}
trap 'stage failed' ERR

"$PY" - "$RUN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "eval_inputs/photo_rows.jsonl").read_text().splitlines()]
for row in rows:
    assert (root / "eval/v3c_28" / row["id"] / "seed4401/main_scores.json").is_file()
reports = sorted((root / "rollout_preflight").glob("rank_*/result.json"))
assert reports, "one real training row must pass before the FSDP check"
for path in reports:
    result = json.loads(path.read_text())
    assert result["passed"] and all(result["conditioning_exact"].values()), result
PY

if [[ "${LAYERING_SKIP_VERIFIED_PREFLIGHT:-0}" != 1 ]]; then
stage fsdp_preflight
"$TORCHRUN" --standalone --nproc-per-node=4 tools/layering_fsdp_preflight.py \
  --checkpoint "$RUN/checkpoint-28-fsdp4-localcuda" \
  --cache "$RUN/eval_inputs/test/coco_L4_r00/cache.pt" \
  --witness "$RUN/rollout_preflight/rank_1/replay_witness.pt" \
  --output "$RUN/fsdp_preflight" > "$RUN/logs/fsdp_preflight.log" 2>&1
else
"$PY" - "$RUN/fsdp_preflight" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for rank in range(4):
    report = json.loads((root / f"rank_{rank}.json").read_text())
    assert report["all_finite"] and report["layers"] == 11
    assert report["prediction_max_abs_diff"] == report["logprob_max_abs_diff"] == 0
PY
fi

train_segment() {
  local stop=$1 checkpoint=$2
  stage "training_to_$stop"
  "$TORCHRUN" --standalone --nproc-per-node=4 -m vrl.scripts.train \
    --config "$CONFIG" model.lora.path=null trainer.resume_from="$checkpoint" \
    trainer.total_epochs="$stop" trainer.output_dir="$RUN/train" \
    > "$RUN/logs/train_to_$stop.log" 2>&1
  "$PY" outputs/layered_grpo/eval/export_live_adapter.py \
    "$RUN/train/checkpoint-$stop" "$RUN/adapters/v3e_$stop"
}

evaluate() {
  local arm=$1 adapter=$2 rows=$3
  stage "evaluating_$arm"
  "$TORCHRUN" --standalone --nproc-per-node=4 tools/layering_eval_four_gpu.py \
    --rows "$rows" --adapter "$adapter" --output "$RUN/eval/$arm" \
    > "$RUN/logs/eval_$arm.log" 2>&1
  "$PY" tools/layering_summarize.py --rows "$rows" --output "$RUN/eval/$arm/summary.json" "$RUN/eval/$arm"
}

train_segment 36 "$RUN/checkpoint-28-fsdp4-localcuda"
evaluate v3e_36 "$RUN/adapters/v3e_36" "$RUN/eval_inputs/photo_rows.jsonl"
train_segment 44 "$RUN/train/checkpoint-36"
evaluate v3e_44 "$RUN/adapters/v3e_44" "$RUN/eval_inputs/all_rows.jsonl"
evaluate v3b_20 /home/ubuntu/VRL-layering/outputs/layered_grpo/eval/adapters/v3b_20 "$RUN/eval_inputs/all_rows.jsonl"
stage summarizing
"$PY" tools/layering_summarize.py --rows "$RUN/eval_inputs/photo_rows.jsonl" \
  --output "$RUN/photo_comparison.json" \
  "$RUN/eval/v3b_20" "$RUN/eval/v3c_28" "$RUN/eval/v3e_36" "$RUN/eval/v3e_44"
"$PY" tools/layering_summarize.py --rows "$RUN/eval_inputs/rendered_rows.jsonl" \
  --output "$RUN/rendered_comparison.json" "$RUN/eval/v3b_20" "$RUN/eval/v3e_44"
stage complete
