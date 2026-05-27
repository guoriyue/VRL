#!/usr/bin/env bash
# Resume claude_anatomy GRPO training from the latest checkpoint.
#
# Looks for checkpoint-final in the original run dir. Falls back to the
# highest-numbered checkpoint-N if checkpoint-final isn't there (i.e. the
# original run crashed before completion).
#
# Overrides at resume time:
#   * trainer.resume_from = <found checkpoint>
#   * trainer.total_epochs = EXTRA_EPOCHS (env var, default 20)
#   * trainer.save_freq = 1   (per-step checkpoint so we never lose progress)
#   * trainer.output_dir = <original>_resume   (don't clobber the original)

set -euo pipefail

REPO=/home/mingfeiguo/Desktop/wm-infra
ORIG_DIR="${ORIG_DIR:-$REPO/outputs/anima_preview3_claude_anatomy_grpo}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-20}"
RESUME_OUT="${ORIG_DIR}_resume"

cd "$REPO"

# Pick checkpoint: prefer checkpoint-final, otherwise highest checkpoint-N.
if [ -d "$ORIG_DIR/checkpoint-final" ]; then
  CHECKPOINT="$ORIG_DIR/checkpoint-final"
else
  CHECKPOINT="$(ls -d "$ORIG_DIR"/checkpoint-* 2>/dev/null \
    | grep -E 'checkpoint-[0-9]+$' \
    | sort -V \
    | tail -1 || true)"
fi

if [ -z "${CHECKPOINT:-}" ] || [ ! -d "$CHECKPOINT" ]; then
  echo "[resume] FATAL: no checkpoint found in $ORIG_DIR" >&2
  exit 1
fi

mkdir -p "$RESUME_OUT"
echo "[resume] from=$CHECKPOINT"  | tee -a "$RESUME_OUT/train.log"
echo "[resume] extra_epochs=$EXTRA_EPOCHS" | tee -a "$RESUME_OUT/train.log"
echo "[resume] output_dir=$RESUME_OUT" | tee -a "$RESUME_OUT/train.log"
echo "[resume] started at $(date -Iseconds)" | tee -a "$RESUME_OUT/train.log"

python -u -c "
import asyncio
from vrl.config.loading import load_config
from vrl.scripts.diffusion.cosmos.train import train_anima_grpo

cfg = load_config('experiment/diffusion/anima_preview3/online_grpo_anatomy')
cfg.trainer.total_epochs = int('$EXTRA_EPOCHS')
cfg.trainer.resume_from = '$CHECKPOINT'
cfg.trainer.save_freq = 1
cfg.trainer.output_dir = '$RESUME_OUT'
asyncio.run(train_anima_grpo(cfg))
" >> "$RESUME_OUT/train.log" 2>&1

echo "[resume] finished at $(date -Iseconds)" | tee -a "$RESUME_OUT/train.log"
