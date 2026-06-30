#!/usr/bin/env bash
# Symmetric colocated DDP 2x1 launcher — resume-capable, run on BOTH nodes.
#
#   node A (master/rank0):  ./ddp_2x1_launch.sh 0
#   node B (worker/rank1):  ./ddp_2x1_launch.sh 1
#
# Each node runs ONE torchrun rank = one DDP replica + its own local colocated
# rollout + reward (each node does its own ray.init; there is NO shared Ray
# cluster — unlike the cross_node topology). Only the gradient all-reduce crosses
# the network. The two ranks draw DISJOINT prompt slices, so 2 ranks = 2x the
# effective batch (genuine data parallelism), not a duplicated run.
#
# Resume: rank1 first rsyncs rank0's checkpoints, then both resume from the
# latest `checkpoint-<N>`. Output is NEVER deleted — safe to re-run after a crash.
#
# Everything run-specific is an env var with a default; override per run, e.g.:
#   OUT=outputs/my_run \
#   EXTRA_OVERRIDES="sampling.width=832 sampling.height=480 sampling.num_frames=33 \
#     rollout.prompts_per_batch=16 rollout.sample_batch_size=4 \
#     rollout.trajectory_storage.dtype=bfloat16 model.torch_compile.enable=false" \
#   ./ddp_2x1_launch.sh 0
set -uo pipefail
RANK="${1:?usage: ddp_2x1_launch.sh <node_rank 0|1>}"

# --- per-cluster config (override via env) ---
NODE_A="${NODE_A:-172.31.36.21}"          # master / rank0 host (reachable from rank1)
PEM="${PEM:-$HOME/vrl.pem}"               # ssh key for rank1->rank0 checkpoint rsync
REPO="${REPO:-$HOME/VRL}"
IFACE="${IFACE:-enp39s0}"                 # NCCL/GLOO socket interface (must match on both)
PORT="${PORT:-29500}"
OUT="${OUT:-outputs/ddp_2x1_run}"
CONFIG="${CONFIG:-experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1}"
SAVE_FREQ="${SAVE_FREQ:-3}"
# run-specific knobs (resolution / batch / memory / compile) — NOT hardcoded:
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

cd "$REPO"; source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HOME="${HF_HOME:-/mnt/nvme/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE" NCCL_DEBUG=WARN
mkdir -p "$OUT"

# rank1 pulls rank0's checkpoints so both resume from the same step
if [ "$RANK" = "1" ]; then
  rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
    "ubuntu@$NODE_A:$REPO/$OUT/checkpoint-*" "$OUT"/ 2>/dev/null || true
fi
RESUME=""
LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | grep -v final | sed 's/.*checkpoint-//' | sort -n | tail -1)
[ -n "$LAST" ] && RESUME="trainer.resume_from=$OUT/checkpoint-$LAST"

# find_unused_parameters: NFT's previous/reference adapters require_grad but run
# under no_grad, so DDP's reducer would otherwise flag them as missing gradients.
exec torchrun --max-restarts=0 --nnodes=2 --node_rank="$RANK" --nproc_per_node=1 \
  --master_addr="$NODE_A" --master_port="$PORT" -m vrl.scripts.train \
  --config "$CONFIG" \
  distributed.training.ddp.find_unused_parameters=true \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.output_dir="$OUT" \
  $EXTRA_OVERRIDES $RESUME \
  >> "$OUT/rank${RANK}.log" 2>&1
