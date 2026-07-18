#!/usr/bin/env bash
# Symmetric colocated FSDP2 2x1 launcher — run on BOTH nodes.
#
#   node A (master/rank0):  ./fsdp_2x1_launch.sh 0
#   node B (worker/rank1):  ./fsdp_2x1_launch.sh 1
#
# Each node runs ONE torchrun rank = one FSDP2 shard + its own local colocated
# rollout + reward (each node does its own ray.init; there is NO shared Ray
# cluster). Params/grads/optimizer are sharded as DTensor across the 2 ranks and
# the per-layer all-gather / reduce-scatter crosses the network. The two ranks
# draw DISJOINT prompt slices, so 2 ranks = 2x the effective batch (genuine data
# parallelism), not a duplicated run.
#
# DIFFERENT from the DDP launcher:
#   - NO trainer.resume_from: FSDP2 optimizer-state export/load is not implemented
#     yet (build_strategy fail-fasts on resume_from), so this launcher does not
#     auto-resume. Re-running starts fresh from the base checkpoint.
#   - NO find_unused_parameters: that is a DDP reducer knob; FSDP has no reducer.
#   - model.torch_compile.enable=false is set in the config (required for FSDP2).
#
# Everything run-specific is an env var with a default; override per run, e.g.:
#   OUT=outputs/my_fsdp_run \
#   EXTRA_OVERRIDES="sampling.width=832 sampling.height=480 sampling.num_frames=33 \
#     rollout.prompts_per_batch=16 rollout.sample_batch_size=4 \
#     rollout.trajectory_storage.dtype=bfloat16" \
#   ./fsdp_2x1_launch.sh 0
set -uo pipefail
RANK="${1:?usage: fsdp_2x1_launch.sh <node_rank 0|1>}"

# --- per-cluster config (override via env) ---
NODE_A="${NODE_A:-172.31.36.21}"          # master / rank0 host (reachable from rank1)
REPO="${REPO:-$HOME/VRL}"
IFACE="${IFACE:-enp39s0}"                 # NCCL/GLOO socket interface (must match on both)
PORT="${PORT:-29500}"
OUT="${OUT:-outputs/fsdp_2x1_run}"
CONFIG="${CONFIG:-experiment/cosmos_predict2_5/online_nft_kling_video_reward_fsdp_2x1}"
SAVE_FREQ="${SAVE_FREQ:-3}"
# run-specific knobs (resolution / batch / memory) — NOT hardcoded:
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

cd "$REPO"; source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HOME="${HF_HOME:-/mnt/nvme/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE" NCCL_DEBUG=WARN
mkdir -p "$OUT"

exec torchrun --max-restarts=0 --nnodes=2 --node_rank="$RANK" --nproc_per_node=1 \
  --master_addr="$NODE_A" --master_port="$PORT" -m vrl.scripts.train \
  --config "$CONFIG" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.output_dir="$OUT" \
  $EXTRA_OVERRIDES \
  >> "$OUT/rank${RANK}.log" 2>&1
