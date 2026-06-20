RESUME=""; LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | grep -v final | sed 's/.*checkpoint-//' | sort -n | tail -1)
[ -n "$LAST" ] && RESUME="trainer.resume_from=$OUT/checkpoint-$LAST"
torchrun --max-restarts=0 --nnodes=2 --node_rank=$RANK --nproc_per_node=1 \
  --master_addr=$NODE_A --master_port=29500 -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1 \
  rollout.rollout_batch_size=16 rollout.sample_batch_size=4 distributed.training.ddp.find_unused_parameters=true \
  sampling.width=832 sampling.height=480 sampling.num_frames=33 \
  rollout.trajectory_storage.device=cpu rollout.trajectory_storage.dtype=bfloat16 \
  model.torch_compile.enable=false \
  trainer.save_freq=3 \
  trainer.output_dir="$OUT" $RESUME \
  >> "$OUT/rank${RANK}.log" 2>&1
