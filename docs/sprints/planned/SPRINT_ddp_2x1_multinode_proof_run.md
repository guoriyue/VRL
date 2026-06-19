# SPRINT: symmetric colocated DDP — 真 2 机×1 卡 proof run (planned)

状态：**planned（2026-06-18 从 [[SPRINT_symmetric_colocated_ddp]] 拆出）**。DDP 的 Slice 1–4 代码已全部落地、CPU 可验证部分全绿（396 passed），那个 sprint 已归档 `done/`；这里收唯一卡住的 live gate——真实跨机 NCCL 跑，本机单卡无法验证。

## 0. 来历

[[SPRINT_symmetric_colocated_ddp]]（`done/`）的 `DDPStrategy`（`vrl/trainers/strategy.py:298-322`）、rank-aware online loop（`vrl/scripts/common/online.py:75/889-892`）、colocated 放置（gpu_pool=trainer）、Ray-copy weight-sync 均已实现并通过 CPU 测试。commit message 自称"first-run-verified on real 2x1 hardware"，但 `docs/training_examples/` 下无任何 ddp/2x1 proof 工件——即 live gate 未留证据。本仓约定：sprint 进 `done/` 需 live gate 过，故 proof run 单列。

## 1. Work items

- **首跑**：在真 2 服务器 × 1 GPU 上跑 `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1.yaml`（strategy=ddp / num_nodes=2 / gpus_per_node=1 / gpu_pool=trainer）。
- **验证三件事**：(a) NCCL 跨机 all-reduce 正确；(b) GPU 与 rollout 同驻显存不溢；(c) DDP-reducer × DiffusionNFT 多前向（boundary 路由下多次 transformer forward）梯度同步正确。
- **留证据**：把 reward 曲线 / loss / 同步开销记到 `docs/training_examples/`（建一个 ddp_2x1 条目），作为归档 done 的 live-gate 依据。

## 2. 边角项（非 2x1 目标的功能缺口）

- `vrl/ray/resources.py:187-191`（`trainer_default_auto`）与 `:394-409`（`_validate_trainer_device_count`）仍只特判 `fsdp`，未含 `ddp`。2 机×1 卡每 rank 恰好 1 本地 GPU、走单卡分支正确放行；但**每节点 >1 GPU 的 ddp 不会 resolve**。若将来要 N>1 卡/节点的 ddp，需把这两处的 fsdp 特判扩到 ddp。

## 3. gate

需真实 2 机硬件（NCCL 跨机），本单卡机无法验证。

## 相关
- [[SPRINT_symmetric_colocated_ddp]]（`done/`，父 sprint，代码已全落地）
- [[SPRINT_wan_2_2_proof_run]]（同类"实现已完成、待 proof run"的姊妹 sprint）
