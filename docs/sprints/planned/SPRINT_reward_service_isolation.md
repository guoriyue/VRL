# SPRINT：reward 全部走独立服务进程（GIL 隔离）

状态：**P1 已落地（7f21d66d，2026-09-14）；P2–P5 planned**。用户决策：不论难度，reward 一律与 trainer 进程隔离。
来源实验：`docs/sprints/SPRINT_four_l40s_execution.md` "Prefetch / reward placement /
compile: final five-arm table"。

## 0. 决策与证据

- SD3.5 512px 的 replay 是发射绑定的（eager 下 evaluate 阶段 GPU0 SM 44%）。
- continuous 调度下，进程内 PaddleOCR 与 trainer 主线程争 GIL：evaluate +29 s、
  backward +32 s、GPU 占用不变；同一 reward 搬到独立进程后 evaluate 回到 strict 水平，
  每 epoch 509 → 398 s（1.28×）。
- 用户决策：**隔离是默认，不是可选**。所有 reward 在训练时都运行在 trainer 之外的进程，
  哪怕这意味着要让服务进程参与 GPU 时分租约。

## 1. 现状（改前必须成立的事实，改后逐条失效）

| 事实 | 位置 |
|---|---|
| 24 个 reward function：13 个 `DiskArtifactRewardFunction`（HTTP-capable），7 个 `InferenceRewardFunction`（纯内存），3 个 `CumemRewardFunction`（GPU 进程内 + CuMem 池），1 个 `RewardFunction`（geneval） | `vrl/rewards/functions/*.py` |
| 传输只有两种：`in_process`（默认）、`http`（操作者手工起 `vrl-reward-service`） | `vrl/config/reward_inference.py` |
| HTTP 服务被定义为"独占加速器"：`generation_overlap_safe` 默认 false；服务禁止 `sleep_offload` | `vrl/rewards/service/server.py:137` |
| 服务协议只有 `/live /ready /info /score DELETE /requests`，没有 park/wake | `server.py:278-281` |
| 10 个配方让 reward 与 trainer 时分同一张卡（phase lease：rollout park → reward wake/score/sleep → trainer restore），靠进程内 `MemoryParkingScorer` | `vrl/rollouts/collector/core.py:200-260`，`gpu_pool: trainer` 的 preset |
| 资源解析：全 HTTP 时直接丢掉 reward 资源（"External services own their accelerator"） | `vrl/ray/resources.py:393-407` |
| 磁盘 artifact：image/video 张量 `torch.save` 为 `.pt`（fp32 原尺寸），视频可 mp4 | `vrl/rewards/artifacts.py` |

## 2. 目标结构

```
trainer 进程（driver）
   │  HttpRewardScorer（唯一的训练期 scorer；含 park/wake 客户端）
   ▼
managed reward service 进程（每个 reward 组件一个；trainer 拉起、等 ready、结束时回收）
   │  InProcessRewardScorer + CuMem 池（原来在 driver 里的那套，原样搬进服务）
   ▼
GPU（独占，或与 trainer/rollout 时分：由租约通过 /park /wake 协调）
```

- `reward.inference.<name>.kind` 词汇：`service`（默认，trainer 托管拉起）与 `http`
  （外部已运行的服务，今日语义）。`in_process` 从在线训练路径删除，只保留给评测脚本
  和测试（`InProcessRewardScorer` 本身不删，它是服务内部的执行器）。
- artifact 走 tmpfs：托管服务默认 `artifact_dir` 在 `/dev/shm/vrl_reward/<run>/`
  （本机 187 GB），避免视频张量落盘的 IO；`.pt` 张量以 uint8 存（480×832×81 帧 fp32 是
  388 MB/样本，uint8 是 97 MB；服务侧 `as_media()` 还原）。

## 3. 阶段

### P1 托管服务传输（`kind: service`）— 已落地 7f21d66d
实际形状与计划的差异：`port`/`artifact_root` 没有做成配置键——端口由 driver 选空闲回环端口，
artifact 根固定为 `${trainer.output_dir}/reward_artifacts/<component>`（run 作用域，也是服务被
允许读取的唯一目录；tmpfs 留给 P2 视频张量再定）。托管服务的 YAML/日志落在
`${trainer.output_dir}/reward_artifacts/reward_service.<component>.{yaml,log}`。
CPU 端到端验证过（preflight 1 s 拉起 PaddleOCR 服务，打分与进程内一致，shutdown 无残留进程）；
GPU 验收（3x1 preset + `reward.inference.ocr.kind=service`，期望 ≈398 s/epoch）等 GPU 空闲。
- `RewardInferenceConfig`：新增 `kind: service`，字段 `port: auto|int`、
  `artifact_root`（默认 tmpfs）。`http` 保持不变。
- 新模块 `vrl/rewards/service/managed.py`：从 reward 组件的 `worker_config` +
  `model_factory` + 解析后的设备生成服务 YAML，`subprocess.Popen(vrl-reward-service)`，
  轮询 `/ready`，把 `HttpRewardScorer` 交给 `MultiReward`，`shutdown()` 时 SIGTERM →
  等待 → SIGKILL；子进程的 stdout/stderr 落到 `${trainer.output_dir}/reward_service.<name>.log`。
- 注册表（`vrl/rewards/functions/registry.py`）：`service` 与 `http` 走同一注入路径；
  `service` 允许 `device`/`worker_config`（它们是要转交给服务的），`http` 仍拒绝。
- 验收：SD3.5 3x1 preset 不写任何 `+reward=ocr_http`，默认即走托管 OCR 服务；
  epoch 墙钟 ≈ 398 s（复现五 arm 表第三行）。

### P2 服务参与 GPU 时分租约
- 服务端：解除 `sleep_offload` 禁令（仅 `kind: service`），新增 `/park`（sleep 池 +
  释放缓存 + 用 `validate_parking_residual` 自检，返回残留字节）和 `/wake`；
  `/park` 幂等，失败返回 5xx 且保留可重试语义（对齐 `core.py` 的 phase-final gate）。
- 客户端：`HttpRewardScorer` 实现 `MemoryParkingScorer`（`requires_memory_parking`、
  `activate`、`park_memory`），超时走 `OperationDeadline`。
- 资源解析：`kind: service` 且 `reward.device: trainer|gpu` 时**保留** reward 资源和
  三条 sharing facts（现在只对 in_process 保留），让 lifecycle 仍规划 handoff；服务
  进程用的是物理卡号，需要和 rank-local `CUDA_VISIBLE_DEVICES` 重映射对齐
  （`vrl/scripts/train.py` 的 narrow 逻辑）：托管服务拉起时显式传物理 id。
- 残留检查：driver 的 `VRL_CUDA_RESIDUAL_BYTES_LIMIT` 现在只看自己进程；改为
  `/park` 响应里带服务进程的 `gpu_process_used_bytes`，driver 汇总后再放 trainer restore。
- 验收（最难的一条）：`online_grpo_hpsv3_fsdp_4x_l40s` 2 个 update，HPSv3 作为托管
  服务与每个 rank 时分同一张卡，metrics 与 2026-09-11 的 smoke 一致，无 Xid、无残留超限。

### P3 剩余 11 个 reward 变为服务可用
- 7 个 `InferenceRewardFunction`（nsfw_safety、wd_tagger、motion_dynamics、
  image_sharpness、target_dino_similarity、grounded_ocr、codex_image_qa）和 3 个
  `CumemRewardFunction`（aesthetic、pickscore、geneval_owl）改为
  `DiskArtifactRewardFunction` 声明块（照 `countgd.py` / 本次 `ocr.py` 的形状）；它们的
  model 类已经是 `__call__(artifact)` 契约，只需 `worker_config` 构造函数。
- `geneval`（规则型 `RewardFunction`）：读代码后决定是包成服务还是留在 driver（无
  模型、无 GIL 争用的纯 Python 规则可以豁免，需在文档写明理由）。
- 每个加一份 `vrl/config/reward_service/<name>.yaml` 模板（托管模式不需要它，但外部
  `http` 模式和运维排查需要）。

### P4 删除在线训练的进程内传输
- 注册表：在线 recipe 中 `kind: in_process` 报错，提示改 `service`；默认值改为
  `service`；31 个 reward preset 去掉 `device`/`sleep_offload` 之类进程内键（它们进
  `worker_config` 转交服务）。
- `vrl/scripts/common/online.py` 的 reward 构建路径只剩托管/外部服务；
  `RewardFunctionRuntime` 对 `MemoryParkingScorer` 的分发改为面向 `HttpRewardScorer`。
- 文档：`docs/CONFIGURATION.md`、`presets/reward/README.md`、`SPRINT_reward_service.md`
  的"两种部署"改为"服务是唯一部署，托管/外部两种拉起方式"。

### P5 全量验收
- 五 arm 表重跑（SD3.5 continuous + 托管 OCR），确认 398 s 量级不回退。
- Wan HPSv3 四卡时分（P2 验收）+ Cosmos DDP 2x1 Kling（`reward: gpu` 独占卡）各一次 smoke。
- `make verify` 绿；`tests/rewards` 里的进程内测试改为直接测服务内的
  `InProcessRewardScorer`，端到端用 aiohttp 测试服务（已有 `tests/rewards/service/`）。

## 4. 非目标
- Ray object-store 传输（tmpfs 已消掉磁盘 IO，跨节点再议）。
- 跨节点服务调度、服务的多副本/负载均衡。
- 把 rollout 生成也搬出 driver（它已经在 Ray actor 里，不在本 sprint 范围）。

## 5. 风险
- P2 是真正的架构改动：租约状态机跨进程后，超时和失败恢复的每条路径都要重新走一遍
  （`core.py` 的 phase-final gate、`RewardCleanupError`、残留检查）。先在 SD3.5 单卡
  colocated 上做 park/wake 往返测试，再上 Wan 四卡。
- 视频张量 reward 的 artifact 体积：uint8 + tmpfs 后单样本 97 MB，24 样本/rank/update
  ≈ 2.3 GB，可接受；如果哪个 reward 必须 fp32，单独标注并用 fp16。
- 每个 reward 一个子进程：4 rank 时分配方里每 rank 一个 HPSv3 服务（7B）常驻主机内存
  park 后约 17 GB × 4，在 372 GB 内；Wan 2.2 这类主机内存紧张的配方要重算预算。
