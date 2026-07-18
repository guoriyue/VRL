# SPRINT: 权重同步传输 seam —— object-store 之外加 NCCL 直传

Status: **PARKED**. Trigger: the first full-parameter large-model multi-GPU
training workload whose weight transport is material; do not start for LoRA-only
workloads.

## 背景

现状只有一种传输:`vrl/generation/ray/weight_sync.py:57` 把整个 state dict
`ray.put` 进 object store,所有 worker 共享一个 ObjectRef。对 LoRA(只传
adapter,MB 级)这是正确且简单的选择;对全参同步大模型(GB 级)则是
GPU→CPU 序列化→plasma→CPU 反序列化→GPU 三次多余拷贝。

slime 把传输做成菜单(NCCL 广播 / 磁盘 / object-store / delta 增量),按拓扑
选;cosmos-rl 全参走动态 NCCL group。两家一致:**传输是策略,不是常量**。
VRL 的 seam 已经存在——`GenerationWeightSync` 是 protocol
(`weight_sync.py:15`),只需加第二个实现。

## 范围

- 新增 `NCCLGenerationWeightSync`:trainer rank 0 与所有 rollout worker actor
  建 collective group(`ray.util.collective` 或手建 ProcessGroup,调研后择一,
  倾向 prior art:vLLM/slime 的做法),state dict 按 key 顺序逐张量 broadcast,
  GPU 直传不落 CPU。
- 选择逻辑:`build.use_lora`(或同步负载估计)决定默认传输;yaml 可显式覆盖
  (`model.weight_sync.transport: object_store | nccl`),读取走 `cfg_path`
  单一读取器,不加专用 reader。
- worker 侧 `update_weights` 拆出接收路径:object-store 路径保持原签名;NCCL
  路径 worker 参与集合通信后本地 load。
- 传输选择记录进启动日志(一行,含负载大小),便于事后核对。

## 验收标准

- 单测:传输选择逻辑(LoRA→object-store、全参→nccl、yaml 覆盖优先)纯函数可测。
- **真实验证(需 2+ GPU,或单卡双进程 gloo 降级)**:同一 state dict 经两种传输
  到达 worker 后逐位一致;NCCL 路径不出现 host 内存峰值(全程 GPU)。
- 既有 LoRA 训练路径行为零变化(默认不变)。
- 微基准:全参 state dict(≥2GB)两种传输的同步耗时对比,数字进 sprint 记录。

## 非目标

- delta/增量同步(slime 有;等全参同步真成瓶颈再评估)。
- 磁盘传输(checkpoint reload 已覆盖该场景)。
- 改变 LoRA 路径的默认行为。
- colocated CUDA-IPC 传输(sleep/wake 手递手已覆盖单卡共享场景)。

## 参考

- 现有 seam:`vrl/generation/ray/weight_sync.py`(protocol + object-store 实现)
- slime 权重同步四传输:https://deepwiki.com/THUDM/slime
- Ray collective:https://docs.ray.io/en/latest/ray-more-libs/ray-collective.html
