# 单卡 vs 多卡：支持与区分 — slime vs cosmos-rl（reading）

类型：cross-system 源码对照（2026-06-17）。file:line 相对各 repo root。
index：[slime_cosmos_study_index.md](slime_cosmos_study_index.md)。

## 共同结论（先记住）
**两家都：(1) 单 flag 切换、不自动探测 GPU 数;(2) 单卡退化成串行时间片,无真重叠;(3) 真 overlap 必须
多卡分离。** 区分的是 flag 名字和切换后换掉的那套机制。

## slime —— `--colocate`

- flag:`--colocate`(`utils/arguments.py:70-74`,默认 False)。
- **开启(单卡/共卡)**:`arguments.py:1745-1756` **强制** 4 件事:
  1. `offload_train=True`、2. `offload_rollout=True`、3. `rollout_num_gpus` 覆盖成
  `actor_num_nodes × actor_num_gpus_per_node`(无视用户输入)、4. 若用户给过别的值则告警。
  - 一个 PACK placement group,`rollout_offset=0`(`placement_group.py:89-91`),MegatronActor 与
    SGLangEngine 钉同 bundle;**分数 GPU**(actor `num_gpus=0.4`、engine `num_gpus=0.2`,
    `placement_group.py:117` / `rollout.py:99`)让 Ray 把两者排到同一物理卡。
  - 从不同跑:`train.py` 每阶段 blocking,`torch_memory_saver.pause/resume` 在两者间分页 GPU。
  - 权重同步走 **CUDA-IPC**(`UpdateWeightFromTensor`,`actor.py:126-127`)——同卡,无 NCCL。
  - **`train_async.py:11` 直接 `assert not colocate`**——async 被禁。
- **关闭(多卡分离)**:rollout 拿独立 bundle(`rollout_offset = actor_nodes × gpus`,`placement_group.py:93-94`);
  权重同步走 **NCCL**(`UpdateWeightFromDistributed`,per-PP-rank broadcast group);解锁 `train_async.py`。

## cosmos-rl —— `mode`

- flag:`config.mode`(`policy/config/__init__.py:1815`,`disaggregated | colocated | colocated_separated`)。
  入口 `policy_entry.py:55` `if mode=='colocated'` fork 到 `ColocatedRLControlWorker`,否则 disaggregated。
- **`colocated`(单卡)**:launcher 强制 `n_rollouts=0`(`launcher/utility.py:509-510`);policy+rollout 同进程
  (`rl_worker.py:59-75`);`CommandDispatcher` 用 **Python Queue 替 Redis**(`colocated/utils.py:22-49`,不起
  redis-server);**P2R 权重同步是指针交换**——`self.rollout.set_underlying_model(self.api_client.get_policy_model())`
  (`colocated/rollout_control.py:68`),**零拷贝零 NCCL**,R2R 只是版本号 +1;主循环串行(`rl_worker.py:132-211`)。
- **`disaggregated`(多卡,默认)**:每 replica 独立 torchrun(TP×PP×DP×CP);HTTP 自注册 controller,controller
  fork redis-server;P2R 用动态建的 NCCL communicator(`ParallelizedShardMapper`,`parallelism_map.py:915-946`),
  R2R 全局 mesh broadcast。
- **`colocated_separated`**:同卡不同进程,P2R 用 **ZMQ + CUDA-IPC** 替 NCCL。

## 对照

| | slime | cosmos-rl |
|---|---|---|
| 切换 flag | `--colocate` | `mode`(3 值) |
| 自动探测? | 否 | 否 |
| 单卡机制 | 同 bundle + 分数 GPU + offload 分页 | 同进程 + Python Queue + 指针交换 |
| 单卡权重同步 | CUDA-IPC(UpdateWeightFromTensor) | **指针交换**(set_underlying_model,零拷贝) |
| 多卡权重同步 | NCCL per-PP broadcast | NCCL shard communicator(P2R)+ mesh(R2R) |
| 单卡能 overlap? | 否(`train_async` 被 assert 禁) | 否(colocated 主循环串行) |
| 切换换掉什么 | 4 个 arg 默认 + placement + 同步后端 | worker 类 + 传输栈(Redis/NCCL ↔ Queue/指针) |

## 对我们
- 我们 continuous 也是单 flag(`mode=continuous`)+ 守卫禁单卡(`schedule.py:198-208`),同范式。
- **最值得抄:cosmos 单卡的指针交换权重同步**(`set_underlying_model`)。我们现在单卡也走
  `CPU state_dict → ray.put → worker load`(`vrl/generation/ray/weight_sync.py:50-59`)——当 rollout 和
  train 共享模型对象时,这是纯浪费,指针交换可以零拷贝。
- 别为单卡造重叠:两家都证明单卡 = 串行,工程精力放多卡分离。
