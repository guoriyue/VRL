# SPRINT: Cross-Node Rollout

## 0. Core Decision

让 trainer 和 rollout 跑在**不同机器**上：单进程 trainer 占 Ray driver/head 节点的 GPU，rollout/generation 的 Ray actor 调度到**集群里另一个节点**的 GPU。

新增一个**可选开关** `distributed.resources.cross_node: true`。开关为 `false` 时单机 / 单机多卡行为**逐字节不变**——本 sprint 不动任何现有路径，只加旁路。

**边界决定：VRL 不管理多机 SSH / provisioning / Ray cluster lifecycle。** VRL 只消费一个已经存在的 Ray 集群，通过 `RAY_ADDRESS` 或 `ray.init(address=...)` 连接；多机启动、SSH key、镜像同步、端口、安全组、KubeRay / Anyscale / Ray cluster launcher 都属于外部部署层。

这是 `SPRINT_multi_gpu_training.md` 提到「目前只能把 rollout 分到同机多张卡，不能跨机」的那块补齐，但范围更窄：只做 **rollout 跨机**，不碰 trainer 侧的 DDP/FSDP。

动机硬件场景：两台单卡实例（都是 L4，同一 VPC），希望一台做训练、另一台做采样，而不是去开一台多卡机。

推荐部署形态：

- HEAD / trainer 节点运行训练脚本，trainer 直接使用本机 CUDA GPU。
- Ray head 进程在 HEAD 上启动，但**推荐显式 `--num-gpus=0`**，让 trainer GPU 不进入 Ray 调度池。
- rollout worker 节点用 `ray start --address=<head>:6379 --num-gpus=<worker GPUs>` 加入集群。
- VRL 配置只声明角色预算，例如 `trainer.num_gpus=1`、`rollout.num_gpus=<worker GPU count>`、`rollout.num_workers=<worker GPU count>`。

这同样覆盖 2 台以上机器：如果是 1 个 trainer/head + N 个单卡 rollout worker，就设 `rollout.num_gpus=N`、`rollout.num_workers=N`；机器清单仍然在 Ray / infra 层，不写进 VRL 训练配置。

**唯一必需的连接输入**：用户只需把 head 地址通过 `RAY_ADDRESS=<head>:6379` 交给 driver。driver 跑在 head 上时 `ray.init()` 自动发现、可省略；跑在别处时必填。worker IP / 机器清单 / SSH key 都不需要——worker 自己拨入 head 报到。**VRL 不新增任何集群地址配置字段**（不加 `distributed.ray.address`）。

## 1. Current Code Reality

资源层把所有 GPU 当成**单机本地 CUDA 序号**。在两节点集群（head 1 卡 + worker 1 卡）上用 `ray_rollout` 预设启动会直接崩：

```
ValueError: Not enough non-overlapping rollout GPUs:
  requested=1, available=0, trainer=[0], visible=[0]
```

根因链：

1. `vrl/ray/resources.py` `_auto_visible_cuda_devices()`（~L827-837）返回 `range(torch.cuda.device_count())`——**只数 head 本地的卡**（=1）。
2. `resolve_distributed_resources()` / `_resolve_rollout_devices()`（L84-134、L393-409）把 trainer/rollout/reward 当成**同一个扁平的本地序号集合**，trainer 占掉 `[0]` 后 rollout 池为空 → 抛错。
3. 即便强行让它以为有 2 张卡（`visible_devices=[0,1]` → rollout=`[1]`），`vrl/ray/placement.py` `validate_actor_gpu_ids()`（L52-84）会**硬断言**每个 actor 的 `gpu_ids ⊆ {1}`；但远端节点上 actor 的本地序号是 `0`，于是报 `assigned GPU ids [0], outside resolved devices [1]`。
4. `vrl/generation/ray/config.py` `_validate_driver_cuda_ownership()`（L137-173）拿 driver 的 `cuda:0` 和 `rollout_devices` 做交集判断——跨机时这个语义无意义。

### 已核实的可行性结论

- **Ray 放置机制支持跨机，但 `SPREAD` 是 best-effort，不等于强隔离。** `vrl/generation/ray/placement.py` 的 rollout bundle 只请求 `{CPU, GPU}` 数量，不绑定节点或序号。Ray 会自己挑有空闲 GPU 的节点，并按节点重映射 `CUDA_VISIBLE_DEVICES`；但如果 head 的 GPU 也被 Ray 暴露，rollout 仍可能落回 head。
- **解析早于 `ray.init()`。** `resolve_distributed_resources` 在 `vrl/scripts/common/online.py:75` 调用，launcher 在 L129 才 `ray.init()`。因此解析期**不能**查 `ray.cluster_resources()`——必须用显式数量推算 GPU 预算。
- **不要依赖当前 unpinned trainer reservation bundle。** `requires_trainer_reservation` 当前只会在 placement group 里加一个 `{CPU, GPU}` bundle，它没有绑定 driver/head 节点，不能证明它吃掉的是 head GPU。跨机推荐改成外部 Ray head `--num-gpus=0`；如果后续要兼容“head Ray 暴露 GPU”的集群，也必须在 `ray.init()` 后创建 driver-node-pinned reservation actor，而不是继续用未绑定 bundle。
- **worker metadata 已带 `node_ip`。** `launcher.py:103`、`runtime.py:104` 已经把 `node_ip` 放进 metadata，所以校验可以改成「`(node_ip, gpu_id)` 组合唯一」，保留安全网而非直接删校验。
- **跨机安全网必须校验 node，而不只是 gpu id。** 放宽序号校验后还要断言 rollout worker 不在 driver/head `node_ip` 上；否则“远端本地 gpu id = 0”这个 case 会通过，但 rollout 可能仍跑在 trainer 节点。

## 2. Consumer Map（哪些字段被当成「真实本地序号」用）

主要破点仍是三处把设备元组当成真实本地序号；另外还有两个跨机边界需要显式约束：trainer reservation 不能再按未绑定节点的 bundle 理解，reward GPU 路径本 sprint 不启用。

| 位置 | 字段 | 当前用法 | 跨机判定 |
|---|---|---|---|
| `placement.py:52` `validate_actor_gpu_ids` | `rollout_devices` 作 `expected_gpu_ids` | 断言 actor 本地 `gpu_ids ⊆ expected` 且 `actual == expected` | **破** |
| `generation/ray/config.py:153` | `rollout_devices` | `driver_cuda_devices & set(rollout_devices)` | **破**（语义无意义） |
| `resources.py:827-837` `_auto_visible_cuda_devices` | visible 池 | 只数本地卡 → rollout 池空 | **破**（启动前即崩） |
| `resources.py:187` `requires_trainer_reservation` | 计数/布尔 | 决定是否加 trainer 预留 bundle | **跨机不能按原语义使用**（bundle 未 pin head） |
| `rollout_num_workers` / `rollout_gpus_per_worker` / `ray_total_bundles` 等 | 计数 | bundle 数量与 GPU 请求量 | 安全 |
| reward 相关（`_reward_gpu_reservation_count` 用 `visible_devices.index`；`RayActorMethodRuntime.expected_gpu_ids` 决定是否建 placement group） | 序号位置 | 仅当 reward 用卡时触发 | **本 sprint 强制 cross-node preset reward GPU=0**；跨机 reward 留作后续 |

## 3. Design / 改动点

### 3.1 开关接线 — `vrl/ray/resources.py`
- `DistributedResourceConfig`（L35-51）加 `cross_node: bool = False`。
- `ResolvedDistributedResources`（L54-78）加 `cross_node: bool`，并在 return（L203-225）填充。
- `_distributed_resource_config_from_cfg`（L289-323）读取 `cross_node`。
- `format_distributed_resource_plan` 打印 `cross_node=True/False`，避免日志里看不出 rollout device 是真实本地序号还是预算 token。

### 3.2 让 rollout 预算无需活集群即可满足 — `vrl/ray/resources.py`
- `cross_node` 且 `visible_devices == auto` 时，用**显式角色数量**合成 visible 池：`range(trainer.num_gpus + rollout.num_gpus + reward.num_gpus)`，替代本地计数。开关传入 `_resolve_visible_devices`（调用点 L93）。
- 跨机下要求 `trainer.num_gpus` / `rollout.num_gpus` 为显式整数（为 `auto` 时报清晰错误，因为解析期拿不到集群信息）。
- 预设结果：`visible=[0,1]`、`trainer_devices=(0,)`、`rollout_devices=(1,)`。`trainer_torch_device` 仍是 head 本地 `cuda:0`（对）；`rollout_devices` 退化为「预算 token」，由 3.3 保证没人再当真实远端序号用。
- 跨机默认不创建原有 placement-group trainer reservation bundle：推荐通过 Ray head `--num-gpus=0` 从源头保证 rollout 不可调度到 trainer GPU。
- 如果后续必须支持“head Ray 暴露 GPU”的集群，单独加显式开关（例如 `distributed.resources.reserve_driver_gpu_in_ray: true`），并在 `ray.init()` 后用 driver-node-pinned actor 占住 head GPU；不要复用当前未绑定节点的 bundle。
- cross_node 下 `requires_trainer_reservation` **强制为 `False`**（不再加未 pin 的预留 bundle）；trainer 卡的隔离改由 head `--num-gpus=0` + §3.4 preflight 保证。

### 3.3 跨机下放宽两处序号校验
- `vrl/ray/placement.py` `validate_actor_gpu_ids`（L52-84）：加 cross-node 分支（新参数或姊妹函数）。用 metadata 的 `node_ip`，把严格子集/相等换成：
  - 每个 worker 至少有 1 个 gpu id。
  - 唯一 `(node_ip, gpu_id)` 组合数 == worker 数，防两 worker 抢同卡。
  - 每个 rollout worker 的 `node_ip != driver_node_ip`，防 rollout 落回 trainer/head。
- `vrl/generation/ray/launcher.py` `_validate_worker_gpu_ids`（L236-248）：跨机走放宽路径，并把 `driver_node_ip` 传入校验。
- `vrl/generation/ray/config.py` `_validate_driver_cuda_ownership`（L137-173）：`resources.cross_node` 为真时在 L144 后提前 return。

### 3.4 Ray cluster preflight — `vrl/generation/ray/launcher.py`
- 在 `ray.init()` 后、placement group 创建前，若 `resources.cross_node` 为真，读取 Ray 活节点信息：
  - 至少存在一个非 driver/head 的 alive node。
  - 非 driver/head 节点的 Ray-visible GPU 总数 >= `resources.rollout_num_gpus`。
  - 如果 driver/head node 仍暴露 Ray-visible GPU，打印明确 warning：推荐用 `ray start --head --num-gpus=0`；没有显式 reservation fallback 时直接 fail-fast，避免 rollout 静默抢 trainer GPU。
- 这个 preflight 是 runtime 校验，不参与解析期预算，因为解析仍早于 `ray.init()`。

### 3.5 新预设 — `configs/base/distributed/ray_rollout_cross_node.yaml`
镜像 `ray_rollout.yaml`，加 `cross_node: true`、显式 `trainer.num_gpus: 1`、`rollout.num_gpus: 1`、`rollout.num_workers: 1`、`placement_strategy: SPREAD`，reward 保持 0 卡。**原 `ray_rollout.yaml` 不动**，保证单机零回归。

2+ rollout worker 用 override 即可：

```
/base/distributed=ray_rollout_cross_node \
distributed.resources.rollout.num_gpus=3 \
distributed.resources.rollout.num_workers=3
```

### 3.6 Reward 路径 — 暂不改（Non-Goal）
cross-node preset 强制 reward GPU=0，相关 consumer 不触发。**不要**用清空 `expected_gpu_ids` 绕过校验——那会让 `runtime.py:117` 连 placement group 都不建。跨机 reward 池留作后续：届时 `_reward_gpu_reservation_count`（`resources.py:717-726`，用 `visible_devices.index(...)`）的位置数学，以及 reward actor 的 node 校验都需重审。

### 3.7 多机配置边界 — 不在 VRL 里做 SSH
不新增以下配置：

```
distributed.cluster.hosts
distributed.cluster.ssh_user
distributed.cluster.ssh_key
distributed.cluster.setup_commands
```

原因：

- Ray 已经有三条成熟路径：手动 `ray start`、VM cluster launcher (`ray up`)、KubeRay / managed Ray。VRL 再做一套 SSH orchestration 会重复 Ray 和 infra 工具，而且会把云厂商、端口、安全组、镜像同步、密钥管理都拉进训练代码。
- 很多生产环境没有 SSH：Kubernetes、Anyscale、Slurm wrapper、托管 Ray Job 都是由平台创建 Ray 集群，再把训练脚本提交到 head。
- VRL 真正需要的是 Ray 可见的资源形状，不是机器登录方式。训练配置应保持可移植：同一份 `ray_rollout_cross_node.yaml` 能跑在手动 VM、Ray launcher、KubeRay 上。

可以在 docs 里提供部署 runbook / 示例 cluster yaml，但不要让训练运行时依赖 SSH。

### 3.8 部署 / 性能注意
- worker 节点需自带 base model 权重 + 与 head 一致的 venv（Ray 只分发代码，不分发依赖/权重）。
- LoRA 权重同步（`sync_trainable_state: lora_only`）现在走节点间网络；体积小，无 NVLink/IB 也可接受。

## 4. Tests
- `tests/ray/test_resources.py`：现有用例全绿（`cross_node` 默认 false，新增字段 additive）。新增 `test_cross_node_rollout_satisfies_budget_from_explicit_counts`（head=1 trainer + 显式 rollout=1 + cross_node=true → trainer=(0,)、rollout=(1,)、num_workers=1、不报错；即原崩溃的回归测试）。断言 cross-node 默认不依赖未 pin 的 trainer reservation bundle。
- placement 测试：放宽校验下，两 worker 在不同 `node_ip` 上本地 id 均为 0 应通过；同节点重复 GPU 应拒绝；worker 落在 `driver_node_ip` 应拒绝。
- `tests/generation/ray/test_rollout_launcher.py` / `test_runtime_config.py`：加 cross-node 变体，断言跳过严格校验 / driver overlap。
- launcher preflight 测试：mock Ray nodes，覆盖非 driver GPU 足够、不足、driver node 暴露 GPU 的 fail-fast。
- `tests/config/test_load_all_experiments.py`：确认新预设可加载/组合。

## 5. Verification（端到端）
1. 单测：`pytest tests/ray/test_resources.py tests/generation/ray/ -q`。
2. 实跑（需先在两台搭好 2 节点 Ray 集群，见附录）：
   ```
   RAY_ADDRESS=172.31.27.241:6379 python -m vrl.scripts.train \
     --config experiment/diffusion/sd3_5/online_grpo_ocr \
     /base/distributed=ray_rollout_cross_node
   ```
3. 确认：解析不再崩；远端 `172.31.26.180` 上 `nvidia-smi` 在采样阶段看到 rollout 的 python 进程占 GPU；head 的卡跑 trainer；首个训练步打出 reward/loss。
4. 清理：两台各 `ray stop`。

## 6. Critical Files
- `vrl/ray/resources.py`（开关 + 合成 visible 池）
- `vrl/ray/placement.py`（`validate_actor_gpu_ids` 放宽）
- `vrl/generation/ray/placement.py`（cross-node 下不再依赖未 pin 的 trainer reservation bundle）
- `vrl/generation/ray/launcher.py`（`_validate_worker_gpu_ids`）
- `vrl/generation/ray/config.py`（`_validate_driver_cuda_ownership`）
- `configs/base/distributed/ray_rollout_cross_node.yaml`（新建）
- `README.md` 或 `docs/deployment/ray_cross_node_rollout.md`（外部 Ray 集群启动方式）
- 上述测试文件

## Appendix A: 测试环境拓扑（本 sprint 验证用）
- HEAD / trainer：私有 IP `172.31.27.241`，安全组 `sg-0473be3f448a2d1f5`（launch-wizard-3）
- WORKER / rollout：私有 IP `172.31.26.180`，安全组 `sg-05802358e1057f90e`（launch-wizard-2）
- 同区域 us-east-2、同 VPC `vpc-0fb4075f06a105b8b`；两台 Python 3.12.3 / Ray 2.55.1，代码与 SD3.5 缓存一致。
- 跨机前提（已就绪）：两个安全组互相放行 All TCP。
- 推荐起集群：
  ```
  # HEAD / trainer: trainer uses local CUDA, Ray head advertises no GPU.
  ray start --head --node-ip-address=172.31.27.241 --port=6379 --num-gpus=0

  # WORKER / rollout: Ray advertises rollout GPU.
  ray start --address=172.31.27.241:6379 --node-ip-address=172.31.26.180 --num-gpus=1
  ```
- 如果后续扩到更多 rollout workers，每台 worker 重复 `ray start --address=... --num-gpus=<count>`，VRL 只改 `distributed.resources.rollout.num_gpus` / `num_workers`。

## Appendix B: SSH / 多机配置答案
这个 sprint 不要求 VRL 拥有 SSH access，也不应该把 SSH 作为训练代码的前提。

可选部署方式：

1. 手动 VM：操作者用 SSH 进入各机器执行 `ray start`。这是最直接的两台 EC2 验证方式，但 SSH 只属于部署动作，不进入 VRL 配置。
2. Ray VM cluster launcher：用 Ray 官方 cluster yaml 配置机器、SSH、file mounts、setup commands，然后 `ray up` / `ray submit`。这种方式仍可能用 SSH，但由 Ray 管，不由 VRL 管。
3. KubeRay / managed Ray：没有 SSH；平台根据 `RayCluster` / `RayJob` 创建 head 和 workers，训练脚本只连接 Ray 地址或直接作为 RayJob 运行。

VRL 应该提供的是：

- 一个 `ray_rollout_cross_node.yaml` 预设。
- 清晰的资源 override 例子。
- runtime preflight，发现 Ray 集群没给非 driver 节点足够 GPU 时 fail fast。
- 一份 deployment doc/runbook，说明如何用手动 VM、Ray launcher 或 KubeRay 准备 Ray 集群。

VRL 不应该提供的是：

- SSH 主机清单解析。
- 私钥路径管理。
- 远程安装依赖或同步代码。
- 自动 `ray start` / `ray stop` orchestration。
